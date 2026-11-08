//! Pipeline model step: `holdout_fill` (shuffles from a NumPy-compatible `default_rng(0)`
//! stream, [`super::rng`]), `comp_table`, `model_step`, `report_text`.

use ts_dsp::ValueError;
use ts_dsp::ndimage::binary_erosion;
use ts_dsp::stats::mean;
use ts_io::json::Value;
use ts_io::{fmt_fixed, repr_shortest, round_dec};

use super::aided::{AidedOpts, cabs, track_aided};
use super::gaps::{model_fill, pchip_fill};
use super::rng::default_rng;
use super::{Amp, Comp, FitOpts, Hier, HierOpts, IdentOpts, Model, explained, fit, fit_hier};
use crate::analyze::gap_fill;
use crate::curve::{Curve, MetaValue};
use crate::hetdec::Channel;
use crate::physics::{self, Amps, ConstrainInfo, Inference};

type Res<T> = Result<T, ValueError>;

/// Aided residual band [Hz] (`B_DEFAULT`).
pub const B_DEFAULT: f64 = 20.0;
/// L3 amplitude knot spacing [s] (`TA_DEFAULT`).
pub const TA_DEFAULT: f64 = 5.0;
pub const MIN_EV: f64 = 0.70;
/// Model step skipped below this file length [s] ...
pub const MODEL_MIN_S: f64 = 60.0;
/// ... or this pass-1 valid share.
pub const MODEL_MIN_VALID: f64 = 0.2;

/// `ident_kw(dur)`: identification window scaled to the file.
pub fn ident_kw(dur: f64) -> IdentOpts {
    IdentOpts::for_duration(dur)
}

// ------------------------------------------------------------------ holdout_fill

/// `holdout_fill` result: number of held-out stretches and, for `n >= 5`, the RMS of
/// `model_fill`, `analyze.gap_fill` and `pchip_fill` inside them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Holdout {
    pub n: usize,
    /// `(model, analyze, pchip)`
    pub rms: Option<(f64, f64, f64)>,
}

impl Holdout {
    pub fn model(&self) -> f64 {
        self.rms.map_or(f64::INFINITY, |r| r.0)
    }
    pub fn analyze(&self) -> f64 {
        self.rms.map_or(f64::INFINITY, |r| r.1)
    }
    pub fn to_value(&self) -> Value {
        let mut v = Value::obj().with("n", self.n);
        if let Some((m, a, p)) = self.rms {
            v.push("model", m);
            v.push("analyze", a);
            v.push("pchip", p);
        }
        v
    }
}

/// `(stretches [s, s + n) in choice order, taken mask)`.
pub type HoldoutSpans = (Vec<(usize, usize)>, Vec<bool>);

/// Hold-out stretches chosen by `holdout_fill` (index ranges `[s, s + n)` in choice order)
/// and the taken mask.
pub fn holdout_spans(
    t: &[f64],
    v: &[bool],
    lens: &[f64],
    n_per: usize,
    seed: u128,
) -> Res<HoldoutSpans> {
    let nt = t.len();
    let dt = t[1] - t[0];
    let mut rng = default_rng(seed);
    let ok = binary_erosion(v, (1.0 / dt) as usize * 2 + 1)?;
    let mut taken = vec![false; nt];
    let mut idx = Vec::new();
    let mut ls = lens.to_vec();
    // sorted(lens, reverse=True) (stable; ties keep their order)
    ls.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cs = vec![0i64; nt + 1];
    for i in 0..nt {
        cs[i + 1] = cs[i] + ok[i] as i64;
    }
    let g = (8.0 / dt) as usize;
    for l in ls {
        let n = (l / dt) as usize;
        if n == 0 || n > nt {
            // cs[n:] - cs[:-n]: numpy gives an empty candidate set here (n == 0: cs[:-0] = [])
            let mut none: Vec<i64> = vec![];
            rng.shuffle(&mut none);
            continue;
        }
        let mut cand: Vec<i64> = (0..=nt - n)
            .filter(|&s| cs[s + n] - cs[s] == n as i64)
            .map(|s| s as i64)
            .collect();
        rng.shuffle(&mut cand);
        let mut k = 0;
        for &s in &cand {
            let s = s as usize;
            let (a, b) = (s.saturating_sub(g), nt.min(s + n + g));
            if k < n_per && !taken[a..b].iter().any(|&x| x) {
                taken[s..s + n].iter_mut().for_each(|x| *x = true);
                idx.push((s, s + n));
                k += 1;
            }
        }
    }
    Ok((idx, taken))
}

/// `holdout_fill(t, r, v, comps, Ta, lens, n_per, seed)`: mask valid stretches of L s
/// (≥ 1 s valid margins, ≥ 8 s apart, random order from `default_rng(seed)`), refit L3 without
/// them, RMS vs the measurement inside the masks for model / analyze / PCHIP fills.
#[allow(clippy::too_many_arguments)]
pub fn holdout_fill(
    t: &[f64],
    r: &[f64],
    v: &[bool],
    comps: &[Comp],
    ta: f64,
    lens: &[f64],
    n_per: usize,
    seed: u128,
) -> Res<Holdout> {
    let (idx, taken) = holdout_spans(t, v, lens, n_per, seed)?;
    let n = idx.len();
    if n < 5 {
        return Ok(Holdout { n, rms: None });
    }
    let vf: Vec<bool> = v.iter().zip(&taken).map(|(&a, &b)| a && !b).collect();
    let w: Vec<f64> = vf.iter().map(|&x| x as u8 as f64).collect();
    let m = fit(
        t,
        r,
        &w,
        comps,
        &FitOpts {
            level: 3,
            ta,
            ..FitOpts::default()
        },
    )?;
    let err = |p: &[f64]| -> f64 {
        let sq: Vec<f64> = idx
            .iter()
            .flat_map(|&(a, b)| a..b)
            .map(|i| {
                let d = p[i] - r[i];
                d * d
            })
            .collect();
        mean(&sq).sqrt()
    };
    let pm = model_fill(t, r, &vf, &m, 0.1);
    let pa = gap_fill(t, r, &vf)?;
    let pp = pchip_fill(t, r, &vf)?;
    Ok(Holdout {
        n,
        rms: Some((err(&pm), err(&pa), err(&pp))),
    })
}

/// [`holdout_fill`] with the defaults (lens 1 s and 2 s, 20 each, seed 0).
pub fn holdout_fill_default(
    t: &[f64],
    r: &[f64],
    v: &[bool],
    comps: &[Comp],
    ta: f64,
) -> Res<Holdout> {
    holdout_fill(t, r, v, comps, ta, &[1.0, 2.0], 20, 0)
}

// ------------------------------------------------------------------ comp_table

/// One `comp_table` row.
#[derive(Debug, Clone, PartialEq)]
pub struct CompRow {
    pub name: String,
    pub kind: &'static str,
    pub h: u32,
    pub f_start: f64,
    pub f_end: f64,
    pub law: [f64; 3],
    pub amp_l2_pct: f64,
    pub amp_l3_rms_pct: f64,
}

impl CompRow {
    pub fn to_value(&self) -> Value {
        Value::obj()
            .with("name", self.name.as_str())
            .with("kind", self.kind)
            .with("h", self.h as i64)
            .with("f_start", self.f_start)
            .with("f_end", self.f_end)
            .with("law", self.law.to_vec())
            .with("amp_l2_pct", self.amp_l2_pct)
            .with("amp_l3_rms_pct", self.amp_l3_rms_pct)
    }
}

fn amp_abs_l2(m: &Model, key: (usize, u32)) -> f64 {
    match m.amp.get(&key) {
        Some(Amp::L2(z)) => z.norm(), // |z| = hypot
        Some(Amp::L3(v)) => {
            // an L3 amplitude vector has no single magnitude; not produced by fit level 2
            v.iter().fold(0.0f64, |a, z| a.max(cabs(*z)))
        }
        None => 0.0,
    }
}

/// `comp_table(H, t, w)`: per component and harmonic: law, frequency span over the valid
/// time range, L2 amplitude, L3 amplitude RMS [%].
pub fn comp_table(comps: &[Comp], m2: &Model, m3: &Model, t: &[f64], w: &[f64]) -> Vec<CompRow> {
    let tv: Vec<f64> = if w.iter().any(|&x| x > 0.0) {
        t.iter()
            .zip(w)
            .filter(|&(_, &x)| x > 0.0)
            .map(|(&a, _)| a)
            .collect()
    } else {
        t.to_vec()
    };
    let tmin = tv.iter().copied().fold(f64::INFINITY, f64::min);
    let tmax = tv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut rows = Vec::new();
    for (k, c) in comps.iter().enumerate() {
        let (f0, f1) = (c.freq_at(tmin), c.freq_at(tmax));
        for &h in &c.harmonics {
            let a3: Vec<f64> = match m3.amp.get(&(k, h)) {
                Some(Amp::L3(v)) => v
                    .iter()
                    .map(|z| {
                        let a = cabs(*z);
                        a * a
                    })
                    .collect(),
                Some(Amp::L2(z)) => vec![z.norm() * z.norm()],
                None => vec![0.0],
            };
            rows.push(CompRow {
                name: c.name.clone(),
                kind: c.kind.as_str(),
                h,
                f_start: h as f64 * f0,
                f_end: h as f64 * f1,
                law: [c.a, c.b, c.beta],
                amp_l2_pct: amp_abs_l2(m2, (k, h)) * 100.0,
                amp_l3_rms_pct: mean(&a3).sqrt() * 100.0,
            });
        }
    }
    rows
}

// ------------------------------------------------------------------ model_step

/// `mode` of `model_step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelMode {
    Auto,
    On,
    Off,
}

impl ModelMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelMode::Auto => "auto",
            ModelMode::On => "on",
            ModelMode::Off => "off",
        }
    }
}

impl std::str::FromStr for ModelMode {
    type Err = ValueError;
    fn from_str(s: &str) -> Res<Self> {
        match s {
            "auto" => Ok(ModelMode::Auto),
            "on" => Ok(ModelMode::On),
            "off" => Ok(ModelMode::Off),
            _ => Err(ValueError(format!("unknown model mode {s:?}"))),
        }
    }
}

/// Keyword arguments of [`model_step`] (with their defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOpts {
    pub mode: ModelMode,
    pub min_ev: f64,
    /// aided residual band [Hz]
    pub b: f64,
    pub refine: usize,
    pub ta: f64,
    pub channel: Channel,
    /// playback tape speed [m/s] for the physics report
    pub v_phys: Option<f64>,
    /// also constrain the reel laws with `v_phys`
    pub v_given: bool,
}

impl Default for StepOpts {
    fn default() -> Self {
        StepOpts {
            mode: ModelMode::Auto,
            min_ev: MIN_EV,
            b: B_DEFAULT,
            refine: 2,
            ta: TA_DEFAULT,
            channel: Channel::Mid,
            v_phys: None,
            v_given: false,
        }
    }
}

/// The fit part of the report (present unless the step was skipped).
#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    pub ev: f64,
    pub ev_l2: f64,
    pub r0: f64,
    pub rejected: f64,
    pub n_comps: usize,
    pub comps: Vec<CompRow>,
}

/// [`model_step`] report (`rep`), keys in report order by [`Self::to_value`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModelReport {
    pub mode: ModelMode,
    pub min_ev: f64,
    pub constrain: Option<ConstrainInfo>,
    pub physics: Option<Inference>,
    pub fit: Option<FitReport>,
    /// `off`, `plain` or `aided`
    pub path: String,
    pub reason: String,
    pub holdout: Option<Holdout>,
    pub gapfill: Option<String>,
    /// aided path: `(aided_B, refine)`
    pub aided: Option<(f64, usize)>,
}

impl ModelReport {
    /// The dict as `json.dump(rep)` sees it.
    pub fn to_value(&self) -> Value {
        let mut v = Value::obj()
            .with("mode", self.mode.as_str())
            .with("min_ev", self.min_ev);
        if let Some(c) = &self.constrain {
            v.push("constrain", c.to_value());
        }
        if let Some(p) = &self.physics {
            v.push("physics", p.to_value());
        }
        if let Some(f) = &self.fit {
            v.push("ev", f.ev);
            v.push("ev_l2", f.ev_l2);
            v.push("r0", f.r0);
            v.push("rejected", f.rejected);
            v.push("n_comps", f.n_comps);
            v.push(
                "comps",
                Value::Arr(f.comps.iter().map(CompRow::to_value).collect()),
            );
        }
        v.push("path", self.path.as_str());
        v.push("reason", self.reason.as_str());
        if let Some((b, refine)) = self.aided {
            v.push("gapfill", self.gapfill.clone().unwrap_or_default());
            v.push("aided_B", b);
            v.push("refine", refine);
        } else {
            if let Some(h) = &self.holdout {
                v.push("holdout", h.to_value());
            }
            if let Some(g) = &self.gapfill {
                v.push("gapfill", g.as_str());
            }
        }
        v
    }
}

/// Fit results of [`model_step`] (per-level fits incl. `m_final`).
#[derive(Debug, Clone)]
pub struct StepFit {
    pub hier: Hier,
    /// aided path: the signal-refined model
    pub m_final: Option<Model>,
}

impl StepFit {
    /// `H.get("m_final", H["m3"])`.
    pub fn final_model(&self) -> &Model {
        self.m_final.as_ref().unwrap_or(&self.hier.m3)
    }
}

fn meta_f64(v: &MetaValue) -> Option<f64> {
    match v {
        MetaValue::Float(x) => Some(*x),
        MetaValue::Int(i) => Some(*i as f64),
        MetaValue::Bool(b) => Some(*b as u8 as f64),
        MetaValue::Str(s) => s.trim().parse().ok(),
    }
}

/// `model_step(x, sr, ref_hz, c, mode, min_ev, B, refine, Ta, channel, v_phys, v_given)`:
/// fit L1–L3 on the pass-1 curve `c`, EV of L3; aided path (model-aided tracking, gaps from
/// the model) when forced or EV ≥ min_ev, else plain (gaps from the model if it beats
/// analyze's fill in a hold-out on this file). `x`: interleaved f32 frames, `ch` channels.
pub fn model_step(
    x: &[f32],
    ch: usize,
    sr: u32,
    ref_hz: f64,
    c: &Curve,
    o: &StepOpts,
) -> Res<(Curve, ModelReport, Option<StepFit>)> {
    let t = &c.t;
    let r = &c.ratio;
    let v0 = &c.valid;
    let mut meta = c.meta.clone();
    let nv = v0.iter().filter(|&&v| v).count();
    let mut rep = ModelReport {
        mode: o.mode,
        min_ev: o.min_ev,
        constrain: None,
        physics: None,
        fit: None,
        path: String::new(),
        reason: String::new(),
        holdout: None,
        gapfill: None,
        aided: None,
    };
    let t_end = t[t.len() - 1];
    if o.mode == ModelMode::Off
        || t_end < MODEL_MIN_S
        || (nv as f64 / v0.len() as f64) < MODEL_MIN_VALID
    {
        let off = o.mode == ModelMode::Off;
        rep.path = if off { "off" } else { "plain" }.into();
        rep.reason = if off {
            "model off"
        } else {
            "file < 60 s or valid < 20 %: no model"
        }
        .into();
        meta.insert("model_path".into(), MetaValue::Str(rep.path.clone()));
        meta.insert("model_ev".into(), MetaValue::Str("nan".into()));
        return Ok((
            Curve {
                t: t.clone(),
                ratio: r.clone(),
                confidence: c.confidence.clone(),
                valid: v0.clone(),
                meta,
            },
            rep,
            None,
        ));
    }
    let span = (t[0], t_end);
    let w0: Vec<f64> = v0.iter().map(|&v| v as u8 as f64).collect();
    let mut h = fit_hier(
        t,
        r,
        &w0,
        &HierOpts {
            ta: o.ta,
            ident: ident_kw(t_end),
            ..HierOpts::default()
        },
    )?;
    let amps = |h: &Hier| -> Amps {
        h.comps
            .iter()
            .enumerate()
            .map(|(k, cc)| {
                let a = cc
                    .harmonics
                    .iter()
                    .map(|&hh| amp_abs_l2(&h.m2, (k, hh)))
                    .fold(f64::NEG_INFINITY, f64::max);
                (k, a)
            })
            .collect()
    };
    if o.v_given
        && let Some(vp) = o.v_phys.filter(|&v| v != 0.0)
    {
        let (comps2, cinfo) = physics::constrain(&h.comps, vp, span)?;
        let refit = cinfo.joint || comps2 != h.comps;
        rep.constrain = Some(cinfo);
        if refit {
            h = fit_hier(
                t,
                r,
                &w0,
                &HierOpts {
                    ta: o.ta,
                    comps: Some(comps2),
                    ..HierOpts::default()
                },
            )?;
        }
    }
    let am = amps(&h);
    rep.physics = Some(physics::infer(
        &h.comps,
        span,
        Some(&am),
        if o.v_given { o.v_phys } else { None },
    ));
    let rej = h.rejected.clone();
    let vv: Vec<bool> = v0.iter().zip(&rej).map(|(&a, &j)| a && !j).collect();
    let wv: Vec<f64> = vv.iter().map(|&x| x as u8 as f64).collect();
    let ex = explained(&[("m1", &h.m1), ("m2", &h.m2), ("m3", &h.m3)], t, r, &wv);
    let ev = ex["expl_m3"];
    let nrej = rej.iter().filter(|&&j| j).count();
    rep.fit = Some(FitReport {
        ev,
        ev_l2: ex["expl_m2"],
        r0: h.m1.r0,
        rejected: nrej as f64 / rej.len() as f64,
        n_comps: h.comps.len(),
        comps: comp_table(&h.comps, &h.m2, &h.m3, t, &wv),
    });
    ts_io::prof::mark("model.fit (L1-L3)");
    let aided = o.mode == ModelMode::On || ev >= o.min_ev;
    rep.path = if aided { "aided" } else { "plain" }.into();
    rep.reason = format!(
        "EV {} {} {}{}",
        fmt_fixed(ev, 2),
        if ev >= o.min_ev { ">=" } else { "<" },
        repr_shortest(o.min_ev),
        if o.mode == ModelMode::On {
            " (forced)"
        } else {
            ""
        }
    );
    let mut m_final = None;
    let (ratio, valid, conf);
    if aided {
        let mut b = o.b;
        let auto_bw = meta
            .get("bw")
            .map(|v| v.to_text().starts_with("auto"))
            .unwrap_or(false);
        if auto_bw && let Some(pb) = meta.get("passband_hz").and_then(meta_f64) {
            // min(B, pb): B unless pb is strictly smaller
            if pb < b {
                b = pb;
            }
        }
        let (ca, dbg) = track_aided(
            x,
            ch,
            sr,
            ref_hz,
            &h.m3,
            &AidedOpts {
                b,
                refine: o.refine,
                channel: o.channel,
                ..AidedOpts::default()
            },
        )?;
        ratio = model_fill(t, &ca.ratio, &ca.valid, &dbg.model, 0.1);
        ts_io::prof::mark("model.aided");
        m_final = Some(dbg.model);
        valid = ca.valid;
        conf = ca.confidence;
        rep.gapfill = Some("model".into());
        rep.aided = Some((b, o.refine));
        meta.insert("aided_B".into(), MetaValue::Float(b));
        meta.insert("refine".into(), MetaValue::Int(o.refine as i64));
    } else {
        let ho = if h.comps.is_empty() {
            Holdout { n: 0, rms: None }
        } else {
            holdout_fill_default(t, r, &vv, &h.comps, o.ta)?
        };
        rep.holdout = Some(ho);
        ts_io::prof::mark("model.holdout");
        if ho.model() < ho.analyze() {
            ratio = model_fill(t, r, &vv, &h.m3, 0.1);
            rep.gapfill = Some("model".into());
        } else {
            let nvv = vv.iter().filter(|&&x| x).count();
            ratio = if nrej > 0 && nvv >= 2 {
                let g = gap_fill(t, r, &vv)?;
                (0..t.len())
                    .map(|i| if rej[i] { g[i] } else { r[i] })
                    .collect()
            } else {
                r.clone()
            };
            rep.gapfill = Some("analyze".into());
        }
        conf = (0..t.len())
            .map(|i| if vv[i] { c.confidence[i] } else { 0.0 })
            .collect();
        valid = vv;
        if let Some((m, a, p)) = ho.rms {
            meta.insert(
                "model_holdout_rms".into(),
                MetaValue::Str(format!(
                    "model {} analyze {} pchip {} n {}",
                    fmt_sci(m, 2),
                    fmt_sci(a, 2),
                    fmt_sci(p, 2),
                    ho.n
                )),
            );
        }
    }
    let fr = rep.fit.as_ref().expect("fit report");
    meta.insert("model_path".into(), MetaValue::Str(rep.path.clone()));
    meta.insert("model_ev".into(), MetaValue::Float(round_dec(ev, 4)));
    meta.insert(
        "model_ev_l2".into(),
        MetaValue::Float(round_dec(fr.ev_l2, 4)),
    );
    meta.insert("model_min_ev".into(), MetaValue::Float(o.min_ev));
    meta.insert("model_comps".into(), MetaValue::Int(h.comps.len() as i64));
    meta.insert(
        "model_rejected_pct".into(),
        MetaValue::Float(round_dec(100.0 * fr.rejected, 2)),
    );
    meta.insert(
        "model_gapfill".into(),
        MetaValue::Str(rep.gapfill.clone().unwrap_or_default()),
    );
    meta.insert("model_Ta".into(), MetaValue::Float(o.ta));
    Ok((
        Curve {
            t: t.clone(),
            ratio,
            confidence: conf,
            valid,
            meta,
        },
        rep,
        Some(StepFit { hier: h, m_final }),
    ))
}

// ------------------------------------------------------------------ report_text

/// Scientific notation with `n` decimals: mantissa with n decimals, exponent sign and ≥ 2 digits.
pub fn fmt_sci(x: f64, n: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let s = format!("{x:.n$e}");
    let (m, e) = s.split_once('e').expect("exponent");
    let (sign, digits) = match e.strip_prefix('-') {
        Some(d) => ('-', d),
        None => ('+', e),
    };
    format!("{m}e{sign}{digits:0>2}")
}

/// Quoted string representation (single quotes unless the text contains `'` and no `"`,
/// backslash escapes for control characters).
pub fn repr_text(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut o = String::with_capacity(s.len() + 2);
    o.push(q);
    for ch in s.chars() {
        match ch {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == q => {
                o.push('\\');
                o.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                o.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

/// Text form of the `constrain` info: `{'format': ..., 'problems': [...], 'joint': ...}`.
pub fn constrain_repr(c: &ConstrainInfo) -> String {
    let probs: Vec<String> = c.problems.iter().map(|p| repr_text(p)).collect();
    let b = |v: bool| if v { "True" } else { "False" };
    let mut s = format!(
        "{{'format': {}, 'problems': [{}], 'joint': {}",
        repr_text(&c.format),
        probs.join(", "),
        b(c.joint)
    );
    if let Some((h, hf, cl)) = c.fit {
        s += &format!(
            ", 'h_um': {}, 'h_free_um': {}, 'clipped': {}",
            repr_shortest(h),
            repr_shortest(hf),
            b(cl)
        );
    }
    s.push('}');
    s
}

/// `x` with `p` decimals, right-aligned to width `w`.
fn fw(x: f64, w: usize, p: usize) -> String {
    format!("{:>w$}", fmt_fixed(x, p))
}

/// General format, 6 significant digits.
fn fmt_g(x: f64) -> String {
    ts_io::fmt_general(x, 6)
}

/// `report_text(rep)`: human-readable component / physics report.
pub fn report_text(rep: &ModelReport) -> String {
    let mut l = vec![format!("model: path {} ({})", rep.path, rep.reason)];
    if let Some(f) = &rep.fit {
        l.push(format!(
            "  EV L2 {} %, L3 {} %, r0 {}, jump-test rejected {} %, gap fill {}",
            fmt_fixed(f.ev_l2 * 100.0, 1),
            fmt_fixed(f.ev * 100.0, 1),
            fmt_fixed(f.r0, 5),
            fmt_fixed(f.rejected * 100.0, 2),
            rep.gapfill.as_deref().unwrap_or("None")
        ));
        if let Some(h) = &rep.holdout
            && let Some((m, a, p)) = h.rms
            && h.n >= 5
        {
            l.push(format!(
                "  gap-fill hold-out RMS (n {}): model {}, analyze {}, pchip {}",
                h.n,
                fmt_sci(m, 2),
                fmt_sci(a, 2),
                fmt_sci(p, 2)
            ));
        }
        for d in &f.comps {
            if d.amp_l3_rms_pct >= 0.01 {
                l.push(format!(
                    "  {:>12} {:5} h{} {}->{} Hz  L2 {} %  L3 rms {} %",
                    d.name,
                    d.kind,
                    d.h,
                    fw(d.f_start, 7, 3),
                    fw(d.f_end, 7, 3),
                    fmt_fixed(d.amp_l2_pct, 3),
                    fmt_fixed(d.amp_l3_rms_pct, 3)
                ));
            }
        }
    }
    if let Some(ph) = &rep.physics {
        let given = match ph.v_given {
            Some(v) if v != 0.0 => format!("given {} cm/s", fmt_fixed(v * 100.0, 2)),
            _ => "inferred".into(),
        };
        let sp: Vec<String> = ph.speeds.iter().map(|(k, v)| format!("{k}: {v}")).collect();
        l.push(format!("  tape speed: {given}; {}", sp.join(", ")));
        if let Some(g) = &ph.geometry {
            for (role, d) in &g.reels {
                l.push(format!(
                    "  {role} reel ({}): R {}->{} cm, tape {} um",
                    d.comp,
                    fmt_fixed(d.r_start_cm, 2),
                    fmt_fixed(d.r_end_cm, 2),
                    fmt_fixed(d.h_um, 1)
                ));
            }
            if let Some(c) = &g.capstan {
                l.push(format!(
                    "  capstan ({}, {} Hz): d {} mm",
                    c.comp,
                    fmt_fixed(c.f_hz, 3),
                    fmt_fixed(c.d_mm, 2)
                ));
            }
            if let Some(p) = &g.pack {
                l.push(format!(
                    "  pack: R_full {} cm (~{}\" reel), play time {} min if file starts at tape start, h take-up/supply {}",
                    fmt_fixed(p.r_full_cm, 1),
                    fmt_g(p.reel_in),
                    fmt_fixed(p.play_min_if_start, 0),
                    fmt_fixed(p.h_agree, 2)
                ));
            }
        }
    }
    if let Some(c) = &rep.constrain {
        l.push(format!("  constrained: {}", constrain_repr(c)));
    }
    l.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_e() {
        assert_eq!(fmt_sci(2.2522083664088617e-4, 2), "2.25e-04");
        assert_eq!(fmt_sci(1.83e-3, 2), "1.83e-03");
        assert_eq!(fmt_sci(0.0, 2), "0.00e+00");
        assert_eq!(fmt_sci(-12345.0, 2), "-1.23e+04");
        assert_eq!(fmt_sci(1e100, 1), "1.0e+100");
    }

    #[test]
    fn repr_str() {
        assert_eq!(repr_text("open reel"), "'open reel'");
        assert_eq!(repr_text("7\" reel"), "'7\" reel'");
        assert_eq!(repr_text("it's"), "\"it's\"");
        assert_eq!(repr_text("a'b\"c"), "'a\\'b\"c'");
    }
}
