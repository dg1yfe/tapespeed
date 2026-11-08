//! physics: tape-transport physics for the speed model.
//!
//! Reel component (tape-driven, no slip): `f = v / (2 pi R)`, `R^2 = R0^2 +- (v h / pi) t`
//! (take-up +, supply -), i.e. the law `f = b / sqrt(1 + beta t)` with `R0 = v / (2 pi b)`,
//! `h = pi R0^2 |beta| / v`. Slip-clutch drive: `f = f_drive - f_takeup(t)`. Capstan:
//! `v = pi d f`. Frequencies fix only R/v and h/v; v is resolved by plausible geometry ranges.

use indexmap::IndexMap;
use ts_dsp::ValueError;
use ts_dsp::linalg::lstsq;
use ts_dsp::stats::np_sum;
use ts_io::json::Value;
use ts_io::{fmt_fixed, fmt_general};

use crate::absolute::quote_str;
use crate::model::law::{Comp, Kind};

/// Standard speeds [m/s] (15/16, 1 7/8, 3 3/4, 7 1/2, 15 ips).
pub const SPEEDS: [f64; 5] = [0.0238, 0.0476, 0.0953, 0.1905, 0.381];
/// Capstan diameter range [m].
pub const CAPSTAN_D: (f64, f64) = (1.5e-3, 30e-3);
/// Standard reel diameters [inch].
pub const REEL_IN: [f64; 4] = [3.0, 5.0, 7.0, 10.5];

/// Tape format: pack radius range [m], tape thickness range [m], speeds.
#[derive(Debug, Clone, Copy)]
pub struct Format {
    pub name: &'static str,
    pub r: (f64, f64),
    pub h: (f64, f64),
    pub v: &'static [f64],
}

/// `FORMATS` (dict order: cassette, open reel).
pub const FORMATS: [Format; 2] = [
    Format {
        name: "cassette",
        r: (0.010, 0.030),
        h: (8e-6, 20e-6),
        v: &[0.0238, 0.0476, 0.0953],
    },
    Format {
        name: "open reel",
        r: (0.024, 0.135),
        h: (12e-6, 60e-6),
        v: &[0.0238, 0.0476, 0.0953, 0.1905, 0.381],
    },
];

/// [`parse_speed`] errors (invalid value / division by zero).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpeedError {
    #[error("{0}")]
    Value(String),
    #[error("{0}")]
    ZeroDivision(String),
}

/// Character range `[a, e)`.
type Span = (usize, usize);
/// `(whole number, numerator, denominator, end)` of `[W[\s-]+]N/D`.
type Frac = (Option<Span>, Span, Span, usize);

fn digits(b: &[char], i: usize) -> usize {
    let mut j = i;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    j
}

fn ascii_int(b: &[char], a: usize, e: usize) -> f64 {
    b[a..e]
        .iter()
        .fold(0.0, |acc, c| acc * 10.0 + c.to_digit(10).unwrap() as f64)
}

/// `parse_speed(s)`: `'19.05cm'`, `'19.05 cm/s'`, `'7.5ips'`, `'7½ips'`, `'1 7/8 ips'`,
/// `'15/16ips'` → m/s (bare number: cm/s). Digits are ASCII only (other Unicode decimal
/// digits are rejected).
pub fn parse_speed(s: &str) -> Result<f64, SpeedError> {
    let mut s0 = s.trim().to_lowercase();
    for (k, v) in [("½", " 1/2"), ("¾", " 3/4"), ("⅞", " 7/8"), ("¼", " 1/4")] {
        s0 = s0.replace(k, v);
    }
    let b: Vec<char> = s0.trim().chars().collect();
    let fail = || {
        SpeedError::Value(format!(
            "tape speed {}: use e.g. 19.05cm or 7.5ips",
            quote_str(s)
        ))
    };
    // number part: (?:(\d+)[\s-]+)?(\d+)/(\d+) | \d+(?:\.\d+)?  -> (value, end)
    let fraction = |i: usize| -> Option<Frac> {
        let frac_at = |k: usize| -> Option<(Span, Span, usize)> {
            let e1 = digits(&b, k);
            if e1 == k || e1 >= b.len() || b[e1] != '/' {
                return None;
            }
            let e2 = digits(&b, e1 + 1);
            if e2 == e1 + 1 {
                return None;
            }
            Some(((k, e1), (e1 + 1, e2), e2))
        };
        // with the whole-number group first
        let e0 = digits(&b, i);
        if e0 > i {
            let mut k = e0;
            while k < b.len() && (b[k].is_whitespace() || b[k] == '-') {
                k += 1;
            }
            if k > e0
                && let Some((n, d, e)) = frac_at(k)
            {
                return Some((Some((i, e0)), n, d, e));
            }
        }
        frac_at(i).map(|(n, d, e)| (None, n, d, e))
    };
    let unit_at = |e: usize| -> Option<f64> {
        let mut k = e;
        while k < b.len() && b[k].is_whitespace() {
            k += 1;
        }
        let rest: String = b[k..].iter().collect();
        match rest.as_str() {
            "" | "cm/s" | "cm" => Some(0.01),
            "ips" | "in/s" => Some(0.0254),
            "m/s" => Some(1.0),
            _ => None,
        }
    };
    if let Some((w, n, d, e)) = fraction(0)
        && let Some(u) = unit_at(e)
    {
        let den = ascii_int(&b, d.0, d.1);
        if den == 0.0 {
            return Err(SpeedError::ZeroDivision("division by zero".into()));
        }
        let whole = w.map_or(0.0, |(a, z)| ascii_int(&b, a, z));
        return Ok((whole + ascii_int(&b, n.0, n.1) / den) * u);
    }
    let e1 = digits(&b, 0);
    if e1 == 0 {
        return Err(fail());
    }
    let mut e = e1;
    if e < b.len() && b[e] == '.' {
        let e2 = digits(&b, e + 1);
        if e2 > e + 1 {
            e = e2;
        }
    }
    let u = unit_at(e).ok_or_else(fail)?;
    let txt: String = b[..e].iter().collect();
    Ok(txt.parse::<f64>().map_err(|_| fail())? * u)
}

/// `fmt_speed(v)`: `"19.05 cm/s"`.
pub fn fmt_speed(v: f64) -> String {
    format!("{} cm/s", fmt_fixed(v * 100.0, 2))
}

/// `classify(comps)` result: indices of take-up / supply reels, clutch, fixed components.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Classes {
    pub take: Option<usize>,
    pub supply: Option<usize>,
    pub clutch: Option<usize>,
    pub fixed: Vec<usize>,
}

/// `max(idx, key=amp_db, default=None)` (first maximum; `None` if empty).
fn max_by_amp(comps: &[Comp], idx: &[usize]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for &k in idx {
        if best.is_none_or(|b| comps[k].amp_db > comps[b].amp_db) {
            best = Some(k);
        }
    }
    best
}

/// `classify(comps)`: take-up / supply reels, clutch (drive - take-up), fixed components.
pub fn classify(comps: &[Comp]) -> Classes {
    let reels: Vec<usize> = (0..comps.len())
        .filter(|&k| {
            let c = &comps[k];
            c.kind == Kind::Chirp && c.a.abs() < 0.1f64.max(0.15 * c.b.abs()) && c.b > 0.0
        })
        .collect();
    let take: Vec<usize> = reels
        .iter()
        .copied()
        .filter(|&k| comps[k].beta > 0.0)
        .collect();
    let sup: Vec<usize> = reels
        .iter()
        .copied()
        .filter(|&k| comps[k].beta < 0.0)
        .collect();
    let tk = max_by_amp(comps, &take);
    let sk = max_by_amp(comps, &sup);
    let mut cl = None;
    if let Some(t) = tk {
        let (bt, et) = (comps[t].b, comps[t].beta);
        for (k, c) in comps.iter().enumerate() {
            if c.kind == Kind::Chirp
                && c.a > 0.0
                && c.b < 0.0
                && (-c.b / bt - 1.0).abs() < 0.3
                && (c.beta / et - 1.0).abs() < 0.3
            {
                cl = Some(k);
            }
        }
    }
    Classes {
        take: tk,
        supply: sk,
        clutch: cl,
        fixed: (0..comps.len())
            .filter(|&k| comps[k].kind == Kind::Fixed)
            .collect(),
    }
}

/// Reel geometry at speed v.
#[derive(Debug, Clone, PartialEq)]
pub struct Reel {
    pub comp: String,
    pub r_start_cm: f64,
    pub r_end_cm: f64,
    pub h_um: f64,
}

/// Fixed component as a roller of diameter `d_mm`.
#[derive(Debug, Clone, PartialEq)]
pub struct Roller {
    pub comp: String,
    pub f_hz: f64,
    pub d_mm: f64,
    /// `amps.get(k, 0.0)` if amplitudes were given.
    pub amp: Option<f64>,
}

/// Pack estimate from both reels.
#[derive(Debug, Clone, PartialEq)]
pub struct Pack {
    pub r_full_cm: f64,
    pub reel_in: f64,
    pub h_agree: f64,
    pub play_min_if_start: f64,
}

/// `geometry(comps, v, span, amps)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub v: f64,
    /// Ordered `take`, `supply` (those present).
    pub reels: Vec<(&'static str, Reel)>,
    pub fixed: Vec<Roller>,
    /// `(comp, f_drive_hz)`.
    pub clutch: Option<(String, f64)>,
    /// Strongest fixed component (amp), assumed capstan.
    pub capstan: Option<Roller>,
    pub pack: Option<Pack>,
}

impl Geometry {
    fn reel(&self, role: &str) -> Option<&Reel> {
        self.reels.iter().find(|(r, _)| *r == role).map(|(_, d)| d)
    }

    /// The report entry as a JSON value (key order is part of the format).
    pub fn to_value(&self) -> Value {
        let mut reels = Value::obj();
        for (role, d) in &self.reels {
            reels.push(
                role,
                Value::obj()
                    .with("comp", d.comp.as_str())
                    .with("R_start_cm", d.r_start_cm)
                    .with("R_end_cm", d.r_end_cm)
                    .with("h_um", d.h_um),
            );
        }
        let fixed: Vec<Value> = self
            .fixed
            .iter()
            .map(|f| {
                Value::obj()
                    .with("comp", f.comp.as_str())
                    .with("f_hz", f.f_hz)
                    .with("d_mm", f.d_mm)
                    .with("amp", f.amp.map(Value::Float).unwrap_or(Value::Null))
            })
            .collect();
        let mut v = Value::obj()
            .with("v", self.v)
            .with("reels", reels)
            .with("fixed", Value::Arr(fixed));
        if let Some((c, f)) = &self.clutch {
            v.push(
                "clutch",
                Value::obj().with("comp", c.as_str()).with("f_drive_hz", *f),
            );
        }
        if let Some(c) = &self.capstan {
            v.push(
                "capstan",
                Value::obj()
                    .with("comp", c.comp.as_str())
                    .with("f_hz", c.f_hz)
                    .with("d_mm", c.d_mm)
                    .with("note", "strongest fixed component, assumed capstan"),
            );
        }
        if let Some(p) = &self.pack {
            v.push(
                "pack",
                Value::obj()
                    .with("R_full_cm", p.r_full_cm)
                    .with("reel_in", p.reel_in)
                    .with("h_agree", p.h_agree)
                    .with("play_min_if_start", p.play_min_if_start)
                    .with(
                        "note",
                        "play time assumes the file starts at the tape start (hub = first take-up radius)",
                    ),
            );
        }
        v
    }
}

/// Component amplitudes by index (`amps` dict); empty = not given.
pub type Amps = IndexMap<usize, f64>;

fn amps_on(amps: Option<&Amps>) -> Option<&Amps> {
    amps.filter(|a| !a.is_empty())
}

/// `geometry(comps, v, span, amps)`: physical parameters at playback speed v over span.
pub fn geometry(comps: &[Comp], v: f64, span: (f64, f64), amps: Option<&Amps>) -> Geometry {
    let cl = classify(comps);
    let amps = amps_on(amps);
    let two_pi = 2.0 * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    let mut out = Geometry {
        v,
        reels: vec![],
        fixed: vec![],
        clutch: None,
        capstan: None,
        pack: None,
    };
    for (role, k) in [("take", cl.take), ("supply", cl.supply)] {
        let Some(k) = k else { continue };
        let c = &comps[k];
        let r0 = v / (two_pi * c.freq_at(span.0));
        let r1 = v / (two_pi * c.freq_at(span.1));
        let rr0 = v / (two_pi * c.b);
        out.reels.push((
            role,
            Reel {
                comp: c.name.clone(),
                r_start_cm: r0 * 100.0,
                r_end_cm: r1 * 100.0,
                h_um: pi * (rr0 * rr0) * c.beta.abs() / v * 1e6,
            },
        ));
    }
    if let Some(k) = cl.clutch {
        out.clutch = Some((comps[k].name.clone(), comps[k].a));
    }
    for &k in &cl.fixed {
        let c = &comps[k];
        out.fixed.push(Roller {
            comp: c.name.clone(),
            f_hz: c.f_mean,
            d_mm: v / (pi * c.f_mean) * 1e3,
            amp: amps.map(|a| a.get(&k).copied().unwrap_or(0.0)),
        });
    }
    if !out.fixed.is_empty() && amps.is_some() {
        let mut cap = &out.fixed[0];
        for f in &out.fixed[1..] {
            // max(key=amp or 0): first maximum
            if f.amp.unwrap_or(0.0) > cap.amp.unwrap_or(0.0) {
                cap = f;
            }
        }
        out.capstan = Some(Roller {
            amp: None,
            ..cap.clone()
        });
    }
    if let (Some(t), Some(s)) = (out.reel("take"), out.reel("supply")) {
        let c2 = (t.r_start_cm * t.r_start_cm + s.r_start_cm * s.r_start_cm) / 1e4;
        let h = 0.5 * (t.h_um + s.h_um) * 1e-6;
        let rhub = t.r_start_cm / 100.0;
        let rfull = (c2 - rhub * rhub).max(0.0).sqrt();
        let target = rfull * 100.0 * 1.05;
        let mut reel_in = REEL_IN[0];
        for &d in &REEL_IN[1..] {
            if (d * 1.27 - target).abs() < (reel_in * 1.27 - target).abs() {
                reel_in = d;
            }
        }
        out.pack = Some(Pack {
            r_full_cm: rfull * 100.0,
            reel_in,
            h_agree: t.h_um / s.h_um,
            play_min_if_start: pi * (rfull * rfull - rhub * rhub) / (v * h) / 60.0,
        });
    }
    out
}

/// `plausibility(geo)`: `(format, problems)` per format; empty problems = consistent.
pub fn plausibility(geo: &Geometry) -> Vec<(&'static str, Vec<String>)> {
    let mut res = Vec::new();
    for fm in &FORMATS {
        let mut prob = Vec::new();
        if !fm.v.iter().any(|&s| (geo.v - s).abs() < 1e-4) {
            prob.push(format!("speed not used on {}", fm.name));
        }
        for (role, d) in &geo.reels {
            let (lo, hi) = fm.r;
            for rv in [d.r_start_cm, d.r_end_cm] {
                let r = rv / 100.0;
                if !(lo * 0.95 <= r && r <= hi * 1.05) {
                    prob.push(format!(
                        "{role} R {} cm outside {}-{}",
                        fmt_fixed(r * 100.0, 2),
                        fmt_general(lo * 100.0, 6),
                        fmt_general(hi * 100.0, 6)
                    ));
                    break;
                }
            }
            let hh = d.h_um * 1e-6;
            if !(fm.h.0 * 0.8 <= hh && hh <= fm.h.1 * 1.2) {
                prob.push(format!(
                    "{role} tape {} um outside {}-{}",
                    fmt_fixed(d.h_um, 0),
                    fmt_general(fm.h.0 * 1e6, 6),
                    fmt_general(fm.h.1 * 1e6, 6)
                ));
            }
        }
        if let Some(c) = &geo.capstan {
            let dm = c.d_mm / 1e3;
            if !(CAPSTAN_D.0 <= dm && dm <= CAPSTAN_D.1) {
                prob.push(format!("capstan {} mm", fmt_fixed(c.d_mm, 1)));
            }
        }
        res.push((fm.name, prob));
    }
    res
}

/// `infer(...)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct Inference {
    /// `"cm/s with 2 decimals" -> verdict` (dict order).
    pub speeds: IndexMap<String, String>,
    pub consistent: Vec<f64>,
    pub has_reel: bool,
    pub v_given: Option<f64>,
    pub v_used: Option<f64>,
    pub geometry: Option<Geometry>,
}

impl Inference {
    /// The report entry as a JSON value (key order is part of the format).
    pub fn to_value(&self) -> Value {
        let mut sp = Value::obj();
        for (k, v) in &self.speeds {
            sp.push(k, v.as_str());
        }
        let opt = |v: Option<f64>| v.map(Value::Float).unwrap_or(Value::Null);
        Value::obj()
            .with("speeds", sp)
            .with("consistent", self.consistent.clone())
            .with("has_reel", self.has_reel)
            .with("v_given", opt(self.v_given))
            .with("v_used", opt(self.v_used))
            .with(
                "geometry",
                self.geometry
                    .as_ref()
                    .map(|g| g.to_value())
                    .unwrap_or(Value::Null),
            )
    }
}

/// `infer(comps, span, amps, v_given)`: geometry and plausibility for each standard speed
/// (and `v_given`); the speed used is `v_given` or the single consistent one.
pub fn infer(
    comps: &[Comp],
    span: (f64, f64),
    amps: Option<&Amps>,
    v_given: Option<f64>,
) -> Inference {
    let vg = v_given.filter(|&v| v != 0.0);
    let mut cand: Vec<f64> = SPEEDS.to_vec();
    if let Some(v) = vg
        && !cand.contains(&v)
    {
        cand.push(v);
    }
    cand.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cl = classify(comps);
    let has_reel = cl.take.is_some() || cl.supply.is_some();
    let mut verd = IndexMap::new();
    let mut ok = Vec::new();
    for &v in &cand {
        let key = fmt_fixed(v * 100.0, 2);
        if !has_reel {
            verd.insert(key, "undetermined (no reel component)".to_string());
            continue;
        }
        let pl = plausibility(&geometry(comps, v, span, amps));
        let good: Vec<&str> = pl
            .iter()
            .filter(|(_, p)| p.is_empty())
            .map(|(f, _)| *f)
            .collect();
        if !good.is_empty() {
            verd.insert(key, format!("consistent ({})", good.join(", ")));
            ok.push(v);
        } else {
            let mut best = &pl[0];
            for e in &pl[1..] {
                if e.1.len() < best.1.len() {
                    best = e;
                }
            }
            verd.insert(
                key,
                format!("implausible ({}: {})", best.0, best.1.join("; ")),
            );
        }
    }
    let v_use = vg.or(if ok.len() == 1 { Some(ok[0]) } else { None });
    Inference {
        speeds: verd,
        consistent: ok,
        has_reel,
        v_given,
        v_used: v_use,
        geometry: v_use.map(|v| geometry(comps, v, span, amps)),
    }
}

/// `constrain` info dict.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstrainInfo {
    pub format: String,
    pub problems: Vec<String>,
    pub joint: bool,
    /// `(h_um, h_free_um, clipped)` when the joint fit ran.
    pub fit: Option<(f64, f64, bool)>,
}

impl ConstrainInfo {
    pub fn to_value(&self) -> Value {
        let mut v = Value::obj()
            .with("format", self.format.as_str())
            .with("problems", self.problems.clone())
            .with("joint", self.joint);
        if let Some((h, hf, c)) = self.fit {
            v.push("h_um", h);
            v.push("h_free_um", hf);
            v.push("clipped", c);
        }
        v
    }
}

/// `np.average(a, weights=w)`.
fn average(a: &[f64], w: &[f64]) -> f64 {
    let aw: Vec<f64> = a.iter().zip(w).map(|(x, y)| x * y).collect();
    np_sum(&aw) / np_sum(w)
}

/// Normalise a modified component: `f_mean = a + b` when it is unset (0).
fn post_init(mut c: Comp) -> Comp {
    if c.f_mean == 0.0 {
        c.f_mean = c.a + c.b;
    }
    c
}

/// `constrain(comps, v, span)`: with known playback speed v, joint reel fit on the ridge
/// points with one tape thickness (clipped to the format's range); clutch tied to the take-up
/// law; reels implausible at v are renamed `name?` (not reels).
pub fn constrain(
    comps: &[Comp],
    v: f64,
    span: (f64, f64),
) -> Result<(Vec<Comp>, ConstrainInfo), ValueError> {
    let cl = classify(comps);
    let geo = geometry(comps, v, span, None);
    let pl = plausibility(&geo);
    let mut fi = 0;
    for (i, e) in pl.iter().enumerate() {
        if e.1.len() < pl[fi].1.len() {
            fi = i;
        }
    }
    let fmt = &FORMATS[fi];
    let mut info = ConstrainInfo {
        format: fmt.name.into(),
        problems: pl[fi].1.clone(),
        joint: false,
        fit: None,
    };
    let mut out: Vec<Comp> = comps.to_vec();
    let mut roles: Vec<(&str, usize)> = [("take", cl.take), ("supply", cl.supply)]
        .into_iter()
        .filter_map(|(r, k)| k.filter(|&k| comps[k].pts.is_some()).map(|k| (r, k)))
        .collect();
    let bad: Vec<&str> = pl[fi]
        .1
        .iter()
        .filter(|p| p.starts_with("take") || p.starts_with("supply"))
        .map(|p| p.split_whitespace().next().unwrap_or(""))
        .collect();
    for &(r, k) in &roles {
        if bad.contains(&r) {
            let mut c = comps[k].clone();
            c.name = format!("{}?", c.name);
            out[k] = post_init(c);
        }
    }
    roles.retain(|(r, _)| !bad.contains(r));
    if roles.is_empty() {
        return Ok((out, info));
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    let nr = roles.len();
    let sgn = |r: &str| if r == "take" { 1.0 } else { -1.0 };
    let (mut a, mut y, mut wv) = (Vec::new(), Vec::new(), Vec::new());
    let mut ys: Vec<Vec<f64>> = Vec::new();
    let mut ws: Vec<Vec<f64>> = Vec::new();
    for (j, &(r, k)) in roles.iter().enumerate() {
        let (tt, ff, ww) = comps[k].pts.as_ref().unwrap();
        let g: Vec<f64> = ff
            .iter()
            .map(|&f| {
                let q = two_pi * f;
                1.0 / (q * q)
            })
            .collect();
        for &t in tt {
            let mut row = vec![0.0; nr + 1];
            row[j] = 1.0;
            row[nr] = t * sgn(r);
            a.push(row);
        }
        let w: Vec<f64> = ww.iter().zip(ff).map(|(&w, &f)| w * f.powf(3.0)).collect();
        y.extend_from_slice(&g);
        wv.extend_from_slice(&w);
        ys.push(g);
        ws.push(w);
    }
    let sw: Vec<f64> = wv.iter().map(|w| w.sqrt()).collect();
    let m = a.len();
    let mut am = Vec::with_capacity(m * (nr + 1));
    for (row, &s) in a.iter().zip(&sw) {
        am.extend(row.iter().map(|&x| x * s));
    }
    let yw: Vec<f64> = y.iter().zip(&sw).map(|(a, b)| a * b).collect();
    let mut x = lstsq(&am, m, nr + 1, &yw, None)?.x;
    let (hlo, hhi) = fmt.h;
    let k_free = x[nr];
    let (lo, hi) = (hlo / (pi * v), hhi / (pi * v));
    // np.clip: max(min(x, hi), lo) with NaN propagating
    let kk = if k_free.is_nan() {
        k_free
    } else {
        k_free.min(hi).max(lo)
    };
    if kk != k_free {
        for (j, &(r, k)) in roles.iter().enumerate() {
            let tt = &comps[k].pts.as_ref().unwrap().0;
            let d: Vec<f64> = ys[j]
                .iter()
                .zip(tt)
                .map(|(&g, &t)| g - kk * t * sgn(r))
                .collect();
            x[j] = average(&d, &ws[j]);
        }
    }
    x[nr] = kk;
    for (j, &(r, k)) in roles.iter().enumerate() {
        let u = x[j];
        let mut c = comps[k].clone();
        c.a = 0.0;
        c.b = 1.0 / (two_pi * u.sqrt());
        c.beta = kk / u * sgn(r);
        let mut c = post_init(c);
        c.f_mean = c.mean_freq(span);
        out[k] = c;
    }
    if let (Some(ck), Some(tk)) = (cl.clutch, cl.take)
        && roles.contains(&("take", tk))
    {
        let c = &comps[ck];
        let t_ = &out[tk];
        let a_ = match &c.pts {
            Some((pt, pf, pw)) => {
                let s: Vec<f64> = pf
                    .iter()
                    .zip(pt)
                    .map(|(&f, &t)| f + t_.freq_at(t))
                    .collect();
                average(&s, pw)
            }
            None => c.a + c.b + t_.b,
        };
        let mut nc = c.clone();
        nc.a = a_;
        nc.b = -t_.b;
        nc.beta = t_.beta;
        out[ck] = post_init(nc);
    }
    info.joint = true;
    info.fit = Some((kk * pi * v * 1e6, k_free * pi * v * 1e6, kk != k_free));
    Ok((out, info))
}
