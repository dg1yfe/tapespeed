//! Survey recommendation and run plan: `find`, `implied_true`, `recommend`, `plan`, `spans`.
//! Parameters resolve through `params::resolve` and L1 through `absolute::apply_l1`, exactly
//! as the run does.

use ts_dsp::stats::median;
use ts_io::{fmt_fixed, fmt_general, repr_shortest};

use super::cands::{Cand, chan_name, first_max, score, usable};
use super::detect::{LOCK_DB, NOMINAL, level_db};
use super::report::{fmt_opt, sign_f};
use super::{Res, Survey, SurveyError};
use crate::absolute::apply_l1;
use crate::curve::{Curve, MetaValue};
use crate::evidence::{COH_MIN, FADE_RHO};
use crate::params::{
    Bw, RunArgs, lowpass_stop, manual_notches, notch_skip, out_paths, path_stem, play_factor,
    ref_notch, resolve,
};

pub use crate::analyze::BW_AUTO;
/// `model.LONG_GAP_S`, `MODEL_MIN_S`, `MODEL_MIN_VALID`.
pub const LONG_GAP_S: f64 = 20.0;
pub const MODEL_MIN_S: f64 = 60.0;
pub const MODEL_MIN_VALID: f64 = 0.2;

/// The command-line fields `--survey` reads (`a.<name>`; CLI defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct SurveyArgs {
    pub input: String,
    pub output: Option<String>,
    pub r#ref: Option<f64>,
    pub survey_blocks: usize,
    pub survey_block_s: f64,
    pub curve: Option<String>,
    pub report: Option<String>,
    pub plot: Option<String>,
    pub no_plot: bool,
    pub keep_ref: bool,
    pub lowpass: Option<f64>,
    pub notch_at: Option<String>,
    pub auto_notch: Option<String>,
    pub auto_notch_min: f64,
    pub notch_width: f64,
    pub lowpass_width: f64,
    pub notch_q: f64,
    pub search: f64,
    pub bw: String,
    pub channel: String,
    pub taps: i64,
    pub model: String,
    pub model_min_ev: f64,
    pub model_band: f64,
    pub model_ta: f64,
    pub abs: String,
    pub pair_dmax: f64,
    pub check_line: Option<f64>,
    pub max_corr_hz: f64,
    pub corr_limit: String,
    pub weak_rf: bool,
    pub tape_speed: Option<String>,
    pub play_speed: Option<String>,
    /// Rust memory planner's peak estimate [bytes] for the run (PLAN step 13); None: not planned.
    pub mem_peak: Option<u64>,
}

impl Default for SurveyArgs {
    fn default() -> Self {
        Self {
            input: String::new(),
            output: None,
            r#ref: None,
            survey_blocks: 12,
            survey_block_s: 20.0,
            curve: None,
            report: None,
            plot: None,
            no_plot: false,
            keep_ref: false,
            lowpass: None,
            notch_at: None,
            auto_notch: None,
            auto_notch_min: 3000.0,
            notch_width: 150.0,
            lowpass_width: 500.0,
            notch_q: 30.0,
            search: 0.03,
            bw: "auto".into(),
            channel: "mid".into(),
            taps: 64,
            model: "auto".into(),
            model_min_ev: 0.70,
            model_band: 20.0,
            model_ta: 5.0,
            abs: "ref".into(),
            pair_dmax: 300.0,
            check_line: None,
            max_corr_hz: 6.0,
            corr_limit: "auto".into(),
            weak_rf: false,
            tape_speed: None,
            play_speed: None,
            mem_peak: None,
        }
    }
}

impl SurveyArgs {
    /// The parameter-resolution view (`a` as `resolve` / `out_paths` see it).
    pub fn run_args(&self) -> RunArgs {
        RunArgs {
            r#ref: self.r#ref.unwrap_or(0.0),
            output: self.output.clone().unwrap_or_default(),
            abs: self.abs.clone(),
            tape_speed: self.tape_speed.clone(),
            play_speed: self.play_speed.clone(),
            bw: self.bw.clone(),
            check_line: self.check_line,
            weak_rf: self.weak_rf,
            corr_limit: self.corr_limit.clone(),
            curve: self.curve.clone(),
            report: self.report.clone(),
            plot: self.plot.clone(),
            no_plot: self.no_plot,
            lowpass: self.lowpass,
            lowpass_width: self.lowpass_width,
            keep_ref: self.keep_ref,
            notch_at: self.notch_at.clone(),
            notch_width: self.notch_width,
        }
    }
}

/// A recommendation value: None, a float (nominal), an int (`round(f / play)`) or a string.
#[derive(Debug, Clone, PartialEq)]
pub enum RecVal {
    None,
    Float(f64),
    Int(i64),
    Str(String),
}

impl RecVal {
    pub fn num(&self) -> Option<f64> {
        match self {
            RecVal::Float(v) => Some(*v),
            RecVal::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// `format(v, "g")` for numbers, `str(v)` otherwise.
    pub fn fmt_g(&self) -> String {
        match self {
            RecVal::Float(v) => fmt_general(*v, 6),
            RecVal::Int(v) => fmt_general(*v as f64, 6),
            RecVal::Str(s) => s.clone(),
            RecVal::None => "None".into(),
        }
    }

    /// Text form: shortest round-trip float, integer, string or `None`.
    pub fn to_text(&self) -> String {
        match self {
            RecVal::Float(v) => repr_shortest(*v),
            RecVal::Int(v) => v.to_string(),
            RecVal::Str(s) => s.clone(),
            RecVal::None => "None".into(),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            RecVal::Str(s) => s,
            _ => "",
        }
    }
}

pub type RecItem = (RecVal, String);

/// `recommend(S)` (keys in report order: ref, channel, abs, check_line, cmd).
#[derive(Debug, Clone, PartialEq)]
pub struct Recommend {
    pub r#ref: RecItem,
    pub channel: Option<RecItem>,
    pub abs: Option<RecItem>,
    pub check_line: Option<RecItem>,
    pub cmd: Option<String>,
    /// UX only: `--search` when the primary lies outside or near the default window.
    pub search: Option<RecItem>,
    /// UX only: `--lowpass` (Hz) for an AM primary.
    pub lowpass: Option<RecItem>,
    /// UX only: `--auto-notch` mode for an AM primary.
    pub auto_notch: Option<RecItem>,
}

/// Shell-quote for the `cmd:` line (`'...'`, inner `'` as `'\''`).
pub fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The suggested output of the `cmd:` line: `<stem>_corrected.<ext>` in the cwd, `<ext>` = the
/// input's for lossless inputs, `flac` for lossy ones; never the input file.
pub fn corrected_name(input: &str, lossy: bool) -> String {
    let stem = path_stem(input);
    let ext = std::path::Path::new(input)
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .filter(|e| !lossy && !e.is_empty())
        .unwrap_or_else(|| "flac".into());
    format!("{stem}_corrected.{ext}")
}

/// `find(S, hz, search)`: strongest non-codec high-band candidate within hz (1 ± search).
pub fn find(s: &Survey, hz: f64, search: f64) -> Option<&Cand> {
    let m = s
        .candidates
        .iter()
        .filter(|c| c.cls != "codec" && !c.low && (c.f / hz - 1.0).abs() <= search);
    first_max(m, |a, b| {
        (usable(a), score(a)).partial_cmp(&(usable(b), score(b)))
            == Some(std::cmp::Ordering::Greater)
    })
}

fn refval(c: &Cand, play: f64) -> RecVal {
    match c.nominal {
        Some(n) if n != 0.0 => RecVal::Float(n),
        _ => RecVal::Int((c.f / play).round_ties_even() as i64),
    }
}

/// `implied_true(p, o)`: primary's true frequency if line `o` sits at its nominal.
pub fn implied_true(p: &Cand, o: &Cand) -> Option<f64> {
    let com: Vec<usize> = p
        .blocks
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| o.block(*k).is_some())
        .collect();
    let nom = o.nominal.filter(|&n| n != 0.0)?;
    if com.is_empty() {
        return None;
    }
    let r: Vec<f64> = com
        .iter()
        .map(|&b| p.block(b).unwrap().f / o.block(b).unwrap().f)
        .collect();
    Some(median(&r) * nom)
}

/// Recommended options for survey `s`: --ref / --channel / --abs / --check-line, plus --search,
/// --lowpass and --auto-notch where they apply, and a ready `cmd:` line.
pub fn recommend(s: &Survey) -> Recommend {
    let (cs, play) = (&s.candidates, s.play);
    let Some(pid) = s.primary else {
        return Recommend {
            r#ref: (
                RecVal::None,
                format!(
                    "no usable line (C/N0 >= {} dB-Hz survey scale, not codec/hum)",
                    fmt_general(LOCK_DB, 6)
                ),
            ),
            channel: None,
            abs: None,
            check_line: None,
            cmd: None,
            search: None,
            lowpass: None,
            auto_notch: None,
        };
    };
    let p = s.cand(pid).unwrap();
    let rf = refval(p, play);
    let mut r_ref = (
        rf.clone(),
        format!(
            "#{} {} at {} Hz, {}/{} blocks, C/N0 {} dB-Hz ({})",
            p.id,
            p.label,
            fmt_fixed(p.f, 1),
            p.lock_blocks,
            p.nb,
            fmt_fixed(p.cn0, 1),
            p.ch
        ),
    );
    let channel = match p.chan("mid") {
        None => (RecVal::Str("mid".into()), "mono file".to_string()),
        Some(mid) => {
            let b = p.ch.as_str();
            let cb = p.chan(b).unwrap().cn0;
            let cm = mid.cn0;
            let n_of = |k: &str| p.chan(k).map_or(0, |c| c.n);
            let side: Vec<&str> = [("L", "R"), ("R", "L")]
                .iter()
                .filter(|(k, o)| n_of(k) > 0 && n_of(o) == 0)
                .map(|(k, _)| *k)
                .collect();
            if let Some(&sd) = side.first() {
                (
                    RecVal::Str(sd.into()),
                    format!(
                        "line in {sd} only ({} none): mid adds the other track's noise; C/N0 {sd} {} vs mid {} dB-Hz",
                        if sd == "L" { "R" } else { "L" },
                        fmt_opt(p.chan(sd).unwrap().cn0),
                        fmt_opt(cm)
                    ),
                )
            } else if b != "mid"
                && (cm.is_none()
                    || cb.unwrap() - cm.unwrap() >= 3.0
                    || mid.lock < p.chan(b).unwrap().lock)
            {
                (
                    RecVal::Str(b.into()),
                    format!(
                        "C/N0 {b} {} vs mid {} dB-Hz, lock blocks {} vs {}",
                        fmt_opt(cb),
                        fmt_opt(cm),
                        p.chan(b).unwrap().lock,
                        mid.lock
                    ),
                )
            } else {
                (
                    RecVal::Str("mid".into()),
                    format!(
                        "mid {} dB-Hz within 3 dB of best ({b} {})",
                        fmt_opt(cm),
                        fmt_opt(cb)
                    ),
                )
            }
        }
    };
    let tv: Vec<&Cand> = cs
        .iter()
        .filter(|c| {
            c.cls == "TV line"
                && c.id != p.id
                && usable(c)
                && !c.coh.as_ref().is_some_and(|k| !k.tape)
        })
        .collect();
    let refn = rf.num().unwrap();
    let mut abs: RecItem = if p.cls != "TV line" && !tv.is_empty() {
        let t = first_max(tv.iter().copied(), |a, b| score(a) > score(b)).unwrap();
        let ft = implied_true(p, t);
        let mut txt = format!(
            "#{} {} at {} Hz, {}/{} blocks: receiving-site line, independent of the RF path",
            t.id,
            t.label,
            fmt_fixed(t.f, 1),
            t.lock_blocks,
            t.nb
        );
        if let Some(k) = &t.coh
            && !k.coh.is_empty()
        {
            txt += &format!(
                "; coherence 0.5-6 / 6-25 Hz {} / {}",
                fmt_fixed(k.get("0.5-6").unwrap_or(f64::NAN), 3),
                fmt_fixed(k.get("6-25").unwrap_or(f64::NAN), 3)
            );
        }
        if let Some(ft) = ft.filter(|&v| v != 0.0) {
            txt += &format!(
                "; implies ref true {} Hz ({} % off nominal)",
                fmt_fixed(ft, 1),
                sign_f((ft / refn - 1.0) * 100.0, 2)
            );
        }
        (
            RecVal::Str(format!("line:{}", fmt_general(t.nominal.unwrap(), 6))),
            txt,
        )
    } else if p.cls == "FM pilot" {
        (
            RecVal::Str("ref".into()),
            "FM pilot is broadcast-locked (19 kHz +-2 Hz)".into(),
        )
    } else if p.cls == "AM beat" {
        (
            RecVal::Str("ref".into()),
            "valid only if both transmitters are on nominal (old or offshore AM transmitters can be off by tens of Hz); no second line to check it: --abs none keeps the average pitch".into(),
        )
    } else if p.cls == "TV line" {
        (
            RecVal::Str("ref".into()),
            "TV line frequency is a broadcast standard".into(),
        )
    } else {
        if p.label.contains("outside --search") {
            r_ref.1 += "; may be a named line at a larger offset (label): then --ref <nominal> with a wider --search, unverified";
        }
        (
            RecVal::Str("none".into()),
            "true frequency of an unknown line is unknown: correct only the variations".into(),
        )
    };
    let mut check_line: Option<RecItem> = None;
    if let Some(pr) = &s.pair {
        if pr.found {
            let f = pr.sol.as_ref().unwrap();
            let a0 = abs.0.as_str().to_string();
            let line_alt = a0.starts_with("line:").then_some(a0);
            let mut txt = format!(
                "beat pair {} / {} Hz true (d {} Hz), {}/{} trials agree, f_true spread {} ppm; implies ref true {} Hz ({} % off nominal)",
                fmt_fixed(f.f_lo, 2),
                fmt_fixed(f.f_hi, 2),
                fmt_fixed(f.delta, 2),
                pr.n_agree.unwrap(),
                pr.n_trials.unwrap(),
                fmt_fixed(f.f_true_spread_ppm, 0),
                fmt_fixed(f.f_true, 1),
                sign_f((f.f_true / pr.hz - 1.0) * 100.0, 2)
            );
            if let Some(x) = f.xcheck_ppm {
                txt += &format!("; TV line cross-check {} ppm", sign_f(x, 0));
            }
            if let Some(la) = &line_alt {
                txt += &format!("; alternative --abs {la}");
            }
            abs = (
                RecVal::Str(if pr.hz != 9000.0 {
                    format!("pair:{}", fmt_general(pr.hz, 6))
                } else {
                    "pair".into()
                }),
                txt,
            );
            if let Some(la) = line_alt {
                let v: f64 = la.split(':').nth(1).unwrap().parse().unwrap();
                check_line = Some((
                    RecVal::Float(v),
                    "TV line: independent cross-check of the pair L1 (both should agree)".into(),
                ));
            }
        } else {
            abs.1 += &format!(
                "; --abs pair: no partner ({})",
                pr.why.clone().unwrap_or_default()
            );
        }
    }
    let abs_line = abs.0.as_str().starts_with("line:");
    // UX: a check line must share the tape's speed modulation, else the run's coherence
    // evidence would limit the correction band; skip lines the survey measured as not doing so
    let not_tape = |c: &Cand| {
        c.coh.as_ref().is_some_and(|k| !k.tape)
            || matches!(c.wow_corr, Some(Some(w)) if w < super::detect::FREE_CORR)
    };
    let oth: Vec<&Cand> = cs
        .iter()
        .filter(|c| {
            usable(c)
                && c.id != p.id
                && refval(c, play) != rf
                && !(abs_line && c.cls == "TV line")
                && !not_tape(c)
        })
        .collect();
    if check_line.is_none() {
        check_line = Some(if abs_line {
            (
                RecVal::None,
                "not needed: line: already gives the two-line coherence".into(),
            )
        } else if let Some(o) = oth.first() {
            (
                refval(o, play),
                format!(
                    "#{} {} at {} Hz: coherence with the ref separates tape from RF-path FM{}",
                    o.id,
                    o.label,
                    fmt_fixed(o.f, 1),
                    if matches!(o.tape, Some(Some(true))) {
                        " (survey: shared speed modulation)"
                    } else {
                        ""
                    }
                ),
            )
        } else {
            (
                RecVal::None,
                "no second usable line: tape origin of the FM rests on the fading metric only"
                    .into(),
            )
        });
    }
    let (mut search, mut lowpass, mut auto_notch) = (None, None, None);
    let nom = p.nominal.filter(|_| p.cls != "unknown" && p.cls != "hum");
    if let Some(n) = nom {
        let need = super::search_needed(p, n, play);
        if need > s.search + 1e-12 {
            search = Some((
                RecVal::Float(need),
                format!(
                    "#{} is {} % off {} Hz (p5-p95 {}-{} Hz): outside or near the edge of --search {} %",
                    p.id,
                    sign_f((p.f / (n * play) - 1.0) * 100.0, 2),
                    fmt_general(n, 6),
                    fmt_fixed(p.spread_hz[0], 0),
                    fmt_fixed(p.spread_hz[1], 0),
                    fmt_general(s.search * 100.0, 6)
                ),
            ));
        }
    }
    if p.cls == "AM beat" {
        // what lies above ~8 kHz on a medium-wave recording is interference (the beats,
        // neighbour splatter, TV line), however loud
        let stop = 8000.0 + 500.0 / 2.0;
        let lev = level_db(&s.spec.f, &s.spec.d, stop, stop + 1000.0);
        lowpass = Some((
            RecVal::Float(8000.0),
            format!(
                "AM: no programme above ~8 kHz; removes the other beats, the TV line and splatter (background {}-{} Hz at {} dB rel. 1-2 kHz)",
                fmt_fixed(stop, 0),
                fmt_fixed(stop + 1000.0, 0),
                fmt_fixed(lev, 1)
            ),
        ));
        auto_notch = Some((
            RecVal::Str("keyed".into()),
            "AM: Morse (CW) is common on medium wave; keyed mode notches on/off keyed lines only where present, steady lines kept (keying is detected in the corrected output: the survey's uncorrected blocks cannot show it)".into(),
        ));
    }
    let cmd = {
        let mut cmd = vec![format!(
            "tapespeed {} --ref {}",
            sh_quote(&s.file),
            rf.fmt_g()
        )];
        if let Some((v, _)) = &search {
            cmd.push(format!("--search {}", v.fmt_g()));
        }
        if channel.0.as_str() != "mid" {
            cmd.push(format!("--channel {}", channel.0.as_str()));
        }
        if abs.0.as_str() != "ref" {
            cmd.push(format!("--abs {}", abs.0.as_str()));
        }
        let cl = check_line.as_ref().unwrap();
        if cl.0.num().is_some_and(|v| v != 0.0) {
            cmd.push(format!("--check-line {}", cl.0.fmt_g()));
        }
        if let Some((RecVal::Float(v), _)) = &lowpass {
            cmd.push(if *v == 8000.0 {
                "--lowpass".to_string()
            } else {
                format!("--lowpass {}", fmt_general(*v, 6))
            });
        }
        if auto_notch.is_some() {
            cmd.push("--auto-notch".into());
        }
        cmd.push(format!(
            "-o {}",
            sh_quote(&corrected_name(&s.file, s.info.lossy))
        ));
        cmd.join(" ")
    };
    Recommend {
        r#ref: r_ref,
        channel: Some(channel),
        abs: Some(abs),
        check_line,
        cmd: Some(cmd),
        search,
        lowpass,
        auto_notch,
    }
}

/// UX: the options PLAN plans when `--ref` is not given: the recommended ref, search,
/// channel, abs, check-line, lowpass and auto-notch wherever the typed value is the default
/// (a typed non-default value wins); every other option as typed.
pub fn with_recommended(a: &SurveyArgs, rec: &Recommend) -> SurveyArgs {
    let d = SurveyArgs::default();
    let mut b = a.clone();
    if let (Some((v, _)), true) = (&rec.search, a.search == d.search) {
        b.search = v.num().unwrap_or(a.search);
    }
    if let (Some((RecVal::Str(v), _)), true) = (&rec.channel, a.channel == d.channel) {
        b.channel = v.clone();
    }
    if let (Some((RecVal::Str(v), _)), true) = (&rec.abs, a.abs == d.abs) {
        b.abs = v.clone();
    }
    if let (Some((v, _)), None) = (&rec.check_line, a.check_line) {
        b.check_line = v.num().filter(|&x| x != 0.0);
    }
    if let (Some((RecVal::Float(v), _)), None) = (&rec.lowpass, a.lowpass) {
        b.lowpass = Some(*v);
    }
    if let (Some((RecVal::Str(v), _)), None) = (&rec.auto_notch, &a.auto_notch) {
        b.auto_notch = Some(v.clone());
    }
    b
}

/// `--bw` value in the resolved dict ("auto" or a number).
#[derive(Debug, Clone, PartialEq)]
pub enum BwVal {
    Auto,
    Hz(f64),
}

/// `plan(...)["resolved"]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub r#ref: f64,
    pub ref_src: String,
    pub centre: f64,
    pub play: f64,
    pub search_hz: [f64; 2],
    pub bw: BwVal,
    pub channel: String,
    pub line_hz: Option<f64>,
    pub corr_limit: String,
    pub model: String,
    pub abs_mode: String,
    pub abs_l1: f64,
    pub true_hz: f64,
    pub notch_hz: Option<f64>,
    pub lowpass: Option<f64>,
    pub extra_notches: Vec<f64>,
    pub out_len_s: f64,
    pub outputs: Vec<(String, String)>,
}

/// `plan(a, S)`: steps, warnings, resolved parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// UX: what PLAN plans ("recommended options = cmd: line" / "typed options");
    /// None: the plain header.
    pub basis: Option<String>,
    pub steps: Vec<(String, String)>,
    pub warnings: Vec<String>,
    pub resolved: Option<Resolved>,
}

/// `spans(S, idx)`: block indices → time spans "a-b s".
pub fn spans(s: &Survey, idx: &[usize]) -> String {
    if idx.is_empty() {
        return "none".into();
    }
    let bl = s.blocks_rounded();
    let mut out = Vec::new();
    let mut s0 = idx[0];
    for (k, &i) in idx.iter().enumerate() {
        let j = idx.get(k + 1).copied();
        if j != Some(i + 1) {
            out.push(format!(
                "{}-{} s",
                fmt_fixed(bl[s0].0, 0),
                fmt_fixed(bl[i].0 + bl[i].1, 0)
            ));
            if let Some(j) = j {
                s0 = j;
            }
        }
    }
    out.join(", ")
}

fn perr<E: std::fmt::Display>(e: E) -> SurveyError {
    SurveyError::Value(e.to_string())
}

fn const_curve(r: f64) -> Curve {
    let t: Vec<f64> = (0..1000).map(|i| i as f64 * 0.005).collect();
    Curve {
        ratio: vec![r; t.len()],
        confidence: vec![1.0; t.len()],
        valid: vec![true; t.len()],
        t,
        meta: Default::default(),
    }
}

fn meta_f(v: Option<&MetaValue>) -> f64 {
    match v {
        Some(MetaValue::Float(x)) => *x,
        Some(MetaValue::Int(x)) => *x as f64,
        _ => f64::NAN,
    }
}

/// `plan(a, S)`. Mutates `a.ref` when it was not given.
pub fn plan(a: &mut SurveyArgs, s: &Survey, rec: &Recommend) -> Res<Plan> {
    let inf = &s.info;
    let mut w: Vec<String> = Vec::new();
    let mut st: Vec<(String, String)> = Vec::new();
    let nb = s.blocks.len();
    let (dur, sr) = (inf.duration, inf.sr as f64);
    let mut src = "given".to_string();
    if a.r#ref.is_none() {
        let Some(r) = rec.r#ref.0.num() else {
            return Ok(Plan {
                basis: None,
                steps: vec![],
                warnings: vec![
                    "no usable reference line found: --ref required, run would fail or track noise"
                        .into(),
                ],
                resolved: None,
            });
        };
        a.r#ref = Some(r);
        src = format!(
            "from survey #{} (--ref not given)",
            s.primary.map_or("None".into(), |v| v.to_string())
        );
    }
    let aref = a.r#ref.unwrap();
    let ra = a.run_args();
    let pp = resolve(&ra).map_err(perr)?;
    let (lo, hi) = (pp.centre * (1.0 - a.search), pp.centre * (1.0 + a.search));
    let rc = find(s, pp.centre, a.search);
    let ch_ = if s.channels.iter().any(|c| c == "mid") {
        a.channel.as_str()
    } else {
        "mono"
    };
    let cliff_hz = s.bandwidth.cliff_hz;
    let g = |v: f64| fmt_general(v, 6);
    let lock_miss = |c: &Cand| -> Vec<usize> {
        (0..nb)
            .filter(|&i| c.block(i).map_or(-99.0, |e| e.cn0) < LOCK_DB)
            .collect()
    };
    if hi >= sr / 2.0 {
        w.push(format!(
            "search window {}..{} Hz reaches Nyquist {} Hz: ref not representable at sr {}",
            fmt_fixed(lo, 0),
            fmt_fixed(hi, 0),
            g(sr / 2.0),
            inf.sr
        ));
    }
    if let Some(ch) = cliff_hz
        && pp.centre >= ch
    {
        w.push(format!(
            "ref centre {} Hz above the spectrum cliff {} Hz (codec lowpass): line removed by the encoder",
            fmt_fixed(pp.centre, 0),
            fmt_fixed(ch, 0)
        ));
    }
    match rc {
        None => {
            w.push(format!(
                "ref {} Hz: no line in the survey within {}..{} Hz",
                g(aref),
                fmt_fixed(lo, 0),
                fmt_fixed(hi, 0)
            ));
            if pp.play != 1.0
                && let Some(alt) = find(s, aref, a.search)
            {
                w.push(format!(
                    "line #{} at {} Hz sits near ref, not near ref x play {}: check --play-speed/--tape-speed",
                    alt.id,
                    fmt_fixed(alt.f, 1),
                    fmt_fixed(pp.play, 4)
                ));
            }
        }
        Some(rc) => {
            let sel = rc.chan(ch_).and_then(|c| c.cn0);
            let best = rc.chan(&rc.ch).unwrap().cn0.unwrap();
            if sel.is_none_or(|v| best - v >= 3.0) {
                w.push(format!(
                    "--channel {}: ref C/N0 {} vs {} dB-Hz in {}: use --channel {}",
                    a.channel,
                    sel.map_or("absent".into(), repr_shortest),
                    repr_shortest(best),
                    rc.ch,
                    chan_name(&rc.ch)
                ));
            }
            if rc.lock_blocks < nb {
                let miss = lock_miss(rc);
                w.push(format!(
                    "ref below {} dB-Hz in {}/{nb} blocks ({}): long-gap fallback (L1 only) there",
                    g(LOCK_DB),
                    miss.len(),
                    spans(s, &miss)
                ));
            }
            if rc.codec_risk && !matches!(rc.tape, Some(Some(true))) {
                w.push(format!(
                    "ref #{} >= 15 kHz in a lossy file without shared-wow evidence: may be a codec artifact",
                    rc.id
                ));
            }
            if rc.cls == "AM beat" && pp.abs.mode() == "ref" {
                w.push(format!(
                    "--abs ref on an AM beat: L1 assumes both transmitters on nominal (old or offshore AM transmitters can be off by tens of Hz); measured {} Hz",
                    fmt_fixed(rc.f, 1)
                ));
            }
            let oth: Vec<&Cand> = s
                .candidates
                .iter()
                .filter(|c| c.id != rc.id && usable(c) && (c.f / pp.centre - 1.0).abs() <= a.search)
                .collect();
            if !oth.is_empty() {
                w.push(format!(
                    "other line(s) in the search window (false-lock risk): {}",
                    oth.iter()
                        .map(|c| format!(
                            "#{} {} Hz {} dB-Hz",
                            c.id,
                            fmt_fixed(c.f, 0),
                            fmt_fixed(c.cn0, 0)
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            // UX: the same rule as RECOMMEND --search (offset + wow headroom, spread)
            let need = super::search_needed(rc, pp.centre, 1.0);
            if need > a.search + 1e-12 {
                w.push(format!(
                    "ref offset {} % near the --search limit {} % (p5-p95 {}-{} Hz): use --search {}",
                    sign_f((rc.f / pp.centre - 1.0) * 100.0, 2),
                    g(a.search * 100.0),
                    fmt_fixed(rc.spread_hz[0], 0),
                    fmt_fixed(rc.spread_hz[1], 0),
                    g(need)
                ));
            } else if (rc.f / pp.centre - 1.0).abs() > 0.8 * a.search {
                w.push(format!(
                    "ref offset {} % near the --search limit {} %",
                    sign_f((rc.f / pp.centre - 1.0) * 100.0, 2),
                    g(a.search * 100.0)
                ));
            }
        }
    }
    let line_hz = pp.line_hz.filter(|&v| v != 0.0);
    let mut lc: Option<&Cand> = None;
    if let Some(lh) = line_hz {
        lc = find(s, lh * pp.play, a.search);
        let what = if pp.abs.mode() == "line" {
            "--abs line"
        } else {
            "--check-line"
        };
        if lh == aref {
            w.push(format!("{what} {} Hz equals --ref", g(lh)));
        } else if lc.is_none() {
            w.push(format!(
                "{what} {} Hz: no line in the survey{}",
                g(lh),
                if pp.abs.mode() == "line" {
                    ": line: mode will fail (too few common valid samples)"
                } else {
                    ""
                }
            ));
        } else if lc.unwrap().coh.as_ref().is_some_and(|k| !k.tape) {
            w.push(format!(
                "{what} {} Hz: survey coherence with the ref < {} (0.5-6 Hz)",
                g(lh),
                repr_shortest(COH_MIN)
            ));
        }
    }
    if pp.play != 1.0 {
        w.push(format!(
            "play factor {} ({} / {}): analyze searches at {} Hz, F is part of L1",
            fmt_fixed(pp.play, 4),
            a.play_speed.clone().unwrap_or("None".into()),
            a.tape_speed.clone().unwrap_or("None".into()),
            fmt_fixed(pp.centre, 1)
        ));
    }
    let lp = a.lowpass.filter(|&v| v != 0.0);
    if let Some(lpv) = lp {
        let stop = lowpass_stop(&ra);
        let lev = level_db(&s.spec.f, &s.spec.d, stop, stop + 1000.0);
        // UX: on an AM beat that background is interference, the reason for --lowpass
        let am = rc.is_some_and(|c| c.cls == "AM beat");
        if lev > -30.0 && !am {
            w.push(format!(
                "--lowpass {}: background {}-{} Hz at {} dB rel. 1-2 kHz (as loud as AM programme content at 7-8 kHz, typically -20..-30 dB): content would be removed",
                g(lpv),
                fmt_fixed(stop, 0),
                fmt_fixed(stop + 1000.0, 0),
                fmt_fixed(lev, 1)
            ));
        }
    }
    if let Bw::Hz(b) = pp.bw
        && b < 25.0
    {
        w.push(format!(
            "--bw {}: passband {} Hz; narrow loops lose lock under strong flutter (one cassette transfer: bw 6/12/25 gave 0.06/1.4/5.3 % valid)",
            g(b),
            fmt_fixed(0.55 * b, 1)
        ));
    }
    for (item, flag, cur) in [
        (&rec.channel, "channel", a.channel.clone()),
        (&rec.abs, "abs", a.abs.clone()),
    ] {
        if let Some((v, _)) = item
            && *v != RecVal::None
            && cur != v.to_text()
            && !matches!(v.as_str(), "ref" | "mid")
        {
            w.push(format!(
                "survey recommends --{flag} {} (see RECOMMEND)",
                v.to_text()
            ));
        }
    }
    if ch_ == "mono" && a.channel != "mid" {
        w.push(format!(
            "--channel {} on a mono file: the run warns and analyses the mono signal",
            a.channel
        ));
    }
    if let Some((RecVal::Float(v), _)) = &rec.search
        && a.search < *v
    {
        w.push(format!(
            "survey recommends --search {} (see RECOMMEND)",
            g(*v)
        ));
    }
    if let Some((RecVal::Float(v), _)) = &rec.check_line
        && a.check_line.is_none()
        && !pp.abs.mode().starts_with("line")
    {
        w.push(format!(
            "survey recommends --check-line {} (see RECOMMEND)",
            g(*v)
        ));
    }
    if let Some((RecVal::Float(_), _)) = &rec.lowpass
        && lp.is_none()
    {
        w.push("survey recommends --lowpass (see RECOMMEND)".into());
    }
    if rec.auto_notch.is_some() && a.auto_notch.as_deref().is_none_or(|v| v.is_empty()) {
        w.push("survey recommends --auto-notch (see RECOMMEND)".into());
    }
    // --- L1 from the survey frequencies (constant curves through the real apply_l1)
    let r0 = rc.map_or(pp.play, |c| c.f / aref);
    let pr = s.pair.as_ref();
    let is_pair = pp.abs.mode() == "pair";
    let pm = match (is_pair, pr, rc) {
        (true, Some(p), Some(rc)) if p.found => {
            let mut m = p.pm.clone().unwrap();
            let ft = p.sol.as_ref().unwrap().f_true;
            if let Some(sol) = m.solution.as_mut() {
                sol.s = rc.f / ft;
            }
            Some(m)
        }
        _ => None,
    };
    if is_pair && pm.is_none() {
        w.push(format!(
            "--abs pair: survey found no partner beat ({}): the run fails with the candidate list",
            match pr {
                Some(p) => p.why.clone().unwrap_or("primary not an AM beat".into()),
                None => "not checked".into(),
            }
        ));
    }
    let line_curve = match (lc, line_hz) {
        (Some(c), Some(lh)) => Some(const_curve(c.f / lh)),
        _ => None,
    };
    let (abs_mode, abs_l1, true_hz) = match apply_l1(
        &const_curve(r0),
        aref,
        &pp.abs,
        pp.play,
        line_curve.as_ref(),
        60.0,
        pm.as_ref(),
    ) {
        Ok((_, info)) => (
            match info.get("abs_mode") {
                Some(MetaValue::Str(s)) => s.clone(),
                _ => a.abs.clone(),
            },
            meta_f(info.get("abs_l1")),
            meta_f(info.get("abs_ref_true_hz")),
        ),
        Err(e) => {
            w.push(format!(
                "--abs {}: L1 not predictable from the survey ({e})",
                a.abs
            ));
            (a.abs.clone(), r0, aref)
        }
    };
    let (notch_hz, notch_why) = ref_notch(&ra, true_hz);
    let mut fa = ra.clone();
    if a.output.as_deref().is_none_or(|o| o.is_empty()) {
        fa.output = corrected_name(&a.input, s.info.lossy);
    }
    let fp = out_paths(&fa).map_err(perr)?;
    let outputs: Vec<(String, String)> = [
        ("audio", Some(fp.audio.clone())),
        ("curve", Some(fp.curve.clone())),
        ("report", Some(fp.report.clone())),
        ("plot", fp.plot.clone()),
        ("model_plot", fp.model_plot.clone()),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.map(|v| (k.to_string(), v)))
    .collect();
    let pres = rc.map_or(0.0, |c| c.lock_blocks as f64 / nb as f64);
    // --- steps
    let br = if inf.bit_rate != 0 {
        format!("{} kb/s ", fmt_fixed(inf.bit_rate as f64 / 1000.0, 0))
    } else {
        String::new()
    };
    st.push((
        "decode".into(),
        format!(
            "{} ({}) {br}{}, {} Hz, {} ch, {} s -> float32 {} MB in memory",
            inf.codec,
            inf.format,
            if inf.lossy { "lossy" } else { "lossless" },
            inf.sr,
            inf.channels,
            fmt_fixed(dur, 1),
            fmt_fixed(dur * sr * inf.channels as f64 * 4.0 / 1e6, 0)
        ),
    ));
    st.push((
        "channel".into(),
        format!(
            "{}{}{}",
            a.channel,
            if ch_ == "mono" {
                " (mono file: ignored)"
            } else {
                ""
            },
            match rc {
                Some(rc) => format!(
                    "; ref C/N0 here {} dB-Hz (best {} {})",
                    fmt_opt(rc.chan(ch_).and_then(|c| c.cn0)),
                    rc.ch,
                    fmt_opt(rc.chan(&rc.ch).unwrap().cn0)
                ),
                None => String::new(),
            }
        ),
    ));
    let bws: Vec<f64> = match pp.bw {
        Bw::Auto => BW_AUTO.to_vec(),
        Bw::Hz(b) => vec![b],
    };
    let bwtxt = match pp.bw {
        Bw::Auto => format!(
            "auto {}",
            bws.iter().map(|b| g(*b)).collect::<Vec<_>>().join("/")
        ),
        Bw::Hz(b) => g(b),
    };
    st.push((
        "analyze".into(),
        format!(
            "ref {} Hz ({src}), centre {} Hz, search +-{} % = {}..{} Hz, bw {bwtxt} Hz -> passband {} Hz, fmax 25 Hz{}{}",
            g(aref),
            fmt_fixed(pp.centre, 1),
            g(a.search * 100.0),
            fmt_fixed(lo, 0),
            fmt_fixed(hi, 0),
            bws.iter().map(|b| fmt_general(25.0f64.min(0.55 * b), 3)).collect::<Vec<_>>().join("/"),
            match rc {
                Some(rc) => format!(
                    "; survey: #{} {} Hz ({} %), {}",
                    rc.id,
                    fmt_fixed(rc.f, 1),
                    sign_f((rc.f / pp.centre - 1.0) * 100.0, 3),
                    rc.verdict
                ),
                None => String::new(),
            },
            match line_hz {
                Some(lh) => format!("; second pass {} Hz (centre {})", g(lh), fmt_fixed(lh * pp.play, 1)),
                None => String::new(),
            }
        ),
    ));
    let why = if a.model == "off" {
        "off".to_string()
    } else if dur < MODEL_MIN_S {
        format!(
            "skipped: file {} s < {} s",
            fmt_fixed(dur, 0),
            g(MODEL_MIN_S)
        )
    } else if pres < MODEL_MIN_VALID {
        format!(
            "likely skipped: ref lockable in {} % of blocks (pass-1 valid < {} % -> no model)",
            fmt_fixed(pres * 100.0, 0),
            g(MODEL_MIN_VALID * 100.0)
        )
    } else {
        format!(
            "fit L1-L3 (Ta {} s, jump test); {} (B {} Hz{}, 2 refinements), else plain + hold-out gap fill",
            g(a.model_ta),
            if a.model == "on" {
                "forced aided".to_string()
            } else {
                format!("EV >= {} -> aided", g(a.model_min_ev))
            },
            g(a.model_band),
            if pp.bw == Bw::Auto {
                ", capped at the pass-1 passband"
            } else {
                ""
            }
        )
    };
    st.push((
        "model".into(),
        format!(
            "--model {}: {why}{}",
            a.model,
            match pp.v_phys.filter(|&v| v != 0.0) {
                Some(v) => format!(
                    "; physics at {} cm/s (reel laws constrained)",
                    fmt_fixed(v * 100.0, 2)
                ),
                None => "; tape speed inferred from reel/capstan physics".into(),
            }
        ),
    ));
    st.push((
        "long gaps".into(),
        format!(
            ">= {} s without valid ref (< 2 % / 10 s): edge-linear inside, ramp to L1 at the ends, invalid + WARNING{}",
            g(LONG_GAP_S),
            match rc {
                Some(rc) if pres < 1.0 => format!("; survey: ref missing in {}", spans(s, &lock_miss(rc))),
                Some(_) => "; survey: ref in all blocks".into(),
                None => String::new(),
            }
        ),
    ));
    st.push((
        "evidence".into(),
        format!(
            "fading metric (plain path limit if rho > {}); {}; --corr-limit {}{}, --max-corr-hz {}{}",
            g(FADE_RHO),
            match line_hz {
                Some(lh) => format!(
                    "coherence with {} Hz decides (bands 0.5-6 / 6-25 Hz >= {})",
                    g(lh),
                    g(COH_MIN)
                ),
                None => "no second line".into(),
            },
            pp.corr_limit,
            if a.weak_rf { " (--weak-rf)" } else { "" },
            g(a.max_corr_hz),
            match pp.corr_limit.as_str() {
                "on" => ": curve lowpassed always (periodic model part kept)",
                "off" => ": never limited",
                _ => "",
            }
        ),
    ));
    let pair_txt = || -> String {
        let mut t = format!(
            "; f_true = 2 x {} x F_p / (F_lo + F_hi) on the pass-1-normalised spectrum (8 s Welch), |d| <= {} Hz, beat-switch repair (spans on the other beat / rho)",
            g(pp.abs.arg().unwrap_or(f64::NAN)),
            g(a.pair_dmax)
        );
        if pm.is_some() {
            let f = pr.unwrap().sol.as_ref().unwrap();
            t += &format!(
                "; survey: d {} Hz, beats {} / {} Hz",
                fmt_fixed(f.delta, 2),
                fmt_fixed(f.f_lo, 2),
                fmt_fixed(f.f_hi, 2)
            );
        }
        t
    };
    let mode_txt = match pp.abs.mode() {
        "ref" => "; L1 = ref nominal".to_string(),
        "none" => "; average speed kept, variations only".into(),
        "line" => "; L1 from r1/r2 (Huber mean)".into(),
        "assume" => "; L1 from the assumed true frequency".into(),
        "value" | "percent" => "; L1 given".into(),
        "curve" => "; slow L1(t) from CSV".into(),
        _ => pair_txt(),
    };
    st.push((
        "L1".into(),
        format!(
            "--abs {abs_mode}: mean speed ~{} ({} %), true ref ~{} Hz; play factor {}; tape speed {}{mode_txt}",
            fmt_fixed(abs_l1, 5),
            sign_f((abs_l1 - 1.0) * 100.0, 3),
            fmt_fixed(true_hz, 1),
            fmt_fixed(pp.play, 4),
            match a.tape_speed.as_deref().filter(|v| !v.is_empty()) {
                Some(t) => format!("given {t}"),
                None => "not given (inferred)".into(),
            }
        ),
    ));
    st.push((
        "correct".into(),
        format!(
            "{}-tap Kaiser sinc, one resampling; length {} s -> ~{} s",
            a.taps,
            fmt_fixed(dur, 1),
            fmt_fixed(dur * abs_l1, 1)
        ),
    ));
    st.push((
        "notch".into(),
        match notch_hz.filter(|&v| v != 0.0) {
            Some(n) => format!("reference {} Hz (Q {})", fmt_fixed(n, 1), g(a.notch_q)),
            None => format!("reference {notch_why}"),
        },
    ));
    st.push((
        "lowpass".into(),
        match lp {
            Some(l) => format!(
                "{} Hz, +-{} Hz, 80 dB, stopband >= {} Hz",
                g(l),
                g(a.lowpass_width / 2.0),
                g(lowpass_stop(&ra))
            ),
            None => "off".into(),
        },
    ));
    let mn_list = manual_notches(&ra).map_err(perr)?;
    let mn: Vec<String> = mn_list
        .iter()
        .map(|&h| match notch_skip(&ra, h, notch_hz) {
            Some(why) => format!("{} Hz skipped ({why})", g(h)),
            None => format!("{} Hz whole file", g(h)),
        })
        .collect();
    st.push((
        "notches".into(),
        format!(
            "{}{}",
            match a.auto_notch.as_deref().filter(|v| !v.is_empty()) {
                Some(an) => format!(
                    "--auto-notch {an}: lines >= {} Hz in the output, width {} Hz",
                    g(a.auto_notch_min),
                    g(a.notch_width)
                ),
                None => "auto-notch off".into(),
            },
            if mn.is_empty() {
                String::new()
            } else {
                format!("; --notch-at {}", mn.join(", "))
            }
        ),
    ));
    let has_out = a.output.as_deref().is_some_and(|o| !o.is_empty());
    st.push((
        "outputs".into(),
        format!(
            "{}{}",
            outputs
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(" "),
            if has_out {
                ""
            } else {
                "  (-o not given: example)"
            }
        ),
    ));
    // min/h measured on a 43 min stereo MP3 (bench/results.md §2/§3: ref, line + lowpass,
    // pair + auto-notch), 10 threads and one thread; a second line / pair add their share.
    let (rt, rt1) = {
        let (b, b1) = if line_hz.is_some() {
            (0.620, 1.643)
        } else {
            (0.447, 1.128)
        };
        if is_pair {
            (b + 0.245, b1 + 1.260)
        } else {
            (b, b1)
        }
    };
    st.push((
        "runtime".into(),
        format!(
            "~{} min ({} min/h measured{}{}, M1 Max 10 threads; one thread ~{} min); memory {}",
            fmt_fixed(dur / 3600.0 * rt, 1),
            g(rt),
            if line_hz.is_some() {
                ", second line"
            } else {
                ""
            },
            if is_pair { ", pair" } else { "" },
            fmt_fixed(dur / 3600.0 * rt1, 1),
            match a.mem_peak {
                Some(b) => format!(
                    "~{} GiB (planner peak estimate, see memory:)",
                    fmt_fixed(b as f64 / (1u64 << 30) as f64, 1)
                ),
                None => "see memory:".into(),
            }
        ),
    ));
    let resolved = Resolved {
        r#ref: aref,
        ref_src: src,
        centre: pp.centre,
        play: pp.play,
        search_hz: [lo, hi],
        bw: match pp.bw {
            Bw::Auto => BwVal::Auto,
            Bw::Hz(b) => BwVal::Hz(b),
        },
        channel: a.channel.clone(),
        line_hz: pp.line_hz,
        corr_limit: pp.corr_limit.clone(),
        model: a.model.clone(),
        abs_mode,
        abs_l1,
        true_hz,
        notch_hz,
        lowpass: a.lowpass,
        extra_notches: mn_list
            .into_iter()
            .filter(|&h| notch_skip(&ra, h, notch_hz).is_none())
            .collect(),
        out_len_s: dur * abs_l1,
        outputs,
    };
    let _ = play_factor; // resolved through `resolve`
    Ok(Plan {
        basis: None,
        steps: st,
        warnings: w,
        resolved: Some(resolved),
    })
}

/// Nominal table check used by the report's LIMITS section.
pub fn nominal_table() -> &'static [(&'static str, f64, &'static str)] {
    &NOMINAL
}

#[cfg(test)]
mod ux_tests {
    use super::*;

    #[test]
    fn corrected_name_never_the_input() {
        assert_eq!(corrected_name("/a/b/x.flac", false), "x_corrected.flac");
        assert_eq!(corrected_name("x.wav", false), "x_corrected.wav");
        assert_eq!(corrected_name("rec.mp3", true), "rec_corrected.flac");
        assert_eq!(corrected_name("noext", false), "noext_corrected.flac");
        assert_eq!(sh_quote("it's.mp3"), "'it'\\''s.mp3'");
    }

    #[test]
    fn search_needed_covers_offset_and_spread() {
        let c = |f: f64, lo: f64, hi: f64| Cand {
            f,
            spread_hz: [lo, hi],
            ..test_cand()
        };
        // 9 kHz beat at -3.12 %: 3.62 % -> 0.04
        assert_eq!(
            super::super::search_needed(&c(8719.0, 8700.0, 8740.0), 9000.0, 1.0),
            0.04
        );
        // +4.19 %, p95 +4.36 %: 4.69 % -> 0.05
        assert_eq!(
            super::super::search_needed(&c(9377.0, 9287.0, 9392.0), 9000.0, 1.0),
            0.05
        );
        // -1 %, narrow: within the default
        assert_eq!(
            super::super::search_needed(&c(8910.0, 8900.0, 8920.0), 9000.0, 1.0),
            0.02
        );
        // spread dominates: -2.69 %, p5 -3.83 %
        assert_eq!(
            super::super::search_needed(&c(8758.0, 8655.3, 8805.8), 9000.0, 1.0),
            0.05
        );
    }

    fn test_cand() -> Cand {
        Cand {
            f: 0.0,
            cls: "AM beat".into(),
            label: String::new(),
            nominal: Some(9000.0),
            codec_risk: false,
            ch: "mono".into(),
            low: false,
            n: 1,
            nb: 1,
            presence: 1.0,
            lock_blocks: 1,
            prom: 10.0,
            cn0: 30.0,
            pers: 1.0,
            wow_pct: 0.1,
            spread_hz: [0.0, 0.0],
            spread_pct: 0.0,
            blocks: vec![],
            channels: vec![],
            per: vec![],
            ratio: None,
            offset_pct: None,
            sigma25_pct: 0.0,
            wow_corr: None,
            tape: None,
            primary: false,
            verdict: String::new(),
            trial_valid: None,
            coh: None,
            id: 1,
        }
    }
}
