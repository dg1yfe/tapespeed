//! absolute: absolute speed L1, decoupled from the variations.
//!
//! Curve ratio `r(t) = f_apparent / ref_hz`. If the tracked line's true frequency is `f_true`, the
//! speed is `s(t) = r(t) * ref_hz / f_true`; only the level `L1 = mean(s)` depends on the mode
//! ([`Abs`]). `pair[:HZ]` measures `f_true` from both adjacent-channel beats ([`pair`]); pass-1
//! spans locked onto the other beat are divided out by [`repair::switch_repair`].
//!
//! Numerics use a fixed operation order (pairwise sums, float32 segment spectra), so results
//! are reproducible to rounding (abs_l1 within 1e-8 of the stored reference vectors).

pub mod pair;
pub mod repair;

use indexmap::IndexMap;
use ts_dsp::ndimage::{Mode, uniform_filter1d};
use ts_dsp::stats::{interp, mean, median, np_sum, ptp, std};
use ts_io::fmt_general;

use crate::curve::{Curve, CurveError, MetaValue, read_curve};
pub use pair::{
    Candidate, PairParams, PairResult, PairSolution, Side, XCheck, pair_info, pair_measure,
    pair_table,
};
pub use repair::{SwitchInfo, ghost_rho, switch_repair};

/// Info dict of [`apply_l1`] / [`pair_info`] (insertion-ordered; key order and value types
/// are part of the output format).
pub type Info = IndexMap<String, MetaValue>;

/// Cross-check lines with a known true frequency (TV line PAL/NTSC, FM pilot) [Hz].
pub const XCHECK: [f64; 3] = [15625.0, 15734.0, 19000.0];

/// Errors with their class (value / runtime) and message.
#[derive(Debug, thiserror::Error)]
pub enum AbsError {
    /// Invalid value (bad specification or argument).
    #[error("{0}")]
    Value(String),
    /// Measurement failed at run time.
    #[error("{0}")]
    Runtime(String),
    #[error("{0}")]
    Curve(#[from] CurveError),
    #[error("{0}")]
    Dsp(#[from] ts_dsp::ValueError),
}

/// `--abs` mode with its argument (`parse_abs` result `(mode, arg)`).
#[derive(Debug, Clone, PartialEq)]
pub enum Abs {
    Ref,
    None,
    Assume(f64),
    Line(f64),
    Value(f64),
    Percent(f64),
    Curve(String),
    Pair(f64),
}

impl Abs {
    /// Mode name as written to the curve meta.
    pub fn mode(&self) -> &'static str {
        match self {
            Abs::Ref => "ref",
            Abs::None => "none",
            Abs::Assume(_) => "assume",
            Abs::Line(_) => "line",
            Abs::Value(_) => "value",
            Abs::Percent(_) => "percent",
            Abs::Curve(_) => "curve",
            Abs::Pair(_) => "pair",
        }
    }

    /// Numeric `arg` (None for ref / none / curve).
    pub fn arg(&self) -> Option<f64> {
        match self {
            Abs::Assume(v) | Abs::Line(v) | Abs::Value(v) | Abs::Percent(v) | Abs::Pair(v) => {
                Some(*v)
            }
            _ => None,
        }
    }

    /// `abs_mode` meta value: `mode` or `f"{mode}:{arg}"` (float `str`).
    pub fn label(&self) -> String {
        match self {
            Abs::Ref | Abs::None => self.mode().to_string(),
            Abs::Curve(p) => format!("curve:{p}"),
            _ => format!(
                "{}:{}",
                self.mode(),
                ts_io::repr_shortest(self.arg().unwrap())
            ),
        }
    }
}

/// Quoted string representation: single quotes unless the text contains `'` and no `"`,
/// backslash escapes for control characters, printable non-ASCII kept.
pub fn quote_str(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut o = String::with_capacity(s.len() + 2);
    o.push(q);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\t' => o.push_str("\\t"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            c if c == q => {
                o.push('\\');
                o.push(c);
            }
            c if (c as u32) < 0x20
                || c as u32 == 0x7f
                || ((c as u32) >= 0x80 && (c as u32) < 0xa0) =>
            {
                o.push_str(&format!("\\x{:02x}", c as u32))
            }
            c if c.is_control() => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

/// Lenient float parse: surrounding whitespace, `_` between digits, inf/nan words; error
/// message `could not convert string to float: '...'`.
pub fn parse_float(s: &str) -> Result<f64, String> {
    let fail = || format!("could not convert string to float: {}", quote_str(s));
    let t = s.trim();
    let b = t.as_bytes();
    for (i, &ch) in b.iter().enumerate() {
        if ch == b'_'
            && !(i > 0 && b[i - 1].is_ascii_digit() && i + 1 < b.len() && b[i + 1].is_ascii_digit())
        {
            return Err(fail());
        }
    }
    let u: String = t.chars().filter(|&c| c != '_').collect();
    if u.is_empty() || !u.is_ascii() {
        return Err(fail());
    }
    u.parse::<f64>().map_err(|_| fail())
}

/// `parse_abs(spec)`: `'ref' | 'none' | 'assume:HZ' | 'line:HZ' | 'value:R' | 'percent:P' |
/// 'curve:PATH' | 'pair[:HZ]'`.
pub fn parse_abs(spec: &str) -> Result<Abs, AbsError> {
    let (mode, arg) = spec.split_once(':').unwrap_or((spec, ""));
    let num = |a: &str| parse_float(a).map_err(AbsError::Value);
    match mode {
        "ref" if arg.is_empty() => return Ok(Abs::Ref),
        "none" if arg.is_empty() => return Ok(Abs::None),
        "pair" => return Ok(Abs::Pair(if arg.is_empty() { 9000.0 } else { num(arg)? })),
        "assume" if !arg.is_empty() => return Ok(Abs::Assume(num(arg)?)),
        "line" if !arg.is_empty() => return Ok(Abs::Line(num(arg)?)),
        "value" if !arg.is_empty() => return Ok(Abs::Value(num(arg)?)),
        "percent" if !arg.is_empty() => return Ok(Abs::Percent(num(arg)?)),
        "curve" if !arg.is_empty() => return Ok(Abs::Curve(arg.to_string())),
        _ => {}
    }
    Err(AbsError::Value(format!(
        "--abs {}: ref | none | assume:HZ | line:HZ | value:R | percent:P | curve:PATH | pair[:HZ]",
        quote_str(spec)
    )))
}

/// `np.minimum(1, x)` (NaN propagates).
#[inline]
fn np_min1(x: f64) -> f64 {
    if x.is_nan() || x < 1.0 { x } else { 1.0 }
}

/// `huber_mean(v, k, it)`: Huber M-estimate of the location (median start, MAD scale).
pub fn huber_mean(v: &[f64], k: f64, it: usize) -> f64 {
    let mut m = median(v);
    for _ in 0..it {
        let e: Vec<f64> = v.iter().map(|&x| x - m).collect();
        let ae: Vec<f64> = e.iter().map(|x| x.abs()).collect();
        let s = 1.4826 * median(&ae) + 1e-15;
        let w: Vec<f64> = ae.iter().map(|&a| np_min1(k * s / (a + 1e-30))).collect();
        let we: Vec<f64> = w.iter().zip(&e).map(|(a, b)| a * b).collect();
        m += np_sum(&we) / np_sum(&w);
    }
    m
}

/// `line_compare(c1, c2, blk)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct LineCompare {
    pub line_valid: f64,
    pub common: f64,
    /// `(q, q_std_pct, q_drift_pct, n_blocks)`; None if fewer than 200 common valid samples.
    pub q: Option<(f64, f64, f64, usize)>,
}

fn bool_mean(v: &[bool]) -> f64 {
    v.iter().filter(|&&b| b).count() as f64 / v.len() as f64
}

/// `line_compare(c1, c2, blk)`: primary vs second-line curve; `q = r1/r2` on common valid
/// samples (Huber mean), spread and slow drift (ptp of `blk`-s block medians).
pub fn line_compare(c1: &Curve, c2: &Curve, blk: f64) -> LineCompare {
    let n = c1.t.len().min(c2.t.len());
    let com: Vec<bool> = (0..n).map(|i| c1.valid[i] && c2.valid[i]).collect();
    let mut out = LineCompare {
        line_valid: bool_mean(&c2.valid),
        common: bool_mean(&com),
        q: None,
    };
    if com.iter().filter(|&&b| b).count() < 200 {
        return out;
    }
    let q: Vec<f64> = (0..n).map(|i| c1.ratio[i] / c2.ratio[i]).collect();
    let qc: Vec<f64> = (0..n).filter(|&i| com[i]).map(|i| q[i]).collect();
    let qm = huber_mean(&qc, 1.5, 5);
    let nb = (blk / (c1.t[1] - c1.t[0])) as usize;
    let mut med = Vec::new();
    for i in (0..n).step_by(nb.max(1)) {
        let e = (i + nb).min(n);
        if bool_mean(&com[i..e]) > 0.3 {
            let v: Vec<f64> = (i..e).filter(|&j| com[j]).map(|j| q[j]).collect();
            med.push(median(&v));
        }
    }
    let drift = if med.len() > 1 {
        ptp(&med).unwrap() * 100.0
    } else {
        0.0
    };
    out.q = Some((qm, std(&qc) * 100.0, drift, med.len()));
    out
}

/// `apply_l1(c, ref_hz, mode, arg, play, line_curve, window, pair)` → (speed ratio `s` on
/// `c.t`, info). `pair`: [`pair_measure`] result (needs a solution).
pub fn apply_l1(
    c: &Curve,
    ref_hz: f64,
    abs: &Abs,
    play: f64,
    line_curve: Option<&Curve>,
    window: f64,
    pair: Option<&PairResult>,
) -> Result<(Vec<f64>, Info), AbsError> {
    let r = &c.ratio;
    let r0 = mean(r);
    let mut info = Info::new();
    info.insert("abs_mode".into(), abs.label().into());
    info.insert("play_factor".into(), play.into());
    let s: Vec<f64> = match abs {
        Abs::Ref => r.clone(),
        Abs::Assume(arg) => r.iter().map(|&v| v * ref_hz / arg).collect(),
        Abs::Line(arg) => {
            let lc_curve = line_curve.ok_or_else(|| {
                AbsError::Value(format!(
                    "line {}: no second-line curve",
                    fmt_general(*arg, 6)
                ))
            })?;
            let lc = line_compare(c, lc_curve, 60.0);
            info.insert("line_line_valid".into(), lc.line_valid.into());
            info.insert("line_common".into(), lc.common.into());
            if let Some((q, sd, dr, nb)) = lc.q {
                info.insert("line_q".into(), q.into());
                info.insert("line_q_std_pct".into(), sd.into());
                info.insert("line_q_drift_pct".into(), dr.into());
                info.insert("line_n_blocks".into(), nb.into());
            }
            info.insert("line_hz".into(), (*arg).into());
            let Some((q, ..)) = lc.q else {
                return Err(AbsError::Runtime(format!(
                    "line {} Hz: too few samples valid in both lines (line valid {} %, common {} %)",
                    fmt_general(*arg, 6),
                    ts_io::fmt_fixed(lc.line_valid * 100.0, 1),
                    ts_io::fmt_fixed(lc.common * 100.0, 1)
                )));
            };
            info.insert("line_disagree_pct".into(), ((q - 1.0) * 100.0).into());
            r.iter().map(|&v| v / q).collect()
        }
        Abs::Value(_) | Abs::Percent(_) | Abs::None => {
            let l1 = match abs {
                Abs::Value(a) => *a,
                Abs::Percent(a) => 1.0 + a / 100.0,
                _ => 1.0,
            };
            r.iter().map(|&v| v * play * l1 / r0).collect()
        }
        Abs::Pair(_) => {
            let pm =
                pair.ok_or_else(|| AbsError::Runtime("--abs pair: no pair measurement".into()))?;
            let sol = pm.solution.as_ref().ok_or_else(|| {
                AbsError::Runtime(
                    pm.error
                        .clone()
                        .unwrap_or_else(|| "--abs pair: no pair measurement".into()),
                )
            })?;
            // ratio-only f_true: independent of the normalising curve's level errors
            let ft = sol.f_true;
            info.extend(pair_info(pm));
            r.iter().map(|&v| v * ref_hz / ft).collect()
        }
        Abs::Curve(path) => {
            let lc = read_curve(path)?;
            let l1 = interp(&c.t, &lc.t, &lc.ratio)?;
            let k = r
                .len()
                .min(((window / (c.t[1] - c.t[0])) as i64).max(1) as usize);
            let uf = uniform_filter1d(r, k, Mode::Reflect)?;
            info.insert("abs_curve".into(), path.as_str().into());
            (0..r.len())
                .map(|i| r[i] * (play * l1[i]) / uf[i])
                .collect()
        }
    };
    let l1 = mean(&s);
    info.insert("abs_l1".into(), l1.into());
    info.insert("abs_l1_pct".into(), ((l1 - 1.0) * 100.0).into());
    info.insert("abs_ref_true_hz".into(), (ref_hz * r0 / l1).into());
    Ok((s, info))
}
