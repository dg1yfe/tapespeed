//! Beat-switch repair: min-cost beat assignment (`viterbi`), ghost-beat ratio (`ghost_rho`)
//! and span repair (`switch_repair`).
//!
//! Pass 1 locks onto whichever beat of the pair is stronger. Both beats carry the same tape
//! modulation, so a span on the other beat measures `rho^+-1 x` the speed (`rho = F_hi /
//! F_lo`). Stage 1: 2-state Viterbi on per-block prominence differences (5 s blocks).
//! Stage 2: fine segmentation (0.25 s) of the curve level relative to a 60 s trend. The
//! majority state is kept; the other is divided out.

use ts_dsp::ndimage::{Mode, median_filter};
use ts_dsp::stats::{mean, median};
use ts_io::fmt_fixed;

use super::AbsError;
use super::pair::{Normalised, PairResult, floor, peak};
use crate::curve::Curve;
use crate::hetdec::Channel;

/// Label blocks [s], Welch segments [s] (0.5 Hz bins).
pub const SW_BLK: f64 = 5.0;
pub const SW_SEG: f64 = 2.0;
/// [dB] per switch (block labels).
pub const SW_COST: f64 = 12.0;
/// [s] fine level resolution, trend window.
pub const SW_FINE: f64 = 0.25;
pub const SW_TREND: f64 = 60.0;
/// Level noise floor (log), switch cost in sigma^2 units.
pub const SW_SIG: f64 = 5e-4;
pub const SW_FCOST: f64 = 8.0;

/// `viterbi(C, cost)`: min-cost state path through costs `c` (`n` rows of `m`) with a
/// switch cost. Ties: first state (numpy argmin).
pub fn viterbi(c: &[Vec<f64>], cost: f64) -> Vec<usize> {
    let n = c.len();
    if n == 0 {
        return vec![];
    }
    let m = c[0].len();
    let mut acc = c[0].clone();
    let mut bk = vec![vec![0usize; m]; n];
    let mut tr = vec![0.0; m];
    for i in 1..n {
        let mut nacc = vec![0.0; m];
        for a in 0..m {
            for (b, t) in tr.iter_mut().enumerate() {
                *t = acc[b] + cost * if a == b { 0.0 } else { 1.0 };
            }
            let j = ts_dsp::stats::argmin(&tr).unwrap();
            bk[i][a] = j;
            nacc[a] = tr[j] + c[i][a];
        }
        acc = nacc;
    }
    let mut st = vec![0usize; n];
    st[n - 1] = ts_dsp::stats::argmin(&acc).unwrap();
    for i in (1..n).rev() {
        st[i - 1] = bk[i][st[i]];
    }
    st
}

/// `ghost_rho(res, tol)`: two candidates `F_a < F_p < F_b` with `F_a F_b = F_p^2` (+-tol Hz at
/// `F_b`) → `rho = sqrt(F_b / F_a)`, else None.
pub fn ghost_rho(res: &PairResult, tol: f64) -> Option<f64> {
    let mut fs: Vec<f64> = res.candidates.iter().map(|e| e.f).collect();
    fs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let fp = res.f_primary;
    let mut best: Option<(f64, f64)> = None;
    for &a in fs.iter().filter(|&&f| f < fp) {
        for &b in fs.iter().filter(|&&f| f > fp) {
            let e = (b - fp * fp / a).abs();
            if e <= tol && best.is_none_or(|(be, _)| e < be) {
                best = Some((e, (b / a).sqrt()));
            }
        }
    }
    // no best candidate, or a zero rho: no ghost ratio
    best.map(|b| b.1).filter(|&r| r != 0.0)
}

/// `switch_repair` info (`sw_*` keys).
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchInfo {
    pub sw_rho: f64,
    pub sw_spans: usize,
    pub sw_s: f64,
    /// `lo` | `hi` (empty if not run).
    pub sw_major: String,
    /// Set when the segmentation ran.
    pub sw_sigma_pct: Option<f64>,
    pub sw_list: Option<String>,
    /// Spans `(t_start, t_end)` on the minority beat (all of them; `sw_list` keeps 200).
    pub spans: Vec<(f64, f64)>,
}

impl SwitchInfo {
    /// `rep["switch"]` as a JSON value (key order is part of the report format).
    pub fn to_value(&self) -> ts_io::json::Value {
        let mut v = ts_io::json::Value::obj()
            .with("sw_rho", self.sw_rho)
            .with("sw_spans", self.sw_spans)
            .with("sw_s", self.sw_s)
            .with("sw_major", self.sw_major.as_str());
        if let Some(s) = self.sw_sigma_pct {
            v.push("sw_sigma_pct", s);
        }
        if let Some(l) = &self.sw_list {
            v.push("sw_list", l.as_str());
        }
        v
    }
}

/// `switch_repair(x, sr, c, ref_hz, rho, channel, blk_s, seg_s)`: curve `c` tracking one
/// beat of a pair with `rho = F_hi / F_lo > 1`, maybe switching to the other → (curve
/// normalised to the majority beat, info).
#[allow(clippy::too_many_arguments)]
pub fn switch_repair(
    x: &[f32],
    ch: usize,
    sr: f64,
    c: &Curve,
    ref_hz: f64,
    rho: f64,
    channel: Channel,
    blk_s: f64,
    seg_s: f64,
) -> Result<(Curve, SwitchInfo), AbsError> {
    let r = &c.ratio;
    let mut info = SwitchInfo {
        sw_rho: rho,
        sw_spans: 0,
        sw_s: 0.0,
        sw_major: String::new(),
        sw_sigma_pct: None,
        sw_list: None,
        spans: vec![],
    };
    if rho == 0.0 || rho.is_nan() || rho <= 1.001 {
        return Ok((c.clone(), info));
    }
    let fc = ref_hz * mean(r);
    let half = fc * (rho - 1.0) * 1.3 + 20.0;
    let sp = Normalised::new(x, ch, sr, fc, half, c, channel)?.spectra(
        1.0,
        seg_s,
        Some(blk_s),
        Some(half),
    );
    let b = sp.blocks;
    if b.len() < 2 {
        return Ok((c.clone(), info));
    }
    let f = &sp.f;
    let df = f[1] - f[0];
    let d: Vec<f64> = b
        .iter()
        .map(|(_, pb)| {
            let n = floor(pb, df, 10.0);
            let v = peak(f, pb, &n, fc / rho, 1.0).1 - peak(f, pb, &n, fc * rho, 1.0).1;
            // np.nan_to_num(neginf=-20, posinf=20) then clip
            let v = if v.is_nan() {
                0.0
            } else if v == f64::NEG_INFINITY {
                -20.0
            } else if v == f64::INFINITY {
                20.0
            } else {
                v
            };
            v.clamp(-20.0, 20.0)
        })
        .collect();
    let cb: Vec<Vec<f64>> = d.iter().map(|&v| vec![v.max(0.0), (-v).max(0.0)]).collect();
    let lab = viterbi(&cb, SW_COST); // state 0 = lower beat tracked, 1 = upper
    let lev = [0.0, rho.ln()];
    let dt = c.t[1] - c.t[0];
    let k = ((SW_FINE / dt) as i64).max(1) as usize;
    let nf = r.len() / k;
    let zb: Vec<f64> = (0..nf)
        .map(|j| {
            let v: Vec<f64> = r[j * k..(j + 1) * k].iter().map(|x| x.ln()).collect();
            median(&v)
        })
        .collect();
    let tb: Vec<f64> = b.iter().map(|(t, _)| *t).collect();
    let mut st: Vec<usize> = (0..nf)
        .map(|j| {
            let tj = c.t[j * k];
            let ss = tb.partition_point(|&v| v <= tj) as i64 - 1;
            lab[ss.clamp(0, tb.len() as i64 - 1) as usize]
        })
        .collect();
    let size = ((SW_TREND / SW_FINE) as usize) | 1;
    let mut sig = 0.0;
    for _ in 0..3 {
        let zl: Vec<f64> = (0..nf).map(|j| zb[j] - lev[st[j]]).collect();
        let tr = median_filter(&zl, size, Mode::Nearest)?;
        let e: Vec<f64> = (0..nf).map(|j| zb[j] - tr[j]).collect();
        let ad: Vec<f64> = (0..nf).map(|j| (e[j] - lev[st[j]]).abs()).collect();
        let sg = 1.4826 * median(&ad);
        sig = if sg >= SW_SIG || sg.is_nan() {
            sg
        } else {
            SW_SIG
        };
        let cc: Vec<Vec<f64>> = e
            .iter()
            .map(|&ej| {
                lev.iter()
                    .map(|&l| {
                        let q = (ej - l) / sig;
                        let q2 = q * q;
                        if q2.is_nan() || q2 < 25.0 { q2 } else { 25.0 }
                    })
                    .collect()
            })
            .collect();
        st = viterbi(&cc, SW_FCOST);
    }
    let last = *st.last().unwrap_or(&0);
    let mut sk: Vec<usize> = st.iter().flat_map(|&s| std::iter::repeat_n(s, k)).collect();
    sk.resize(r.len(), last);
    let major = usize::from(sk.iter().filter(|&&s| s == 1).count() as f64 / sk.len() as f64 > 0.5);
    let g: Vec<f64> = sk.iter().map(|&s| (lev[s] - lev[major]).exp()).collect();
    // span edges: flatnonzero(diff(r_[0, sk != major, 0]))
    let mut edges = Vec::new();
    let mut prev = false;
    for (i, &s) in sk.iter().enumerate() {
        let on = s != major;
        if on != prev {
            edges.push(i);
        }
        prev = on;
    }
    if prev {
        edges.push(sk.len());
    }
    let nlast = r.len() - 1;
    let spans: Vec<(f64, f64)> = edges
        .chunks_exact(2)
        .map(|p| (c.t[p[0]], c.t[p[1].min(nlast)]))
        .collect();
    info.sw_spans = spans.len();
    info.sw_s = spans.iter().fold(0.0, |a, (s0, s1)| a + (s1 - s0));
    info.sw_sigma_pct = Some(sig * 100.0);
    info.sw_major = if major == 1 { "hi" } else { "lo" }.into();
    info.sw_list = Some(
        spans
            .iter()
            .take(200)
            .map(|(a, b)| format!("{}-{}", fmt_fixed(*a, 1), fmt_fixed(*b, 1)))
            .collect::<Vec<_>>()
            .join(" "),
    );
    info.spans = spans;
    let mut out = c.clone();
    out.ratio = r.iter().zip(&g).map(|(a, b)| a / b).collect();
    Ok((out, info))
}
