//! The numbers behind the speed plot (spans, band RMS, detail window, wow / flutter / L1),
//! computed without any drawing.
use crate::PlotError;
use ts_core::curve::{Curve, MetaValue};
use ts_dsp::ndimage::{Mode, uniform_filter1d};
use ts_dsp::spectral::welch;
use ts_dsp::stats::{mean, median, np_sum, percentiles};

/// Everything `plot_speed` computes before drawing.
#[derive(Debug, Clone)]
pub struct SpeedNumbers {
    /// wow: rms of the speed variation in 0.5..6 Hz [%]
    pub wow_pct: f64,
    /// flutter: rms in 6..25 Hz [%]
    pub flutter_pct: f64,
    /// L1 absolute speed offset [%]
    pub l1_pct: f64,
    /// `(s - 1) * 100`
    pub dev: Vec<f64>,
    /// pass-1 curve L1-aligned, `(c1.ratio[:n] * k - 1) * 100`
    pub raw: Vec<f64>,
    /// `uniform_filter1d(dev, 2 s)`
    pub trend: Vec<f64>,
    /// L1 factor `median(s / cm.ratio)`
    pub k: f64,
    /// Welch frequencies / density of `dev - mean(dev)`
    pub f: Vec<f64>,
    pub p: Vec<f64>,
    /// spans (start t, end t) of the invalid samples of `c`
    pub invalid_spans: Vec<(f64, f64)>,
    /// `ref_gaps` header entries
    pub gaps: Vec<(f64, f64)>,
    /// detail window [t0, t0 + w]
    pub detail_t0: f64,
    pub detail_w: f64,
}

/// A header value as a float (bools as 0 / 1, strings parsed); `None` if absent or unparsable.
pub fn meta_f64(c: &Curve, key: &str) -> Option<f64> {
    match c.meta.get(key)? {
        MetaValue::Float(x) => Some(*x),
        MetaValue::Int(i) => Some(*i as f64),
        MetaValue::Bool(b) => Some(*b as i64 as f64),
        MetaValue::Str(s) => s.trim().parse().ok(),
    }
}

/// A header value as text (`MetaValue::to_text`), or `default`.
pub fn meta_str(c: &Curve, key: &str, default: &str) -> String {
    c.meta
        .get(key)
        .map(|v| v.to_text())
        .unwrap_or_else(|| default.to_string())
}

/// `spans(t, mask)`: runs of true as (t[a], t[min(b, len-1)]).
pub fn spans(t: &[f64], mask: &[bool]) -> Vec<(f64, f64)> {
    let n = mask.len();
    let mut out = vec![];
    let mut i = 0;
    while i < n {
        if mask[i] {
            let a = i;
            while i < n && mask[i] {
                i += 1;
            }
            out.push((t[a], t[i.min(t.len() - 1)]));
        } else {
            i += 1;
        }
    }
    out
}

/// `band_rms(f, P, lo, hi)`: sqrt of the trapezoid integral over `lo <= f < hi`, NaN if < 2 bins.
pub fn band_rms(f: &[f64], p: &[f64], lo: f64, hi: f64) -> f64 {
    let (mut fs, mut ps) = (vec![], vec![]);
    for (&fi, &pi) in f.iter().zip(p) {
        if fi >= lo && fi < hi {
            fs.push(fi);
            ps.push(pi);
        }
    }
    if fs.len() < 2 {
        return f64::NAN;
    }
    let terms: Vec<f64> = (1..fs.len())
        .map(|i| (fs[i] - fs[i - 1]) * (ps[i] + ps[i - 1]) / 2.0)
        .collect();
    np_sum(&terms).sqrt()
}

/// `detail_window(t, valid, width)`: start of the `width`-s window with the highest valid share,
/// nearest the file centre.
pub fn detail_window(t: &[f64], valid: &[bool], width: f64) -> f64 {
    let dt = t[1] - t[0];
    let n = ((width / dt).round_ties_even() as i64).max(1) as usize;
    if t.len() <= n {
        return t[0];
    }
    // integer window counts (exact); share = count / n
    let mut pre = vec![0usize; t.len() + 1];
    for i in 0..t.len() {
        pre[i + 1] = pre[i] + valid[i] as usize;
    }
    let share: Vec<f64> = (0..=t.len() - n)
        .map(|i| (pre[i + n] - pre[i]) as f64 / n as f64)
        .collect();
    let mx = share.iter().cloned().fold(f64::MIN, f64::max);
    let mid = (t.len() - n) as f64 / 2.0;
    let mut best = (f64::INFINITY, 0usize);
    for (i, &s) in share.iter().enumerate() {
        if s >= mx - 1e-9 {
            let d = (i as f64 - mid).abs();
            if d < best.0 {
                best = (d, i);
            }
        }
    }
    t[best.1]
}

/// `ref_gaps` header: whitespace-separated `a-b` pairs.
pub fn ref_gaps(c: &Curve) -> Vec<(f64, f64)> {
    let s = meta_str(c, "ref_gaps", "");
    s.split_whitespace()
        .filter(|g| g.contains('-'))
        .filter_map(|g| {
            let (a, b) = g.split_once('-')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .collect()
}

/// The numbers `plot_speed(c1, c, cm, ...)` reports and draws.
pub fn speed_plot_numbers(c1: &Curve, c: &Curve, cm: &Curve) -> Result<SpeedNumbers, PlotError> {
    let (t, s) = (&c.t, &c.ratio);
    if t.len() < 2 || cm.ratio.len() != s.len() || c.valid.len() != s.len() || c1.t.is_empty() {
        return Err(PlotError::Input(
            "curves too short or of different length".into(),
        ));
    }
    let dev: Vec<f64> = s.iter().map(|x| (x - 1.0) * 100.0).collect();
    let v = &c.valid;
    let k = if v.iter().any(|&b| b) {
        let q: Vec<f64> = (0..s.len())
            .filter(|&i| v[i])
            .map(|i| s[i] / cm.ratio[i])
            .collect();
        median(&q)
    } else {
        let q: Vec<f64> = s.iter().zip(&cm.ratio).map(|(a, b)| a / b).collect();
        median(&q)
    };
    let n = c1.t.len().min(t.len());
    let raw: Vec<f64> = c1.ratio[..n]
        .iter()
        .map(|r| (r * k - 1.0) * 100.0)
        .collect();
    let dt = t[1] - t[0];
    let size = ((2.0 / dt) as usize).max(1);
    let trend = uniform_filter1d(&dev, size, Mode::Nearest)?;
    let l1 = (meta_f64(c, "abs_l1").unwrap_or_else(|| mean(s)) - 1.0) * 100.0;
    let dm = mean(&dev);
    let x: Vec<f64> = dev.iter().map(|d| d - dm).collect();
    let nper = dev.len().min((40.0 / dt) as usize);
    let (f, p) = welch(&x, 1.0 / dt, nper)?;
    let inv: Vec<bool> = v.iter().map(|b| !b).collect();
    let w = 5.0f64.min(t[t.len() - 1]);
    Ok(SpeedNumbers {
        wow_pct: band_rms(&f, &p, 0.5, 6.0),
        flutter_pct: band_rms(&f, &p, 6.0, 25.0),
        l1_pct: l1,
        invalid_spans: spans(t, &inv),
        gaps: ref_gaps(c),
        detail_t0: detail_window(t, v, w),
        detail_w: w,
        dev,
        raw,
        trend,
        k,
        f,
        p,
    })
}

/// `np.percentile(x, [lo, hi])` helper used for the y limits.
pub(crate) fn pct2(x: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    match percentiles(x, &[lo, hi]) {
        Ok(v) => (v[0], v[1]),
        Err(_) => (-1.0, 1.0),
    }
}
