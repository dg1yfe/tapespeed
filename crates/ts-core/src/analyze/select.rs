//! Stage 6b: loop-bandwidth selection.
//!
//! Tie-breaking is fixed: `max(ok, key=pb)` returns the first maximal pb (bw 45
//! and 80 both have pb = fmax = 25), `np.argmin` / `np.argmax` first occurrence with the
//! first NaN winning (`ts_dsp::stats`), scalar `min`/`max` with first-argument NaN semantics
//! (`max2` / `min2`).

use ts_dsp::ValueError;
use ts_dsp::spectral::welch;
use ts_dsp::stats::{argmax, argmin, mean, median, np_sum};

use super::gaps::{DT, max2, min2};

/// One bw candidate as [`select`] sees it (track output + grid outputs).
#[derive(Debug, Clone, Copy)]
pub struct SelIn<'a> {
    /// `c["vg"]` valid on the output grid
    pub vg: &'a [bool],
    /// `c["ratio"]` gap-filled ratio on the output grid
    pub ratio: &'a [f64],
    /// `c["pb"]` curve passband [Hz]
    pub pb: f64,
    /// `c["cn0"]` (fs_p grid)
    pub cn0: &'a [f64],
    /// `c["valid"]` (fs_p grid)
    pub valid: &'a [bool],
}

/// [`select`] return value `(k, vf, err)`; `err` is `sqrt(err)` or `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub k: usize,
    pub valid_frac: Vec<f64>,
    pub pred_err: Option<Vec<f64>>,
}

/// `select(cands, ref_hz, tol=0.03)`.
pub fn select(cands: &[SelIn], ref_hz: f64, tol: f64) -> Result<Selection, ValueError> {
    if cands.is_empty() {
        return Err(ValueError("_select: no candidates".into()));
    }
    // np.mean(bool) = count / n (exact integer sum)
    let vf: Vec<f64> = cands
        .iter()
        .map(|c| c.vg.iter().filter(|&&b| b).count() as f64 / c.vg.len() as f64)
        .collect();
    // vf.max(): NaN propagates (empty vg)
    let vmax = vf.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let tol = min2(tol, 0.25 * vmax);
    let ok: Vec<usize> = (0..cands.len())
        .filter(|&i| vf[i] >= vmax - tol && vf[i] > 0.0)
        .collect();
    if ok.len() <= 1 {
        let k = ok
            .first()
            .copied()
            .unwrap_or_else(|| argmax(&vf).expect("non-empty"));
        return Ok(Selection {
            k,
            valid_frac: vf,
            pred_err: None,
        });
    }
    // max(ok, key=pb): first maximum
    let mut j = ok[0];
    for &i in &ok[1..] {
        if cands[i].pb > cands[j].pb {
            j = i;
        }
    }
    let cj = &cands[j];
    let cv: Vec<f64> = cj
        .cn0
        .iter()
        .zip(cj.valid)
        .filter(|(_, v)| **v)
        .map(|(c, _)| *c)
        .collect();
    let cn0 = median(&cv);
    let m = mean(cj.ratio);
    let x: Vec<f64> = cj.ratio.iter().map(|r| r - m).collect();
    let (fr, p) = welch(&x, 1.0 / DT, cj.ratio.len().min(4096))?;
    let pn: Vec<f64> = fr
        .iter()
        .map(|f| f * f / (cn0 * (ref_hz * ref_hz)))
        .collect();
    let df = fr[1] - fr[0];
    let hi = 1.3 * cj.pb;
    let mut err = Vec::with_capacity(cands.len());
    for (i, c) in cands.iter().enumerate() {
        if !ok.contains(&i) {
            err.push(f64::INFINITY);
            continue;
        }
        let pb = c.pb;
        let sel: Vec<f64> = (0..fr.len())
            .filter(|&q| fr[q] > pb && fr[q] <= hi)
            .map(|q| p[q] - pn[q])
            .collect();
        let miss = max2(np_sum(&sel) * df, 0.0);
        let lo = 1.27 * pb;
        let nsel: Vec<f64> = (0..fr.len())
            .filter(|&q| fr[q] <= lo)
            .map(|q| pn[q])
            .collect();
        err.push(miss + np_sum(&nsel) * df);
    }
    let k = argmin(&err).expect("non-empty");
    Ok(Selection {
        k,
        valid_frac: vf,
        pred_err: Some(err.iter().map(|e| e.sqrt()).collect()),
    })
}
