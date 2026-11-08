//! `scipy.signal.find_peaks(x, height=..., distance=...)`.
//!
//! - Local maxima as scipy `_local_maxima_1d`: strict rise, flat plateau, strict fall;
//!   the reported index is the plateau midpoint `(left + right) // 2`; the first and last
//!   samples are never peaks; NaN compares false and therefore never forms a peak.
//! - `height` keeps peaks with `hmin <= x[peak]`.
//! - `distance` as `_select_by_peak_distance`: peaks visited from highest priority
//!   (`x[peak]`) to lowest via `np.argsort` (numpy introsort, unstable), each surviving peak
//!   removes neighbours closer than `ceil(distance)`. Tie order among equal heights follows
//!   [`crate::stats::argsort`], which follows NumPy's scalar `aquicksort` (the path NumPy takes
//!   on aarch64), so ties resolve as there (NumPy on x86 with AVX-512 uses another argsort).

use crate::stats::argsort;
use crate::{ValueError, verr};

/// Local maxima (plateau midpoints) as scipy `_local_maxima_1d`.
pub fn local_maxima_1d(x: &[f64]) -> Vec<usize> {
    let mut out = Vec::new();
    if x.len() < 3 {
        return out;
    }
    let i_max = x.len() - 1;
    let mut i = 1;
    while i < i_max {
        if x[i - 1] < x[i] {
            let mut ahead = i + 1;
            while ahead < i_max && x[ahead] == x[i] {
                ahead += 1;
            }
            if x[ahead] < x[i] {
                out.push((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        i += 1;
    }
    out
}

/// `scipy.signal._peak_finding._select_by_peak_distance` → keep mask.
fn select_by_peak_distance(peaks: &[usize], priority: &[f64], distance: f64) -> Vec<bool> {
    let n = peaks.len();
    let d = distance.ceil();
    let mut keep = vec![true; n];
    let order = argsort(priority);
    for &j in order.iter().rev() {
        if !keep[j] {
            continue;
        }
        let mut k = j;
        while k > 0 && ((peaks[j] - peaks[k - 1]) as f64) < d {
            keep[k - 1] = false;
            k -= 1;
        }
        let mut k = j + 1;
        while k < n && ((peaks[k] - peaks[j]) as f64) < d {
            keep[k] = false;
            k += 1;
        }
    }
    keep
}

/// `find_peaks(x, height=height, distance=distance)[0]` (`None` = argument not given).
pub fn find_peaks(
    x: &[f64],
    height: Option<f64>,
    distance: Option<f64>,
) -> Result<Vec<usize>, ValueError> {
    if let Some(d) = distance
        && d < 1.0
    {
        return verr("`distance` must be greater or equal to 1");
    }
    let mut peaks = local_maxima_1d(x);
    if let Some(h) = height {
        peaks.retain(|&p| h <= x[p]);
    }
    if let Some(d) = distance {
        let pr: Vec<f64> = peaks.iter().map(|&p| x[p]).collect();
        let keep = select_by_peak_distance(&peaks, &pr, d);
        peaks = peaks
            .into_iter()
            .zip(keep)
            .filter_map(|(p, k)| k.then_some(p))
            .collect();
    }
    Ok(peaks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateau_and_edges() {
        let x = [3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 5.0];
        assert_eq!(find_peaks(&x, None, None).unwrap(), vec![3]);
        let y = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(find_peaks(&y, None, None).unwrap(), vec![1]);
        assert!(find_peaks(&[1.0, 2.0], None, None).unwrap().is_empty());
    }
}
