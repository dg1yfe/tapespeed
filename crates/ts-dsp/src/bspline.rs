//! `scipy.interpolate.BSpline` pieces used by the model drift basis:
//! `BSpline.design_matrix(x, t, k)` and `BSpline(t, c, k, extrapolate=False)(x)`.
//! Same definitions and operation order as SciPy's `_find_interval` and `_deBoor_D`; the right
//! end `t[n]` belongs to the last interval. `h += w * d` and `out += c * B` are fused
//! multiply-adds (`mul_add`), as SciPy's C++ evaluates them when compiled with FP contraction
//! (clang default on arm64); bit-exact against the stored reference vectors on arm64.

use crate::{ValueError, verr};

/// scipy `_find_interval(t, k, x, prev_l, extrapolate)`: `l` with `t[l] <= x < t[l+1]`,
/// `k <= l < n` (`n = len(t) - k - 1`); `x == t[n]` → `n - 1`. `None` for NaN or (without
/// extrapolation) `x` outside `[t[k], t[n]]`.
pub fn find_interval(
    t: &[f64],
    k: usize,
    x: f64,
    prev_l: usize,
    extrapolate: bool,
) -> Option<usize> {
    let n = t.len() - k - 1;
    let (tb, te) = (t[k], t[n]);
    if x.is_nan() || ((x < tb || x > te) && !extrapolate) {
        return None;
    }
    let mut l = if k < prev_l && prev_l < n { prev_l } else { k };
    while x < t[l] && l != k {
        l -= 1;
    }
    l += 1;
    while x >= t[l] && l != n {
        l += 1;
    }
    Some(l - 1)
}

/// scipy `_deBoor_D(t, x, k, ell, m=0, result)`: the `k + 1` non-zero B-spline values
/// `B_{ell-k..=ell, k}(x)` into `out[..=k]` (`out` needs `2k + 2` slots, the upper half is
/// scratch).
pub fn de_boor(t: &[f64], x: f64, k: usize, ell: usize, out: &mut [f64]) {
    let (h, hh) = out.split_at_mut(k + 1);
    h[0] = 1.0;
    for j in 1..=k {
        hh[..j].copy_from_slice(&h[..j]);
        h[0] = 0.0;
        for n in 1..=j {
            let ind = ell + n;
            let xb = t[ind];
            let xa = t[ind - j];
            if xb == xa {
                h[n] = 0.0;
                continue;
            }
            let w = hh[n - 1] / (xb - xa);
            h[n - 1] = w.mul_add(xb - x, h[n - 1]);
            h[n] = w * (x - xa);
        }
    }
}

/// `BSpline.design_matrix(x, t, k)` (extrapolate=False) as CSR rows with `k + 1` entries:
/// returns `(offsets, data)`, row `i` has columns `offsets[i] ..= offsets[i] + k` with values
/// `data[i*(k+1) ..][..k+1]`. Error if an `x` is outside the base interval (scipy raises).
pub fn design_matrix(x: &[f64], t: &[f64], k: usize) -> Result<(Vec<usize>, Vec<f64>), ValueError> {
    if t.len() < 2 * k + 2 {
        return verr("Need at least 2*k + 2 knots");
    }
    let mut off = Vec::with_capacity(x.len());
    let mut data = Vec::with_capacity(x.len() * (k + 1));
    let mut work = vec![0.0; 2 * k + 2];
    let mut ind = k;
    for &xv in x {
        let Some(l) = find_interval(t, k, xv, ind, false) else {
            return verr(format!("Out of bounds: x = {xv}"));
        };
        ind = l;
        de_boor(t, xv, k, l, &mut work);
        off.push(l - k);
        data.extend_from_slice(&work[..=k]);
    }
    Ok((off, data))
}

/// `BSpline(t, c, k, extrapolate=False)(x)`: NaN outside `[t[k], t[n]]`;
/// `out = Σ_a c[l - k + a] * B_a` in `a` order.
pub fn eval(t: &[f64], c: &[f64], k: usize, x: &[f64]) -> Vec<f64> {
    let mut work = vec![0.0; 2 * k + 2];
    let mut ind = k;
    x.iter()
        .map(|&xv| match find_interval(t, k, xv, ind, false) {
            None => f64::NAN,
            Some(l) => {
                ind = l;
                de_boor(t, xv, k, l, &mut work);
                let mut s = 0.0;
                for (a, &w) in work[..=k].iter().enumerate() {
                    s = c[l + a - k].mul_add(w, s);
                }
                s
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_of_unity_and_right_end() {
        let t: Vec<f64> = (-3..8).map(|i| i as f64).collect(); // base [0, 4]
        let x = [0.0, 0.3, 1.0, 2.5, 4.0];
        let (off, d) = design_matrix(&x, &t, 3).unwrap();
        for i in 0..x.len() {
            let s: f64 = d[4 * i..4 * i + 4].iter().sum();
            assert!((s - 1.0).abs() < 1e-14);
        }
        assert_eq!(off[4], 3); // x == t[n] → last interval
        assert!(eval(&t, &[1.0; 7], 3, &[4.1])[0].is_nan());
        assert!(design_matrix(&[-0.1], &t, 3).is_err());
    }
}
