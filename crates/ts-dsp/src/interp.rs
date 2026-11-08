//! `scipy.interpolate.PchipInterpolator`: Fritsch–Carlson derivatives with the
//! weighted harmonic mean, scipy `_edge_case` one-sided 3-point ends, two points → linear.
//! Evaluation follows scipy's `PPoly` (local power basis, `extrapolate=True`, interval
//! closed on the right at the last knot, powers accumulated as in `evaluate_poly1`).

use crate::{ValueError, verr};

/// Piecewise cubic Hermite interpolant (PCHIP) in scipy's PPoly form.
#[derive(Debug, Clone)]
pub struct Pchip {
    x: Vec<f64>,
    /// `c[k][i]`, k = 0..4 (highest power first), i = interval.
    c: [Vec<f64>; 4],
}

#[inline]
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else if v == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

/// `PchipInterpolator._edge_case(h0, h1, m0, m1)`.
fn edge_case(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    // numpy sign comparisons: NaN != NaN is true
    let mask = sign(d) != sign(m0);
    let mask2 = (sign(m0) != sign(m1)) && (d.abs() > 3.0 * m0.abs());
    if mask {
        0.0
    } else if mask2 {
        3.0 * m0
    } else {
        d
    }
}

impl Pchip {
    /// `PchipInterpolator(x, y)`; `x` strictly increasing, at least 2 points.
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, ValueError> {
        let n = x.len();
        if n != y.len() {
            return verr("x and y arrays must be equal in length along interpolation axis.");
        }
        if n < 2 {
            return verr("at least 2 breakpoints are needed.");
        }
        if x.iter().any(|v| !v.is_finite()) {
            return verr("`x` must contain only finite values.");
        }
        if x.windows(2).any(|w| w[1] - w[0] <= 0.0) {
            return verr("`x` must be strictly increasing sequence.");
        }
        let hk: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let mk: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / hk[i]).collect();
        let mut dk = vec![0.0; n];
        if n == 2 {
            dk[0] = mk[0];
            dk[1] = mk[0];
        } else {
            for i in 1..n - 1 {
                let (m0, m1) = (mk[i - 1], mk[i]);
                let cond = sign(m1) != sign(m0) || m1 == 0.0 || m0 == 0.0;
                if !cond {
                    let w1 = 2.0 * hk[i] + hk[i - 1];
                    let w2 = hk[i] + 2.0 * hk[i - 1];
                    let whmean = (w1 / m0 + w2 / m1) / (w1 + w2);
                    dk[i] = 1.0 / whmean;
                }
            }
            dk[0] = edge_case(hk[0], hk[1], mk[0], mk[1]);
            dk[n - 1] = edge_case(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3]);
        }
        // CubicHermiteSpline -> PPoly coefficients
        let m = n - 1;
        let mut c = [vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m]];
        for i in 0..m {
            let dx = hk[i];
            let slope = (y[i + 1] - y[i]) / dx;
            let t = (dk[i] + dk[i + 1] - 2.0 * slope) / dx;
            c[0][i] = t / dx;
            c[1][i] = (slope - dk[i]) / dx - t;
            c[2][i] = dk[i];
            c[3][i] = y[i];
        }
        Ok(Self { x: x.to_vec(), c })
    }

    /// Interval index as scipy `find_interval_ascending` with extrapolation; `None` for NaN.
    #[inline]
    fn interval(&self, xv: f64) -> Option<usize> {
        let x = &self.x;
        let nx = x.len();
        let (a, b) = (x[0], x[nx - 1]);
        if xv.is_nan() {
            None
        } else if xv < a {
            Some(0)
        } else if xv >= b {
            Some(nx - 2)
        } else {
            Some(x.partition_point(|&v| v <= xv) - 1)
        }
    }

    /// Evaluate at one point (`PPoly.__call__`, nu = 0, extrapolate).
    pub fn eval_one(&self, xv: f64) -> f64 {
        let Some(i) = self.interval(xv) else {
            return f64::NAN;
        };
        let s = xv - self.x[i];
        // evaluate_poly1: res += c[k-kp] * z * 1.0, z *= s
        let mut res = 0.0;
        let mut z = 1.0;
        for kp in 0..4 {
            res += self.c[3 - kp][i] * z * 1.0;
            if kp < 3 {
                z *= s;
            }
        }
        res
    }

    /// Evaluate at many points.
    pub fn eval(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&v| self.eval_one(v)).collect()
    }

    /// PPoly coefficients `[c0, c1, c2, c3]` (highest power first), one entry per interval.
    pub fn coeffs(&self) -> &[Vec<f64>; 4] {
        &self.c
    }
}
