//! `scipy.interpolate.CubicSpline` (bc "not-a-knot"; n = 2 linear, n = 3 parabola) and
//! `PPoly` evaluation / antiderivative (extrapolate=True).
//!
//! Numerics follow SciPy: tridiagonal solve = LAPACK `dgtsv` (partial pivoting), n = 3 dense
//! `gesv`; Hermite coefficients as `CubicHermiteSpline`; evaluation as `_ppoly.evaluate_poly1`
//! (power sum, `z *= s`); antiderivative constants accumulated sequentially (`fix_continuity`).

use std::sync::Arc;

use crate::{ValueError, verr as err};

/// Piecewise polynomial in the local power basis (SciPy `PPoly`, real, extrapolate=True).
#[derive(Debug, Clone, PartialEq)]
pub struct PPoly {
    /// breakpoints, shared with derived polynomials (antiderivative)
    x: Arc<[f64]>,
    /// interval-major: `c[i * k + j]` == SciPy `c[j, i]` (j = 0 highest power)
    c: Vec<f64>,
    k: usize,
}

impl PPoly {
    /// Build from SciPy-layout coefficients `c[j][i]` (k rows, m = len(x)-1 columns).
    pub fn from_scipy(c: &[Vec<f64>], x: &[f64]) -> Self {
        let k = c.len();
        let m = x.len() - 1;
        let mut cc = vec![0.0; k * m];
        for (j, row) in c.iter().enumerate() {
            for i in 0..m {
                cc[i * k + j] = row[i];
            }
        }
        Self {
            x: x.into(),
            c: cc,
            k,
        }
    }

    /// Breakpoints.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Polynomial order k (degree + 1).
    pub fn order(&self) -> usize {
        self.k
    }

    /// SciPy `c[j, i]`.
    pub fn coef(&self, j: usize, i: usize) -> f64 {
        self.c[i * self.k + j]
    }

    /// Coefficients in SciPy layout (k rows).
    pub fn coeffs(&self) -> Vec<Vec<f64>> {
        let m = self.x.len() - 1;
        (0..self.k)
            .map(|j| (0..m).map(|i| self.coef(j, i)).collect())
            .collect()
    }

    #[inline]
    fn poly1(&self, s: f64, i: usize) -> f64 {
        let c = &self.c[i * self.k..(i + 1) * self.k];
        let mut res = 0.0;
        let mut z = 1.0;
        for kp in 0..self.k {
            res += c[self.k - kp - 1] * z;
            if kp < self.k - 1 {
                z *= s;
            }
        }
        res
    }

    /// `find_interval` (SciPy `_ppoly`): i with x[i] <= xv < x[i+1], clamped; None for NaN.
    #[inline]
    fn interval(&self, xv: f64, hint: usize) -> Option<usize> {
        let x = &self.x;
        let nx = x.len();
        let (a, b) = (x[0], x[nx - 1]);
        if !(a <= xv && xv <= b) {
            if xv < a {
                return Some(0);
            } else if xv > b {
                return Some(nx - 2);
            }
            return None;
        }
        if xv == b {
            return Some(nx - 2);
        }
        if hint < nx - 1 && x[hint] <= xv && xv < x[hint + 1] {
            return Some(hint);
        }
        // largest i with x[i] <= xv
        let i = x.partition_point(|&t| t <= xv) - 1;
        Some(i.min(nx - 2))
    }

    /// Interval index SciPy would use for `xv` (None for NaN); `hint` only speeds up the search.
    #[inline]
    pub fn locate(&self, xv: f64, hint: usize) -> Option<usize> {
        self.interval(xv, hint)
    }

    /// Evaluate piece `i` (from [`locate`](Self::locate)) at `xv`.
    #[inline]
    pub fn eval_piece(&self, i: usize, xv: f64) -> f64 {
        self.poly1(xv - self.x[i], i)
    }

    /// Evaluate at `xv` (`PPoly.__call__`, nu = 0).
    pub fn eval(&self, xv: f64) -> f64 {
        let mut h = 0;
        self.eval_hint(xv, &mut h)
    }

    /// Evaluate reusing / updating an interval hint (same result as [`eval`](Self::eval)).
    #[inline]
    pub fn eval_hint(&self, xv: f64, hint: &mut usize) -> f64 {
        match self.interval(xv, *hint) {
            Some(i) => {
                *hint = i;
                self.poly1(xv - self.x[i], i)
            }
            None => f64::NAN,
        }
    }

    /// `PPoly.antiderivative(1)`.
    pub fn antiderivative(&self) -> PPoly {
        let k = self.k;
        let k2 = k + 1;
        let m = self.x.len() - 1;
        let mut c = vec![0.0; k2 * m];
        for i in 0..m {
            for j in 0..k {
                // poch(k - j, 1) = k - j
                c[i * k2 + j] = self.c[i * k + j] / (k - j) as f64;
            }
        }
        let mut out = PPoly {
            x: self.x.clone(),
            c,
            k: k2,
        };
        for ip in 1..m {
            let s = out.x[ip] - out.x[ip - 1];
            let v = out.poly1(s, ip - 1);
            out.c[ip * k2 + k2 - 1] = v;
        }
        out
    }
}

/// LAPACK `dgtsv` (one right-hand side): dl (n-1), d (n), du (n-1), b (n) -> solution in b.
fn dgtsv(dl: &mut [f64], d: &mut [f64], du: &mut [f64], b: &mut [f64]) -> Result<(), ValueError> {
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    let singular = || err("singular matrix");
    if n > 2 {
        for i in 0..n - 2 {
            if d[i].abs() >= dl[i].abs() {
                if d[i] == 0.0 {
                    return singular();
                }
                let fact = dl[i] / d[i];
                d[i + 1] -= fact * du[i];
                b[i + 1] -= fact * b[i];
                dl[i] = 0.0;
            } else {
                let fact = d[i] / dl[i];
                d[i] = dl[i];
                let temp = d[i + 1];
                d[i + 1] = du[i] - fact * temp;
                dl[i] = du[i + 1];
                du[i + 1] = -fact * dl[i];
                du[i] = temp;
                let temp = b[i];
                b[i] = b[i + 1];
                b[i + 1] = temp - fact * b[i + 1];
            }
        }
    }
    if n > 1 {
        let i = n - 2;
        if d[i].abs() >= dl[i].abs() {
            if d[i] == 0.0 {
                return singular();
            }
            let fact = dl[i] / d[i];
            d[i + 1] -= fact * du[i];
            b[i + 1] -= fact * b[i];
        } else {
            let fact = d[i] / dl[i];
            d[i] = dl[i];
            let temp = d[i + 1];
            d[i + 1] = du[i] - fact * temp;
            du[i] = temp;
            let temp = b[i];
            b[i] = b[i + 1];
            b[i + 1] = temp - fact * b[i + 1];
        }
    }
    if d[n - 1] == 0.0 {
        return singular();
    }
    b[n - 1] /= d[n - 1];
    if n > 1 {
        b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / d[n - 2];
    }
    if n > 2 {
        for i in (0..n - 2).rev() {
            b[i] = (b[i] - du[i] * b[i + 1] - dl[i] * b[i + 2]) / d[i];
        }
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
/// Dense LU with partial pivoting (reference LAPACK `dgetrf2` + `dgetrs` order), small n.
fn gesv(mut a: Vec<[f64; 3]>, mut b: [f64; 3]) -> Result<[f64; 3], ValueError> {
    let n = 3;
    for col in 0..n {
        let mut p = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[p][col].abs() {
                p = r;
            }
        }
        if a[p][col] == 0.0 {
            return err("singular matrix");
        }
        a.swap(col, p);
        b.swap(col, p);
        let r = 1.0 / a[col][col];
        for row in col + 1..n {
            a[row][col] *= r;
        }
        for row in col + 1..n {
            for cc in col + 1..n {
                a[row][cc] -= a[row][col] * a[col][cc];
            }
        }
    }
    for k in 0..n {
        if b[k] != 0.0 {
            for i in k + 1..n {
                b[i] -= b[k] * a[i][k];
            }
        }
    }
    for k in (0..n).rev() {
        if b[k] != 0.0 {
            b[k] /= a[k][k];
            for i in 0..k {
                b[i] -= b[k] * a[i][k];
            }
        }
    }
    Ok(b)
}

/// `scipy.interpolate.CubicSpline(x, y)` (bc_type "not-a-knot", extrapolate=True) as `PPoly`.
pub fn cubic_spline(x: &[f64], y: &[f64]) -> Result<PPoly, ValueError> {
    let n = x.len();
    if n < 2 {
        return err("`x` must contain at least 2 elements.");
    }
    if y.len() != n {
        return err("The length of `y` along `axis`=0 doesn't match the length of `x`");
    }
    if x.iter().any(|v| !v.is_finite()) {
        return err("`x` must contain only finite values.");
    }
    if y.iter().any(|v| !v.is_finite()) {
        return err("`y` must contain only finite values.");
    }
    let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    if dx.iter().any(|&d| d <= 0.0) {
        return err("`x` must be strictly increasing sequence.");
    }
    let slope: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / dx[i]).collect();
    let s: Vec<f64> = if n == 2 {
        let mut dl = vec![0.0];
        let mut d = vec![1.0, 1.0];
        let mut du = vec![0.0];
        let mut b = vec![slope[0], slope[0]];
        dgtsv(&mut dl, &mut d, &mut du, &mut b)?;
        b
    } else if n == 3 {
        let a = vec![
            [1.0, 1.0, 0.0],
            [dx[1], 2.0 * (dx[0] + dx[1]), dx[0]],
            [0.0, 1.0, 1.0],
        ];
        let b = [
            2.0 * slope[0],
            3.0 * (dx[0] * slope[1] + dx[1] * slope[0]),
            2.0 * slope[1],
        ];
        gesv(a, b)?.to_vec()
    } else {
        let mut d = vec![0.0; n];
        let mut du = vec![0.0; n - 1];
        let mut dl = vec![0.0; n - 1];
        let mut b = vec![0.0; n];
        for i in 1..n - 1 {
            d[i] = 2.0 * (dx[i - 1] + dx[i]);
            b[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
        }
        du[1..n - 1].copy_from_slice(&dx[..n - 2]); // A[0, 2:] = dx[:-1]
        dl[..n - 2].copy_from_slice(&dx[1..n - 1]); // A[-1, :-2] = dx[1:]
        // not-a-knot start
        d[0] = dx[1];
        let dd = x[2] - x[0];
        du[0] = dd;
        b[0] = ((dx[0] + 2.0 * dd) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / dd;
        // not-a-knot end
        d[n - 1] = dx[n - 3];
        let dd = x[n - 1] - x[n - 3];
        dl[n - 2] = dd;
        b[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3]
            + (2.0 * dd + dx[n - 2]) * dx[n - 3] * slope[n - 2])
            / dd;
        dgtsv(&mut dl, &mut d, &mut du, &mut b)?;
        drop((dl, d, du));
        b
    };
    // CubicHermiteSpline coefficients
    let k = 4;
    let mut c = vec![0.0; k * (n - 1)];
    for i in 0..n - 1 {
        let t = (s[i] + s[i + 1] - 2.0 * slope[i]) / dx[i];
        c[i * k] = t / dx[i];
        c[i * k + 1] = (slope[i] - s[i]) / dx[i] - t;
        c[i * k + 2] = s[i];
        c[i * k + 3] = y[i];
    }
    Ok(PPoly { x: x.into(), c, k })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_cubic_and_integrates() {
        let x: Vec<f64> = (0..20)
            .map(|i| i as f64 * 0.3 + (i * i) as f64 * 0.01)
            .collect();
        let f = |t: f64| 1.0 + 0.5 * t - 0.2 * t * t + 0.03 * t * t * t;
        let y: Vec<f64> = x.iter().map(|&t| f(t)).collect();
        let sp = cubic_spline(&x, &y).unwrap();
        let ad = sp.antiderivative();
        for &t in &[0.0, 0.17, 2.5, 5.0, 7.9, -0.5, 9.0] {
            assert!((sp.eval(t) - f(t)).abs() < 1e-9, "t={t}");
            let big = |t: f64| t + 0.25 * t * t - 0.2 / 3.0 * t.powi(3) + 0.0075 * t.powi(4);
            assert!((ad.eval(t) - big(t)).abs() < 1e-9);
        }
        // n = 2 linear, n = 3 parabola
        let l = cubic_spline(&[0.0, 2.0], &[1.0, 5.0]).unwrap();
        assert_eq!(l.eval(1.0), 3.0);
        let p = cubic_spline(&[0.0, 1.0, 3.0], &[0.0, 1.0, 9.0]).unwrap();
        assert!((p.eval(2.0) - 4.0).abs() < 1e-14);
        assert!(cubic_spline(&[0.0, 0.0], &[1.0, 2.0]).is_err());
    }
}
