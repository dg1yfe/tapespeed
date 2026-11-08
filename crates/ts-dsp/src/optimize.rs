//! Bounded scalar minimisation with the definition of
//! `scipy.optimize.minimize_scalar(method="bounded")` (`_minimize_scalar_bounded`, SciPy 1.18):
//! Brent's method with golden-section fallback, same steps and termination test.

use crate::{ValueError, verr};

/// Result of [`minimize_scalar_bounded`] (`OptimizeResult` fields used by callers).
#[derive(Debug, Clone, Copy)]
pub struct Bounded {
    pub x: f64,
    pub fun: f64,
    pub nfev: usize,
    /// 0 converged, 1 maxiter reached, 2 NaN encountered.
    pub status: u8,
}

#[inline]
fn sign(x: f64) -> f64 {
    // np.sign (0 → 0, NaN → NaN)
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

/// `minimize_scalar(f, bounds=(x1, x2), method="bounded", options=dict(xatol, maxiter))`.
pub fn minimize_scalar_bounded<F: FnMut(f64) -> f64>(
    mut func: F,
    bounds: (f64, f64),
    xatol: f64,
    maxiter: usize,
) -> Result<Bounded, ValueError> {
    let (x1, x2) = bounds;
    if !(x1.is_finite() && x2.is_finite()) {
        return verr("Optimization bounds must be finite scalars.");
    }
    if x1 > x2 {
        return verr("The lower bound exceeds the upper bound.");
    }
    let maxfun = maxiter;
    let mut flag = 0u8;
    let sqrt_eps = 2.2e-16f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let mut rat = 0.0f64;
    let mut e = 0.0f64;
    let mut x = xf;
    let mut fx = func(x);
    let mut num = 1usize;
    let mut fu = f64::INFINITY;
    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = (p + 0.0) / q;
                x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        x = xf + si * rat.abs().max(tol1);
        fu = func(x);
        num += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxfun {
            flag = 1;
            break;
        }
    }
    if xf.is_nan() || fx.is_nan() || fu.is_nan() {
        flag = 2;
    }
    Ok(Bounded {
        x: xf,
        fun: fx,
        nfev: num,
        status: flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parabola() {
        let r =
            minimize_scalar_bounded(|x| (x - 0.3) * (x - 0.3), (-1.0, 2.0), 1e-12, 500).unwrap();
        assert!((r.x - 0.3).abs() < 1e-9 && r.status == 0);
    }
}
