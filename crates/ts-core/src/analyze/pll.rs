//! Stage 3: 2nd-order PLL with atan2 detector.
//!
//! The loop is strictly sequential and scalar, in a fixed operation order, without FMA
//! contraction: `cos`/`sin`/`atan2` are the platform libm, `hypot` is a correctly-rounded
//! two-coordinate norm (≠ libm `hypot`), see [`hypot_accurate`].
//! Bit-exact against the stored reference vectors (`pll_zp`).
//! Parallelism: the forward and the time-reversed loop of one bandwidth run as two rayon
//! tasks ([`pll_fwd_bwd`]); `analyze()` runs the bandwidths in parallel → 10 independent loops.

use num_complex::Complex64;

use crate::hetdec::cos_sin;

/// PLL defaults (damping, integrator leak time constant [s]).
pub const ZETA: f64 = 0.707;
pub const LEAK_TC: f64 = 0.3;

/// `(hi, lo)` with `hi + lo == x * y` exactly (TwoProduct, FMA variant).
#[inline]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

/// Fast2Sum: `(s, err)` with `s + err == a + b` exactly (requires `|a| >= |b|`).
#[inline]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let z = x - a;
    (x, b - z)
}

/// Correctly-rounded `hypot(x, y)` (two-coordinate vector norm: lossless scaling,
/// double-length squares and sums, one differential correction). Differs from libm `hypot` in the last bit for ~15 % of inputs.
/// Third-party origin and licence: see NOTICE (item 3).
pub fn hypot_accurate(x: f64, y: f64) -> f64 {
    let (x, y) = (x.abs(), y.abs());
    let found_nan = x.is_nan() || y.is_nan();
    // `if (x > max) max = x` from max = 0.0, so NaN never becomes max
    let mut max = 0.0f64;
    if x > max {
        max = x;
    }
    if y > max {
        max = y;
    }
    vector_norm2([x, y], max, found_nan)
}

fn vector_norm2(mut vec: [f64; 2], max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return max;
    }
    let max_e = frexp_exp(max);
    if max_e < -1023 {
        // subnormal max: rescale (recursion with DBL_MIN)
        for v in &mut vec {
            *v /= f64::MIN_POSITIVE;
        }
        return f64::MIN_POSITIVE * vector_norm2(vec, max / f64::MIN_POSITIVE, found_nan);
    }
    let scale = ldexp1(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for &v in &vec {
        let x = v * scale;
        let (hi, lo) = dl_mul(x, x);
        let (s, t) = dl_fast_sum(csum, hi);
        csum = s;
        frac1 += lo;
        frac2 += t;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (hi, lo) = dl_mul(-h, h);
    let (s, t) = dl_fast_sum(csum, hi);
    csum = s;
    frac1 += lo;
    frac2 += t;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// Exponent `e` of C `frexp(x)` (`x = m·2^e`, `0.5 <= m < 1`) for finite `x > 0`.
fn frexp_exp(x: f64) -> i32 {
    let bits = x.to_bits();
    let be = ((bits >> 52) & 0x7ff) as i32;
    if be == 0 {
        let m = bits & ((1u64 << 52) - 1);
        let lz = m.leading_zeros() as i32 - 12; // leading zeros inside the 52-bit field
        -1022 - lz
    } else {
        be - 1022
    }
}

/// `ldexp(1.0, e)`; called with `e = -max_e`, `-1024 <= e <= 1023`.
fn ldexp1(e: i32) -> f64 {
    if e > 1023 {
        f64::INFINITY
    } else if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (52 + (e + 1022)))
    }
}

/// NCO phase per sample [rad] and lock metric (`cos e`, smoothed) of [`pll`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PllOut {
    pub th: Vec<f64>,
    pub lock: Vec<f64>,
}

/// Loop constants of `pll(z, fs, fn, wmax, zeta, leak_tc)`, computed in a fixed order.
#[derive(Debug, Clone, Copy)]
struct Consts {
    kp: f64,
    ki: f64,
    al: f64,
    leak: f64,
    imax: f64,
}

impl Consts {
    fn new(fs: f64, fn_hz: f64, wmax: f64, zeta: f64, leak_tc: f64) -> Self {
        let tau = 2.0 * std::f64::consts::PI;
        let wt = tau * fn_hz / fs;
        Consts {
            kp: 2.0 * zeta * wt,
            ki: wt * wt,
            al: 1.0 / (0.05 * fs),
            leak: 1.0 - 1.0 / (leak_tc * fs),
            imax: tau * wmax / fs,
        }
    }
}

/// The loop over an iterator of `(re, im)` samples.
fn run<I: Iterator<Item = (f64, f64)>>(z: I, n: usize, k: Consts) -> PllOut {
    let mut th = 0.0f64;
    let mut integ = 0.0f64;
    let mut lock = 1.0f64;
    let mut ths = Vec::with_capacity(n);
    let mut locks = Vec::with_capacity(n);
    for (zr, zi) in z {
        let (c, s) = cos_sin(th);
        let re = zr * c + zi * s;
        let im = zi * c - zr * s;
        let e = im.atan2(re);
        ths.push(th);
        lock += k.al * (re / (hypot_accurate(re, im) + 1e-30) - lock);
        locks.push(lock);
        integ += k.ki * e;
        if lock < 0.2 {
            integ *= k.leak; // unlocked: relax towards coarse track (residual 0 Hz)
        }
        // min(max(integ, -imax), imax)
        if -k.imax > integ {
            integ = -k.imax;
        }
        if k.imax < integ {
            integ = k.imax;
        }
        th += integ + k.kp * e;
    }
    PllOut {
        th: ths,
        lock: locks,
    }
}

/// `pll(z, fs, fn, wmax=wmax, zeta=zeta, leak_tc=leak_tc)`.
pub fn pll_with(
    z: &[Complex64],
    fs: f64,
    fn_hz: f64,
    wmax: f64,
    zeta: f64,
    leak_tc: f64,
) -> PllOut {
    let k = Consts::new(fs, fn_hz, wmax, zeta, leak_tc);
    run(z.iter().map(|c| (c.re, c.im)), z.len(), k)
}

/// `pll(z, fs, fn, wmax=wmax)` with the default `zeta`, `leak_tc`.
pub fn pll(z: &[Complex64], fs: f64, fn_hz: f64, wmax: f64) -> PllOut {
    pll_with(z, fs, fn_hz, wmax, ZETA, LEAK_TC)
}

/// `pll(conj(z[::-1]), fs, fn, wmax=wmax)` without materialising the reversed input;
/// output in the loop's own (reversed) time order.
pub fn pll_rev(z: &[Complex64], fs: f64, fn_hz: f64, wmax: f64) -> PllOut {
    let k = Consts::new(fs, fn_hz, wmax, ZETA, LEAK_TC);
    run(z.iter().rev().map(|c| (c.re, -c.im)), z.len(), k)
}

/// Forward loop and raw time-reversed loop of [`crate::analyze::track::track`], run as two
/// parallel rayon tasks.
/// Returns `(fwd, bwd_raw)`; the tracker uses `th_b = -bwd.th[::-1]`, `lk_b = bwd.lock[::-1]`.
pub fn pll_fwd_bwd(zp: &[Complex64], fs: f64, fn_hz: f64, wmax: f64) -> (PllOut, PllOut) {
    rayon::join(|| pll(zp, fs, fn_hz, wmax), || pll_rev(zp, fs, fn_hz, wmax))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypot_basics() {
        assert_eq!(hypot_accurate(3.0, 4.0), 5.0);
        assert_eq!(hypot_accurate(-3.0, 0.0), 3.0);
        assert_eq!(hypot_accurate(0.0, 0.0), 0.0);
        assert!(hypot_accurate(f64::NAN, 1.0).is_nan());
        assert_eq!(hypot_accurate(f64::NAN, f64::INFINITY), f64::INFINITY);
        assert_eq!(hypot_accurate(5e-324, 0.0), 5e-324);
        assert_eq!(hypot_accurate(1e308, 1e308), 1.4142135623730952e308);
    }

    #[test]
    fn frexp() {
        assert_eq!(frexp_exp(1.0), 1);
        assert_eq!(frexp_exp(0.5), 0);
        assert_eq!(frexp_exp(0.75), 0);
        assert_eq!(frexp_exp(f64::MIN_POSITIVE), -1021);
        assert_eq!(frexp_exp(5e-324), -1073);
    }
}
