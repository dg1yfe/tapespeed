//! Stages 3-5 for one loop bandwidth ([`track`] and its helpers [`integ`], [`cdiff_freq`],
//! [`fir0`], [`runmed`], [`bridge`]).
//!
//! Element-wise numpy arithmetic is reproduced to the bit where numpy's formula is known:
//! complex multiply `re = fma(ar,br,-(ai·bi))`, `im = fma(ar,bi,ai·br)` ([`np_cmul`]), complex
//! `abs` ([`np_cabs`]), `exp(±1j·φ)` = separate libm `cos`/`sin`, sequential `cumsum`, numpy
//! pairwise `sum`. FFT convolutions (`oaconvolve`), `sosfiltfilt` and `uniform_filter1d`
//! differ from SciPy in summation order only (≈1e-16 relative).
//!
//! Known deviation (degenerate candidates only): when the good mask becomes empty inside the
//! refinement loop (no reference line at all, e.g. the `reel` test case at bw 6: 16 good
//! samples of 11025), [`bridge`] returns `f` unchanged and the next pass integrates
//! `cdiff_freq` of a phasor that is pure FFT rounding noise (~1e-17, the FIR of an all-zero masked span).
//! `dip`/`intf` then depend on the FFT's rounding and are not reproducible; `valid`/`soft`
//! stay identical (all false). Regression tests exempt dip/intf where no sample is valid.
//!
//! Parallelism (deterministic, no shared reductions): forward/backward PLL, the two
//! [`fir0`] calls of each refinement pass, and the two Butterworth filters of the noise
//! estimate run as `rayon::join` pairs.

use num_complex::Complex64;
use ts_dsp::ValueError;
use ts_dsp::conv::{ConvMode, oaconvolve, oaconvolve_c, resample_poly_down_c};
use ts_dsp::iir::{PadType, butter_lowpass_sos, sosfiltfilt};
use ts_dsp::ndimage::{
    Mode, binary_dilation, binary_erosion, binary_opening, median_filter, uniform_filter1d,
};
use ts_dsp::stats::{interp, interp_one_by, median};

use super::coarse::np_cabs;
use super::pll::{self, PllOut};
use crate::hetdec::{cos_sin, lpf};

type Res<T> = Result<T, ValueError>;

/// Output of `track(z, fs_d, fcd, ref_hz, bw, fmax)`: one field per output key, in key
/// order. All arrays live on the fs_p grid and have length `tp.len()`. `ratio`/`cg`/`vg` are
/// derived later in stage 6 (not stored here).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackOut {
    /// time of each fs_p sample [s] (`arange(len(zp)) / fs_p`)
    pub tp: Vec<f64>,
    /// coarse track on the fs_p grid (`fcd[::D2][:len(zp)]`) [Hz offset]
    pub fc: Vec<f64>,
    /// refined residual frequency (relative to `fc`) [Hz]
    pub f: Vec<f64>,
    /// forward-loop frequency [Hz]
    pub ff: Vec<f64>,
    /// backward-loop frequency [Hz]
    pub fb: Vec<f64>,
    /// forward-loop lock metric
    pub lk_f: Vec<f64>,
    /// backward-loop lock metric
    pub lk_b: Vec<f64>,
    pub soft: Vec<bool>,
    pub agree: Vec<f64>,
    pub snr_db: Vec<f64>,
    pub conf: Vec<f64>,
    pub valid: Vec<bool>,
    pub fs_p: f64,
    pub bw: f64,
    pub pb: f64,
    pub slip: Vec<bool>,
    /// output key `dip`
    pub dip: Vec<bool>,
    pub intf: Vec<bool>,
    /// `S / N0`
    pub cn0: Vec<f64>,
}

/// `analyze.BW_AUTO`: candidate loop bandwidths for `bw="auto"`.
pub const BW_AUTO: [f64; 5] = [6.0, 12.0, 25.0, 45.0, 80.0];

// ---------------------------------------------------------------- numpy element-wise --

/// numpy complex multiply (contiguous SIMD loop, FMA): bit-identical to `a * b` on arrays.
#[inline]
pub fn np_cmul(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(
        a.re.mul_add(b.re, -(a.im * b.im)),
        a.re.mul_add(b.im, a.im * b.re),
    )
}

/// `exp(-1j * phi)` as numpy evaluates it: `(cos(-phi), sin(-phi))`.
#[inline]
fn exp_mj(phi: f64) -> Complex64 {
    let (c, s) = cos_sin(-phi);
    Complex64::new(c, s)
}

/// `exp(1j * th)`.
#[inline]
fn exp_j(th: f64) -> Complex64 {
    let (c, s) = cos_sin(th);
    Complex64::new(c, s)
}

/// `z * np.exp(-1j * phi)` element-wise.
pub fn derot(z: &[Complex64], phi: &[f64]) -> Vec<Complex64> {
    z.iter()
        .zip(phi)
        .map(|(&a, &p)| np_cmul(a, exp_mj(p)))
        .collect()
}

/// numpy pairwise summation of a complex array (`CDOUBLE_pairwise_sum`, blocks of 128
/// doubles, 4 complex accumulators), added to the reduction's initial `+0`.
pub fn np_csum(a: &[Complex64]) -> Complex64 {
    fn pw(a: &[Complex64]) -> (f64, f64) {
        let n = 2 * a.len(); // numpy counts doubles
        if n < 8 {
            let (mut rr, mut ri) = (-0.0f64, -0.0f64);
            for c in a {
                rr += c.re;
                ri += c.im;
            }
            (rr, ri)
        } else if n <= 128 {
            let mut r = [0.0f64; 8];
            for q in 0..4 {
                r[2 * q] = a[q].re;
                r[2 * q + 1] = a[q].im;
            }
            let full = a.len() - a.len() % 4;
            let mut i = 4;
            while i < full {
                for q in 0..4 {
                    r[2 * q] += a[i + q].re;
                    r[2 * q + 1] += a[i + q].im;
                }
                i += 4;
            }
            let mut rr = (r[0] + r[2]) + (r[4] + r[6]);
            let mut ri = (r[1] + r[3]) + (r[5] + r[7]);
            for c in &a[full..] {
                rr += c.re;
                ri += c.im;
            }
            (rr, ri)
        } else {
            // n2 = n / 2; n2 -= n2 % 8 (in doubles)
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            let (h1, h2) = a.split_at(n2 / 2);
            let (r1, i1) = pw(h1);
            let (r2, i2) = pw(h2);
            (r1 + r2, i1 + i2)
        }
    }
    let (r, i) = pw(a);
    Complex64::new(0.0 + r, 0.0 + i)
}

/// `np.maximum(x, 0)` (NaN propagates, unlike `f64::max`).
#[inline]
fn np_max0(x: f64) -> f64 {
    if x.is_nan() || x >= 0.0 { x } else { 0.0 }
}

/// `np.clip(x, lo, hi)` = `minimum(maximum(x, lo), hi)`, NaN propagates.
#[inline]
fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() { x } else { x.max(lo).min(hi) }
}

/// numpy `np.pad(a, (l, r), mode="reflect", reflect_type=...)` for 1-D input, including the
/// iterative re-reflection when a pad exceeds `len(a) - 1`. `odd(edge, v)` computes the
/// odd reflection `2*edge - v`; `None` = even reflection.
fn np_pad_reflect<T: Copy + Default>(
    a: &[T],
    l: usize,
    r: usize,
    odd: Option<&dyn Fn(T, T) -> T>,
) -> Vec<T> {
    let n = a.len();
    let mut p = vec![T::default(); n + l + r];
    p[l..l + n].copy_from_slice(a);
    if n == 1 {
        // legacy: singleton extended with its edge value
        for v in p[..l].iter_mut() {
            *v = a[0];
        }
        for v in p[l + 1..].iter_mut() {
            *v = a[0];
        }
        return p;
    }
    let total = p.len();
    let (mut lp, mut rp) = (l, r);
    while lp > 0 || rp > 0 {
        let mut old = total - rp - lp;
        old = (old - 1) / (n - 1) * (n - 1) + 1;
        old -= 1;
        if lp > 0 {
            let cl = old.min(lp);
            let edge = p[lp];
            for t in 0..cl {
                let v = p[lp + cl - t];
                p[lp - cl + t] = match odd {
                    Some(f) => f(edge, v),
                    None => v,
                };
            }
            lp -= cl;
        }
        if rp > 0 {
            let cl = old.min(rp);
            let edge = p[total - rp - 1];
            for t in 0..cl {
                let v = p[total - rp - 2 - t];
                p[total - rp + t] = match odd {
                    Some(f) => f(edge, v),
                    None => v,
                };
            }
            rp -= cl;
        }
    }
    p
}

// ---------------------------------------------------------------- analyze helpers ----

/// `integ(f, fs)`: trapezoidal phase [rad] of frequency f [Hz] (sequential cumsum).
pub fn integ(f: &[f64], fs: f64) -> Vec<f64> {
    let c = 2.0 * std::f64::consts::PI / fs;
    let mut out = Vec::with_capacity(f.len().max(1));
    out.push(c * 0.0);
    let mut acc = 0.0f64;
    for w in f.windows(2) {
        acc += (w[1] + w[0]) / 2.0;
        out.push(c * acc);
    }
    out
}

/// `cdiff_freq(v, fs)`: zero-phase instantaneous frequency of complex v [Hz] (len >= 3).
pub fn cdiff_freq(v: &[Complex64], fs: f64) -> Vec<f64> {
    let n = v.len();
    let mut f = vec![0.0; n];
    if n < 3 {
        // numpy leaves np.empty garbage here; not reached for real inputs
        return f;
    }
    let k = 4.0 * std::f64::consts::PI;
    for i in 1..n - 1 {
        let p = np_cmul(v[i + 1], v[i - 1].conj());
        f[i] = p.im.atan2(p.re) * fs / k;
    }
    f[0] = f[1];
    f[n - 1] = f[n - 2];
    f
}

/// `fir0(a, h)` for real `a`: zero-phase FIR (odd-length h) with odd-reflection edges.
pub fn fir0(a: &[f64], h: &[f64]) -> Vec<f64> {
    let m = h.len() / 2;
    let ext = np_pad_reflect(a, m, m, Some(&|e: f64, v: f64| 2.0 * e - v));
    oaconvolve(&ext, h, ConvMode::Valid)
}

/// `fir0(a, h, phasor=True)`: phase-odd / amplitude-even edge extension around the mean
/// phasor of the first / last `len(h)//8` samples.
pub fn fir0_phasor(a: &[Complex64], h: &[f64]) -> Vec<Complex64> {
    let m = h.len() / 2;
    let n = a.len();
    let k = (h.len() / 8).min(n).max(1);
    let unit = |s: Complex64| exp_j(s.im.atan2(s.re));
    let (p0, p1) = (unit(np_csum(&a[..k])), unit(np_csum(&a[n - k..])));
    let ext = if m == 0 {
        a.to_vec()
    } else {
        let a0: Vec<Complex64> = a.iter().map(|&v| np_cmul(v, p0.conj())).collect();
        let b = np_pad_reflect(&a0, m, 0, None);
        let a1: Vec<Complex64> = a.iter().map(|&v| np_cmul(v, p1.conj())).collect();
        let c = np_pad_reflect(&a1, 0, m, None);
        let mut ext = Vec::with_capacity(n + 2 * m);
        ext.extend(b[..m].iter().map(|v| np_cmul(v.conj(), p0)));
        ext.extend_from_slice(a);
        ext.extend(c[c.len() - m..].iter().map(|v| np_cmul(v.conj(), p1)));
        ext
    };
    oaconvolve_c(&ext, h, ConvMode::Valid)
}

/// `runmed(a, fs, sec)`: 50 ms block medians, median-filtered over `sec`, interpolated.
pub fn runmed(a: &[f64], fs: f64, sec: f64) -> Res<Vec<f64>> {
    let k = ((0.05 * fs) as usize).max(1);
    let nb = (a.len() / k).max(1);
    if a.len() < nb * k {
        return Err(ValueError(format!(
            "cannot reshape array of size {} into shape ({nb},{k})",
            a.len()
        )));
    }
    let bm: Vec<f64> = a[..nb * k].chunks_exact(k).map(median).collect();
    let bm = median_filter(&bm, ((sec / 0.05) as usize).max(1) | 1, Mode::Nearest)?;
    let xp: Vec<f64> = (0..nb).map(|j| (j as f64 + 0.5) * k as f64).collect();
    let x: Vec<f64> = (0..a.len()).map(|i| i as f64).collect();
    interp_mono(&x, &xp, &bm)
}

/// `bridge(f, good)`: linear interpolation over the bad samples.
pub fn bridge(f: &[f64], good: &[bool]) -> Res<Vec<f64>> {
    let ng = good.iter().filter(|&&g| g).count();
    if ng == good.len() || ng == 0 {
        return Ok(f.to_vec());
    }
    let xi: Vec<f64> = (0..f.len())
        .filter(|&i| good[i])
        .map(|i| i as f64)
        .collect();
    let fi: Vec<f64> = (0..f.len()).filter(|&i| good[i]).map(|i| f[i]).collect();
    let mut hint = 0usize;
    Ok((0..f.len())
        .map(|i| {
            if good[i] {
                f[i]
            } else {
                interp_one_by(i as f64, &xi, |j| fi[j], &mut hint)
            }
        })
        .collect())
}

/// `np.interp(x, xp, fp)` for non-decreasing `x` (all call sites here): same arithmetic as
/// [`interp`], with the interval search hint carried from one query to the next.
pub fn interp_mono(x: &[f64], xp: &[f64], fp: &[f64]) -> Res<Vec<f64>> {
    if xp.is_empty() || xp.len() != fp.len() {
        return interp(x, xp, fp);
    }
    let mut hint = 0usize;
    Ok(x.iter()
        .map(|&v| interp_one_by(v, xp, |j| fp[j], &mut hint))
        .collect())
}

fn sq_abs(v: &[Complex64]) -> Vec<f64> {
    v.iter()
        .map(|&c| {
            let a = np_cabs(c);
            a * a
        })
        .collect()
}

// ----------------------------------------------------------------- track -------------

/// Decimation of [`track`]: `(W, pb, D2, fs_p, h2)`.
pub fn track_design(
    fs_d: f64,
    ref_hz: f64,
    bw: f64,
    fmax: f64,
) -> Res<(f64, f64, usize, f64, Vec<f64>)> {
    let w = 4.0 * bw + 0.003 * ref_hz;
    let pb = if 0.55 * bw < fmax { 0.55 * bw } else { fmax };
    let den = if 30.0 * bw > 2.5 * w {
        30.0 * bw
    } else {
        2.5 * w
    };
    let d2 = ((fs_d / den) as usize).max(1);
    let fs_p = fs_d / d2 as f64;
    let fstop = if 1.6 * w < fs_p - w {
        1.6 * w
    } else {
        fs_p - w
    };
    let h2 = lpf(w, fstop, fs_d, 90.0)?;
    Ok((w, pb, d2, fs_p, h2))
}

/// `zp` of [`track`]: baseband mixed down by the coarse track and decimated to fs_p (the input
/// of both PLL loops).
pub fn track_zp(
    z: &[Complex64],
    fs_d: f64,
    fcd: &[f64],
    d2: usize,
    h2: &[f64],
) -> Res<Vec<Complex64>> {
    let zp = derot(z, &integ(fcd, fs_d));
    if d2 > 1 {
        resample_poly_down_c(&zp, d2, h2)
    } else {
        Ok(oaconvolve_c(&zp, h2, ConvMode::Same))
    }
}

/// `track(z, fs_d, fcd, ref_hz, bw, fmax)`: stages 3-5 for one loop bandwidth on the
/// stage-1 baseband `z` (rate `fs_d`) with the coarse track `fcd` interpolated onto z's grid.
pub fn track(
    z: &[Complex64],
    fs_d: f64,
    fcd: &[f64],
    ref_hz: f64,
    bw: f64,
    fmax: f64,
) -> Res<TrackOut> {
    let (w, _, d2, fs_p, h2) = track_design(fs_d, ref_hz, bw, fmax)?;
    let zp = track_zp(z, fs_d, fcd, d2, &h2)?;
    let (fwd, bwd) = pll::pll_fwd_bwd(&zp, fs_p, bw, w);
    track_post(z, fs_d, fcd, ref_hz, bw, fmax, &zp, &fwd, &bwd)
}

/// Stages 4-5 of [`track`] from given `zp` and PLL outputs (`bwd` raw, in reversed time as
/// [`pll::pll`] returns it). Used by [`track`] and by the stage tests to inject stored
/// reference values.
#[allow(clippy::too_many_arguments)]
pub fn track_post(
    z: &[Complex64],
    fs_d: f64,
    fcd: &[f64],
    ref_hz: f64,
    bw: f64,
    fmax: f64,
    zp: &[Complex64],
    fwd: &PllOut,
    bwd: &PllOut,
) -> Res<TrackOut> {
    let (w, pb, d2, fs_p, _) = track_design(fs_d, ref_hz, bw, fmax)?;
    let n = zp.len();
    let tp: Vec<f64> = (0..n).map(|i| i as f64 / fs_p).collect();
    let sm =
        |a: &[f64], sec: f64| uniform_filter1d(a, ((sec * fs_p) as usize).max(1), Mode::Nearest);
    let tau = 2.0 * std::f64::consts::PI;

    // PLL forward and time-reversed (conj keeps frequency sign)
    let th_f = &fwd.th;
    let lk_f = fwd.lock.clone();
    let th_b: Vec<f64> = bwd.th.iter().rev().map(|&t| -t).collect();
    let lk_b: Vec<f64> = bwd.lock.iter().rev().copied().collect();
    let ff = cdiff_freq(&th_f.iter().map(|&t| exp_j(t)).collect::<Vec<_>>(), fs_p);
    let fb = cdiff_freq(&th_b.iter().map(|&t| exp_j(t)).collect::<Vec<_>>(), fs_p);
    let ramp: Vec<f64> = tp
        .iter()
        .map(|&t| np_clip(t * bw / 3.0, 1e-3, 1.0))
        .collect();
    let mut f: Vec<f64> = (0..n)
        .map(|i| {
            let cf = np_clip(lk_f[i], 1e-3, 1.0);
            let cb = np_clip(lk_b[i], 1e-3, 1.0);
            let wf = cf * cf * ramp[i];
            let wb = cb * cb * ramp[n - 1 - i];
            (wf * ff[i] + wb * fb[i]) / (wf + wb)
        })
        .collect();
    // slip guard
    let dth: Vec<f64> = (0..n).map(|i| th_f[i] - th_b[i]).collect();
    let k: Vec<f64> = sm(&dth, 0.05)?
        .iter()
        .map(|&v| (v / tau).round_ties_even())
        .collect();
    let mut jumps = vec![false; n];
    for i in 1..n {
        jumps[i] = k[i] != k[i - 1];
    }
    let slip_n = (((2.0 / pb).min(0.3) * fs_p) as usize).max(1) | 1;
    let slip = binary_dilation(&jumps, slip_n)?;
    let agree = sm(&dth.iter().map(|v| v.cos()).collect::<Vec<_>>(), 0.5)?;
    let mut good: Vec<bool> = (0..n)
        .map(|i| !slip[i] && agree[i] > 0.3 && np_maxf(lk_f[i], lk_b[i]) > 0.3)
        .collect();

    // refinement: zero-phase discriminator on the PLL-derotated residual
    let h = lpf(pb, 1.55 * pb, fs_p, 60.0)?;
    let dil = ((1.0 / pb * fs_p) as usize).max(1) | 1;
    f = bridge(&f, &good)?;
    let mut dips = vec![false; n];
    for _ in 0..3 {
        let x1 = derot(zp, &integ(&f, fs_p));
        let xg: Vec<Complex64> = x1
            .iter()
            .zip(&good)
            .map(|(&v, &g)| np_cmul(v, Complex64::new(g as u8 as f64, 0.0)))
            .collect();
        let (u, v) = rayon::join(|| fir0_phasor(&x1, &h), || fir0_phasor(&xg, &h));
        let a: Vec<f64> = u.iter().map(|&c| np_cabs(c)).collect();
        let rm = runmed(&a, fs_p, 1.0)?;
        let low: Vec<bool> = a.iter().zip(&rm).map(|(&x, &r)| x < 0.3 * r).collect();
        dips = binary_dilation(&low, dil)?;
        let fl = fir0(&f, &h);
        let cd = cdiff_freq(&v, fs_p);
        let sum: Vec<f64> = fl.iter().zip(&cd).map(|(a, b)| a + b).collect();
        let gd: Vec<bool> = good.iter().zip(&dips).map(|(&g, &d)| g && !d).collect();
        f = bridge(&sum, &gd)?;
    }
    for (g, &d) in good.iter_mut().zip(&dips) {
        *g = *g && !d;
    }

    // confidence: SNR in the measurement band; noise density from the stage-1 baseband
    let u = derot(zp, &integ(&f, fs_p));
    let v = fir0_phasor(&u, &h);
    let nd_len = z.len();
    let td: Vec<f64> = (0..nd_len).map(|i| i as f64 / fs_d).collect();
    let fi = interp_mono(&td, &tp, &f)?;
    let fsum: Vec<f64> = fcd.iter().zip(&fi).map(|(a, b)| a + b).collect();
    let ud = derot(z, &integ(&fsum, fs_d));
    let lo = 0.012 * ref_hz;
    let hi = if 0.45 * fs_d < 0.03 * ref_hz {
        0.45 * fs_d
    } else {
        0.03 * ref_hz
    };
    let (sh, sl) = (
        butter_lowpass_sos(4, hi, fs_d)?,
        butter_lowpass_sos(4, lo, fs_d)?,
    );
    let (yh, yl) = rayon::join(
        || sosfiltfilt(&sh, &ud, PadType::Odd, None),
        || sosfiltfilt(&sl, &ud, PadType::Odd, None),
    );
    let (yh, yl) = (yh?, yl?);
    let nd2: Vec<f64> = yh
        .iter()
        .zip(&yl)
        .map(|(&a, &b)| {
            let m = np_cabs(a - b);
            m * m
        })
        .collect();
    let pn = uniform_filter1d(&nd2, ((0.2 * fs_d) as usize).max(1), Mode::Nearest)?;
    let n0: Vec<f64> = interp_mono(&tp, &td, &pn)?
        .iter()
        .map(|&p| p / (2.0 * (hi - lo)) + 1e-30)
        .collect();
    let bm = 1.25 * pb;
    let v2 = sq_abs(&v);
    let sv2 = sm(&v2, 0.2)?;
    let s_: Vec<f64> = (0..n).map(|i| np_max0(sv2[i] - n0[i] * 2.0 * bm)).collect();
    let snr_db: Vec<f64> = (0..n)
        .map(|i| 10.0 * (s_[i] / (n0[i] * 2.0 * bm) + 1e-30).log10())
        .collect();
    // near-line interference
    let sn = butter_lowpass_sos(
        4,
        if 0.9 * w < 4.0 * pb {
            0.9 * w
        } else {
            4.0 * pb
        },
        fs_p,
    )?;
    let nn: Vec<Complex64> = sosfiltfilt(&sn, &u, PadType::Odd, None)?
        .iter()
        .zip(&v)
        .map(|(&a, &b)| a - b)
        .collect();
    let qn = sm(&sq_abs(&nn), 0.1)?;
    let qd = sm(&v2, 0.1)?;
    let q: Vec<f64> = qn.iter().zip(&qd).map(|(&a, &b)| a / (b + 1e-30)).collect();
    // in-band interference: fast envelope modulation
    let env: Vec<f64> = v.iter().map(|&c| np_cabs(c)).collect();
    let se = sm(&env, 0.5)?;
    let dev: Vec<f64> = env
        .iter()
        .zip(&se)
        .map(|(&e, &s)| {
            let d = e - s;
            d * d
        })
        .collect();
    let sd = sm(&dev, 0.2)?;
    let am: Vec<f64> = sd
        .iter()
        .zip(&se)
        .map(|(&d, &s)| np_max0(d).sqrt() / (s + 1e-30))
        .collect();
    let (rq, ra) = (runmed(&q, fs_p, 10.0)?, runmed(&am, fs_p, 10.0)?);
    let intf0: Vec<bool> = (0..n)
        .map(|i| (q[i] > 0.5 && q[i] > 3.0 * rq[i]) || (am[i] > 0.25 && am[i] > 2.5 * ra[i]))
        .collect();
    let intf = binary_dilation(&intf0, ((0.2 * fs_p) as usize).max(1) | 1)?;
    for (g, &x) in good.iter_mut().zip(&intf) {
        *g = *g && !x;
    }
    let inband: Vec<bool> = f.iter().map(|&x| x.abs() < 0.9 * w).collect();
    let conf: Vec<f64> = (0..n)
        .map(|i| {
            np_clip(snr_db[i] / 12.0, 0.0, 1.0)
                * np_clip(agree[i] / 0.6, 0.0, 1.0)
                * (inband[i] as u8 as f64)
                * (good[i] as u8 as f64)
        })
        .collect();
    let n3 = ((0.3 * fs_p) as usize).max(1);
    let valid = binary_opening(&conf.iter().map(|&c| c >= 0.5).collect::<Vec<_>>(), n3)?;
    let valid = binary_erosion(&valid, n3 | 1)?;
    let soft: Vec<bool> = (0..n)
        .map(|i| !valid[i] && snr_db[i] >= 3.0 && !slip[i] && agree[i] > 0.3 && inband[i])
        .collect();
    let fc: Vec<f64> = fcd.iter().step_by(d2).take(n).copied().collect();
    let cn0: Vec<f64> = (0..n).map(|i| s_[i] / n0[i]).collect();
    Ok(TrackOut {
        tp,
        fc,
        f,
        ff,
        fb,
        lk_f,
        lk_b,
        soft,
        agree,
        snr_db,
        conf,
        valid,
        fs_p,
        bw,
        pb,
        slip,
        dip: dips,
        intf,
        cn0,
    })
}

/// `np.maximum(a, b)` for scalars (NaN propagates).
#[inline]
fn np_maxf(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_reflect_matches_numpy() {
        // np.pad([1,2,3], 5, mode="reflect") = [2 3 2 1 2 | 1 2 3 | 2 1 2 3 2]
        let a = [1.0, 2.0, 3.0];
        assert_eq!(
            np_pad_reflect(&a, 5, 5, None),
            vec![
                2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0
            ]
        );
        // values from numpy 2.5.4
        let a = [1.0, 2.0, 4.0];
        let odd = |e: f64, v: f64| 2.0 * e - v;
        assert_eq!(
            np_pad_reflect(&a, 7, 7, Some(&odd)),
            vec![
                -10.0, -8.0, -6.0, -5.0, -4.0, -2.0, 0.0, 1.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 12.0,
                13.0, 14.0
            ]
        );
        assert_eq!(np_pad_reflect(&[5.0], 2, 1, Some(&odd)), vec![5.0; 4]);
        assert_eq!(
            np_pad_reflect(&[1.0, 2.0], 3, 4, Some(&odd)),
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn integ_and_bridge() {
        let p = integ(&[1.0, 3.0, 5.0], 2.0);
        let c = 2.0 * std::f64::consts::PI / 2.0;
        assert_eq!(p, vec![0.0, c * 2.0, c * 6.0]);
        let b = bridge(
            &[0.0, 9.0, 2.0, 9.0, 4.0],
            &[true, false, true, false, true],
        )
        .unwrap();
        assert_eq!(b, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn csum_sign_and_order() {
        let z = Complex64::new(-0.0, -0.0);
        let s = np_csum(&[z; 9]);
        assert!(s.re.is_sign_positive() && s.im.is_sign_positive());
        let v: Vec<Complex64> = (0..1000)
            .map(|i| Complex64::new(i as f64 * 0.1, 1.0 / (i + 1) as f64))
            .collect();
        let s = np_csum(&v);
        assert!((s.re - 49950.0).abs() < 1e-9);
    }
}
