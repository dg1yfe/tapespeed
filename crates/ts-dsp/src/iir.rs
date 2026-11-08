//! IIR design and zero-phase filtering with SciPy-compatible definitions:
//! `iirnotch`, `lfilter` (direct form II transposed), `lfilter_zi`, `filtfilt` (method "pad"),
//! `butter(N, fc, fs=fs, output="sos")` (lowpass: bilinear + prewarp, `zpk2sos` pairing
//! "nearest"), `sosfilt`, `sosfilt_zi`, `sosfiltfilt` (real and complex input),
//! and a block-streaming `filtfilt` ([`FiltFiltStream`]).
//!
//! Operation order follows SciPy/numpy (`_sosfilt`, `lfilter.c`, numpy complex division) so
//! results agree to rounding; recursions stay scalar and in sample order, so results do not
//! depend on the thread count.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

use rustfft::num_complex::Complex64;

use crate::stats::np_sum;
use crate::{ValueError, verr as err};

/// Sample type accepted by the filters: `f64` or `Complex64` (real coefficients).
pub trait Sample:
    Copy + Default + Add<Output = Self> + Sub<Output = Self> + Mul<f64, Output = Self> + Send + Sync
{
}
impl Sample for f64 {}
impl Sample for Complex64 {}

/// Edge extension used by `filtfilt` / `sosfiltfilt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadType {
    Odd,
    Even,
    Constant,
    None,
}

// ---------------------------------------------------------------- numpy complex helpers

/// numpy complex multiply.
#[inline]
pub fn cmul(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// numpy complex division (Smith's algorithm with reciprocal scale, `loops.c.src`).
#[inline]
pub fn cdiv(a: Complex64, b: Complex64) -> Complex64 {
    let (in1r, in1i, in2r, in2i) = (a.re, a.im, b.re, b.im);
    let in2r_abs = in2r.abs();
    let in2i_abs = in2i.abs();
    if in2r_abs >= in2i_abs {
        if in2r_abs == 0.0 && in2i_abs == 0.0 {
            Complex64::new(in1r / in2r_abs, in1i / in2i_abs)
        } else {
            let rat = in2i / in2r;
            let scl = 1.0 / (in2r + in2i * rat);
            Complex64::new((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl)
        }
    } else {
        let rat = in2r / in2i;
        let scl = 1.0 / (in2i + in2r * rat);
        Complex64::new((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl)
    }
}

// ---------------------------------------------------------------- design

/// `scipy.signal.iirnotch(w0, Q, fs)` -> (b, a).
pub fn iirnotch(w0: f64, q: f64, fs: f64) -> Result<([f64; 3], [f64; 3]), ValueError> {
    let w0 = 2.0 * w0 / fs;
    if !(0.0..=1.0).contains(&w0) {
        return err("w0 should be such that 0 < w0 < 1");
    }
    let bw = w0 / q;
    let bw = bw * PI;
    let w0 = w0 * PI;
    let beta = (bw / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    let c = w0.cos();
    let b = [gain * 1.0, gain * (-2.0 * c), gain * 1.0];
    let a = [1.0, -2.0 * gain * c, 2.0 * gain - 1.0];
    Ok((b, a))
}

fn is_real(z: Complex64) -> bool {
    z.im == 0.0
}

/// `_cplxreal`: (complex with positive imag, sorted), (reals, sorted).
fn cplxreal(z: &[Complex64]) -> Result<(Vec<Complex64>, Vec<f64>), ValueError> {
    if z.is_empty() {
        return Ok((vec![], vec![]));
    }
    let tol = 100.0 * f64::EPSILON;
    let mut z: Vec<Complex64> = z.to_vec();
    // lexsort((abs(imag), real)): primary real, secondary abs(imag); stable
    z.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap()
            .then(a.im.abs().partial_cmp(&b.im.abs()).unwrap())
    });
    let mut zr = vec![];
    let mut zc = vec![];
    for v in &z {
        if v.im.abs() <= tol * v.re.hypot(v.im) {
            zr.push(v.re);
        } else {
            zc.push(*v);
        }
    }
    if zc.is_empty() {
        return Ok((vec![], zr));
    }
    let mut zp: Vec<Complex64> = zc.iter().copied().filter(|v| v.im > 0.0).collect();
    let mut zn: Vec<Complex64> = zc.iter().copied().filter(|v| v.im < 0.0).collect();
    if zp.len() != zn.len() {
        return err("Array contains complex value with no matching conjugate.");
    }
    // runs of equal real part: sort each run by abs(imag)
    let m = zp.len();
    let mut i = 0;
    while i < m {
        let mut j = i;
        while j + 1 < m && zp[j + 1].re - zp[j].re <= tol * zp[j].re.hypot(zp[j].im) {
            j += 1;
        }
        if j > i {
            for chunk in [&mut zp[i..=j], &mut zn[i..=j]] {
                chunk.sort_by(|a, b| a.im.abs().partial_cmp(&b.im.abs()).unwrap());
            }
        }
        i = j + 1;
    }
    for (p, n) in zp.iter().zip(&zn) {
        let d = *p - n.conj();
        if d.re.hypot(d.im) > tol * n.re.hypot(n.im) {
            return err("Array contains complex value with no matching conjugate.");
        }
    }
    let out = zp
        .iter()
        .zip(&zn)
        .map(|(p, n)| {
            let s = *p + n.conj();
            Complex64::new(s.re / 2.0, s.im / 2.0)
        })
        .collect();
    Ok((out, zr))
}

/// `np.poly` of complex roots, real part returned (all call sites pass conjugate-closed sets).
fn poly(roots: &[Complex64]) -> Vec<f64> {
    let mut a = vec![Complex64::new(1.0, 0.0)];
    let one = Complex64::new(1.0, 0.0);
    for &r in roots {
        let v = [one, -r];
        let mut na = vec![Complex64::new(0.0, 0.0); a.len() + 1];
        for (i, slot) in na.iter_mut().enumerate() {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, vj) in v.iter().enumerate() {
                if i >= j && i - j < a.len() {
                    acc += cmul(a[i - j], *vj);
                }
            }
            *slot = acc;
        }
        a = na;
    }
    a.iter().map(|c| c.re).collect()
}

fn single_zpksos(z: &[Complex64], p: &[Complex64]) -> [f64; 6] {
    let b = poly(z);
    let a = poly(p);
    let mut s = [0.0; 6];
    s[3 - b.len()..3].copy_from_slice(&b);
    s[6 - a.len()..6].copy_from_slice(&a);
    s
}

fn argmin_first(v: impl Iterator<Item = f64>) -> usize {
    let mut best = f64::INFINITY;
    let mut idx = 0;
    for (i, x) in v.enumerate() {
        if i == 0 || x < best {
            best = x;
            idx = i;
        }
    }
    idx
}

#[derive(Clone, Copy, PartialEq)]
enum Which {
    Real,
    Complex,
    Any,
}

fn nearest_idx(fro: &[Complex64], to: Complex64, which: Which) -> usize {
    let mut order: Vec<usize> = (0..fro.len()).collect();
    let d: Vec<f64> = fro
        .iter()
        .map(|f| {
            let e = *f - to;
            e.re.hypot(e.im)
        })
        .collect();
    order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());
    match which {
        Which::Any => order[0],
        Which::Real => *order.iter().find(|&&i| is_real(fro[i])).unwrap(),
        Which::Complex => *order.iter().find(|&&i| !is_real(fro[i])).unwrap(),
    }
}

/// `scipy.signal.zpk2sos(z, p, k)` with pairing "nearest" (digital).
pub fn zpk2sos(z: &[Complex64], p: &[Complex64], k: f64) -> Result<Vec<[f64; 6]>, ValueError> {
    if z.is_empty() && p.is_empty() {
        return Ok(vec![[k, 0.0, 0.0, 1.0, 0.0, 0.0]]);
    }
    let mut p = p.to_vec();
    let mut z = z.to_vec();
    let zero = Complex64::new(0.0, 0.0);
    while p.len() < z.len() {
        p.push(zero);
    }
    while z.len() < p.len() {
        z.push(zero);
    }
    let n_sections = p.len().div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(zero);
        z.push(zero);
    }
    let (zc, zr) = cplxreal(&z)?;
    let mut z: Vec<Complex64> = zc
        .into_iter()
        .chain(zr.into_iter().map(|r| Complex64::new(r, 0.0)))
        .collect();
    let (pc, pr) = cplxreal(&p)?;
    let mut p: Vec<Complex64> = pc
        .into_iter()
        .chain(pr.into_iter().map(|r| Complex64::new(r, 0.0)))
        .collect();
    let idx_worst =
        |p: &[Complex64]| argmin_first(p.iter().map(|v| (1.0 - v.re.hypot(v.im)).abs()));
    let mut sos = vec![[0.0f64; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let p1_idx = idx_worst(&p);
        let p1 = p.remove(p1_idx);
        let n_real_p = p.iter().filter(|v| is_real(**v)).count();
        if is_real(p1) && n_real_p == 0 {
            let z1_idx = nearest_idx(&z, p1, Which::Real);
            let z1 = z.remove(z1_idx);
            sos[si] = single_zpksos(&[z1, zero], &[p1, zero]);
        } else if p.len() + 1 == z.len()
            && !is_real(p1)
            && n_real_p == 1
            && z.iter().filter(|v| is_real(**v)).count() == 1
        {
            let z1_idx = nearest_idx(&z, p1, Which::Complex);
            let z1 = z.remove(z1_idx);
            sos[si] = single_zpksos(&[z1, z1.conj()], &[p1, p1.conj()]);
        } else {
            let p2 = if is_real(p1) {
                let preal: Vec<usize> = (0..p.len()).filter(|&i| is_real(p[i])).collect();
                let sub: Vec<Complex64> = preal.iter().map(|&i| p[i]).collect();
                let p2_idx = preal[idx_worst(&sub)];
                p.remove(p2_idx)
            } else {
                p1.conj()
            };
            if !z.is_empty() {
                let z1_idx = nearest_idx(&z, p1, Which::Any);
                let z1 = z.remove(z1_idx);
                if !is_real(z1) {
                    sos[si] = single_zpksos(&[z1, z1.conj()], &[p1, p2]);
                } else if !z.is_empty() {
                    let z2_idx = nearest_idx(&z, p1, Which::Real);
                    let z2 = z.remove(z2_idx);
                    sos[si] = single_zpksos(&[z1, z2], &[p1, p2]);
                } else {
                    sos[si] = single_zpksos(&[z1], &[p1, p2]);
                }
            } else {
                sos[si] = single_zpksos(&[], &[p1, p2]);
            }
        }
    }
    for v in &mut sos[0][..3] {
        *v *= k;
    }
    Ok(sos)
}

/// `scipy.signal.butter(n, fc, btype="low", fs=fs, output="sos")`.
pub fn butter_lowpass_sos(n: usize, fc: f64, fs: f64) -> Result<Vec<[f64; 6]>, ValueError> {
    let wn = fc / (fs / 2.0);
    if wn <= 0.0 {
        return err("filter critical frequencies must be greater than 0");
    }
    if wn >= 1.0 {
        return err(format!(
            "Digital filter critical frequencies must be 0 < Wn < fs/2 (fs={fs} -> fs/2={})",
            fs / 2.0
        ));
    }
    // buttap
    let nn = n as f64;
    let p: Vec<Complex64> = (0..n)
        .map(|i| {
            let m = -(n as f64) + 1.0 + 2.0 * i as f64;
            let th = PI * m / (2.0 * nn);
            -Complex64::new(th.cos(), th.sin())
        })
        .collect();
    // prewarp (fs = 2)
    let warped = 2.0 * 2.0 * (PI * wn / 2.0).tan();
    // lp2lp_zpk: complex multiply by (wo, 0); k = wo**degree
    let wo = Complex64::new(warped, 0.0);
    let p: Vec<Complex64> = p.iter().map(|&v| cmul(wo, v)).collect();
    let k = 1.0 * warped.powf(nn);
    // bilinear_zpk, fs = 2 -> fs2 = 4
    let fs2 = 4.0;
    let pz: Vec<Complex64> = p
        .iter()
        .map(|v| {
            cdiv(
                Complex64::new(fs2 + v.re, v.im),
                Complex64::new(fs2 - v.re, -v.im),
            )
        })
        .collect();
    let zz = vec![Complex64::new(-1.0, 0.0); n];
    let mut prod = Complex64::new(1.0, 0.0);
    for v in &p {
        prod = cmul(prod, Complex64::new(fs2 - v.re, -v.im));
    }
    let kz = k * cdiv(Complex64::new(1.0, 0.0), prod).re;
    zpk2sos(&zz, &pz, kz)
}

// ---------------------------------------------------------------- filtering

/// Normalised, equal-length (b, a) as `lfilter` uses them.
fn normalize(b: &[f64], a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), ValueError> {
    if a.is_empty() || a[0] == 0.0 {
        return err("First coefficient of parameter `a` must be non-zero!");
    }
    let n = a.len().max(b.len());
    let a0 = a[0];
    let mut bb: Vec<f64> = b
        .iter()
        .map(|v| if a0 != 1.0 { v / a0 } else { *v })
        .collect();
    let mut aa: Vec<f64> = a
        .iter()
        .map(|v| if a0 != 1.0 { v / a0 } else { *v })
        .collect();
    bb.resize(n, 0.0);
    aa.resize(n, 0.0);
    Ok((bb, aa))
}

/// `scipy.signal.lfilter_zi(b, a)` (SciPy 1.18 closed form).
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, ValueError> {
    if a.is_empty() || a[0] == 0.0 {
        return err("First coefficient of parameter `a` must be non-zero!");
    }
    let (b, a) = if a[0] != 1.0 {
        (
            b.iter().map(|v| v / a[0]).collect::<Vec<_>>(),
            a.iter().map(|v| v / a[0]).collect::<Vec<_>>(),
        )
    } else {
        (b.to_vec(), a.to_vec())
    };
    let sum_a = np_sum(&a);
    if sum_a == 0.0 {
        return err("Filter not stable due to sum(a) == 0, i.e., having a pole at z = 1!");
    }
    let y_inf = np_sum(&b) / sum_a;
    let n = a.len().max(b.len());
    let d: Vec<f64> = (0..n)
        .map(|i| b.get(i).copied().unwrap_or(0.0) - y_inf * a.get(i).copied().unwrap_or(0.0))
        .collect();
    // flip(cumsum(flip(d)))[1:]
    let mut zi = vec![0.0; n];
    let mut acc = 0.0;
    for i in (0..n).rev() {
        acc = if i == n - 1 { d[i] } else { acc + d[i] };
        zi[i] = acc;
    }
    Ok(zi[1..].to_vec())
}

/// Direct form II transposed over `x`, state `z` (len n-1) updated in place; (b, a) normalised.
#[inline]
fn df2t<T: Sample>(
    b: &[f64],
    a: &[f64],
    z: &mut [T],
    x: impl Iterator<Item = T>,
    mut out: impl FnMut(T),
) {
    let n = b.len();
    if n == 1 {
        for xn in x {
            out(xn * b[0]);
        }
        return;
    }
    if n == 3 {
        let (b0, b1, b2, a1, a2) = (b[0], b[1], b[2], a[1], a[2]);
        let (mut z0, mut z1) = (z[0], z[1]);
        for xn in x {
            let yn = z0 + xn * b0;
            z0 = z1 + xn * b1 - yn * a1;
            z1 = xn * b2 - yn * a2;
            out(yn);
        }
        z[0] = z0;
        z[1] = z1;
        return;
    }
    for xn in x {
        let yn = z[0] + xn * b[0];
        for k in 0..n - 2 {
            z[k] = z[k + 1] + xn * b[k + 1] - yn * a[k + 1];
        }
        z[n - 2] = xn * b[n - 1] - yn * a[n - 1];
        out(yn);
    }
}

/// `scipy.signal.lfilter(b, a, x, zi=zi)` -> (y, zf); zero initial state if `zi` is None.
pub fn lfilter(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    zi: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>), ValueError> {
    let (b, a) = normalize(b, a)?;
    let mut z = match zi {
        Some(z) => {
            if z.len() != b.len() - 1 {
                return err("Unexpected shape for zi");
            }
            z.to_vec()
        }
        None => vec![0.0; b.len() - 1],
    };
    let mut y = Vec::with_capacity(x.len());
    df2t(&b, &a, &mut z, x.iter().copied(), |v| y.push(v));
    Ok((y, z))
}

fn validate_pad<T: Sample>(
    padtype: PadType,
    padlen: Option<usize>,
    x: &[T],
    ntaps: usize,
) -> Result<(usize, Vec<T>), ValueError> {
    let edge = if padtype == PadType::None {
        0
    } else {
        padlen.unwrap_or(3 * ntaps)
    };
    if x.len() <= edge {
        return err(format!(
            "The length of the input vector x must be greater than padlen, which is {edge}."
        ));
    }
    if padtype == PadType::None || edge == 0 {
        return Ok((edge, x.to_vec()));
    }
    let n = x.len();
    let mut ext = Vec::with_capacity(n + 2 * edge);
    let (x0, xl) = (x[0], x[n - 1]);
    for k in 0..edge {
        let v = x[edge - k];
        ext.push(match padtype {
            PadType::Odd => x0 * 2.0 - v,
            PadType::Even => v,
            _ => x0,
        });
    }
    ext.extend_from_slice(x);
    for k in 0..edge {
        let v = x[n - 2 - k];
        ext.push(match padtype {
            PadType::Odd => xl * 2.0 - v,
            PadType::Even => v,
            _ => xl,
        });
    }
    Ok((edge, ext))
}

/// `scipy.signal.filtfilt(b, a, x, padtype=..., padlen=...)` (method "pad").
pub fn filtfilt(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    padtype: PadType,
    padlen: Option<usize>,
) -> Result<Vec<f64>, ValueError> {
    let ntaps = a.len().max(b.len());
    let (edge, ext) = validate_pad(padtype, padlen, x, ntaps)?;
    let zi = lfilter_zi(b, a)?;
    let (bn, an) = normalize(b, a)?;
    let mut z: Vec<f64> = zi.iter().map(|v| v * ext[0]).collect();
    let mut y = Vec::with_capacity(ext.len());
    df2t(&bn, &an, &mut z, ext.iter().copied(), |v| y.push(v));
    let y0 = *y.last().unwrap();
    let mut z: Vec<f64> = zi.iter().map(|v| v * y0).collect();
    let mut r = vec![0.0; y.len()];
    let mut k = y.len();
    df2t(&bn, &an, &mut z, y.iter().rev().copied(), |v| {
        k -= 1;
        r[k] = v;
    });
    r.truncate(r.len() - edge);
    r.drain(..edge);
    Ok(r)
}

/// `scipy.signal.sosfilt_zi(sos)`.
pub fn sosfilt_zi(sos: &[[f64; 6]]) -> Result<Vec<[f64; 2]>, ValueError> {
    let mut zi = Vec::with_capacity(sos.len());
    let mut scale = 1.0;
    for s in sos {
        let (b, a) = (&s[..3], &s[3..]);
        let z = lfilter_zi(b, a)?;
        zi.push([scale * z[0], scale * z[1]]);
        scale *= np_sum(b) / np_sum(a);
    }
    Ok(zi)
}

fn validate_sos(sos: &[[f64; 6]]) -> Result<(), ValueError> {
    if sos.is_empty() {
        return err("sos array must be shape (n_sections, 6)");
    }
    if sos.iter().any(|s| s[3] != 1.0) {
        return err("sos[:, 3] should be all ones");
    }
    Ok(())
}

/// `scipy.signal.sosfilt(sos, x, zi=zi)` in place; `zi` (per section) updated to `zf`.
pub fn sosfilt_inplace<T: Sample>(sos: &[[f64; 6]], x: &mut [T], zi: &mut [[T; 2]]) {
    for xv in x.iter_mut() {
        let mut x_cur = *xv;
        for (s, z) in sos.iter().zip(zi.iter_mut()) {
            let x_new = x_cur * s[0] + z[0];
            z[0] = x_cur * s[1] - x_new * s[4] + z[1];
            z[1] = x_cur * s[2] - x_new * s[5];
            x_cur = x_new;
        }
        *xv = x_cur;
    }
}

/// `scipy.signal.sosfilt(sos, x, zi)` -> (y, zf).
pub fn sosfilt<T: Sample>(
    sos: &[[f64; 6]],
    x: &[T],
    zi: Option<&[[T; 2]]>,
) -> Result<(Vec<T>, Vec<[T; 2]>), ValueError> {
    validate_sos(sos)?;
    let mut z = match zi {
        Some(z) => z.to_vec(),
        None => vec![[T::default(); 2]; sos.len()],
    };
    let mut y = x.to_vec();
    sosfilt_inplace(sos, &mut y, &mut z);
    Ok((y, z))
}

/// `scipy.signal.sosfiltfilt(sos, x, padtype=..., padlen=...)`; real or complex `x`.
pub fn sosfiltfilt<T: Sample>(
    sos: &[[f64; 6]],
    x: &[T],
    padtype: PadType,
    padlen: Option<usize>,
) -> Result<Vec<T>, ValueError> {
    validate_sos(sos)?;
    let ns = sos.len();
    let zb2 = sos.iter().filter(|s| s[2] == 0.0).count();
    let za2 = sos.iter().filter(|s| s[5] == 0.0).count();
    let ntaps = 2 * ns + 1 - zb2.min(za2);
    let (edge, mut ext) = validate_pad(padtype, padlen, x, ntaps)?;
    let zi = sosfilt_zi(sos)?;
    let x0 = ext[0];
    let mut z: Vec<[T; 2]> = zi.iter().map(|v| [x0 * v[0], x0 * v[1]]).collect();
    sosfilt_inplace(sos, &mut ext, &mut z);
    let y0 = *ext.last().unwrap();
    ext.reverse();
    let mut z: Vec<[T; 2]> = zi.iter().map(|v| [y0 * v[0], y0 * v[1]]).collect();
    sosfilt_inplace(sos, &mut ext, &mut z);
    ext.reverse();
    ext.truncate(ext.len() - edge);
    ext.drain(..edge);
    Ok(ext)
}

// ---------------------------------------------------------------- streaming filtfilt

/// Block-streaming `filtfilt(b, a, x, padtype="odd", padlen)`.
///
/// Forward pass: exact (carried state, same op sequence as SciPy). Backward pass: per emitted
/// block from zero state, started `lookahead` samples after the block end (error ~ r^lookahead
/// for pole radius r); at the end of the stream the tail is exact (SciPy odd extension + zi).
pub struct FiltFiltStream {
    b: Vec<f64>,
    a: Vec<f64>,
    zi: Vec<f64>,
    padlen: usize,
    lookahead: usize,
    min_emit: usize,
    head: Vec<f64>,
    started: bool,
    zf: Vec<f64>,
    /// forward outputs; `yf[0]` is ext index `yf_off`
    yf: Vec<f64>,
    yf_off: usize,
    /// next ext index to emit
    emit: usize,
    /// last padlen+1 raw inputs
    last: Vec<f64>,
    n: usize,
    tmp: Vec<f64>,
}

impl FiltFiltStream {
    /// `padlen`: SciPy edge (default 3·max(len a, len b)); `lookahead` samples of backward
    /// warm-up per block (16384 in pass 2).
    pub fn new(
        b: &[f64],
        a: &[f64],
        padlen: Option<usize>,
        lookahead: usize,
    ) -> Result<Self, ValueError> {
        let zi = lfilter_zi(b, a)?;
        let ntaps = a.len().max(b.len());
        let (b, a) = normalize(b, a)?;
        let padlen = padlen.unwrap_or(3 * ntaps);
        Ok(Self {
            zf: vec![0.0; b.len() - 1],
            b,
            a,
            zi,
            padlen,
            lookahead,
            min_emit: 65536,
            head: Vec::new(),
            started: false,
            yf: Vec::new(),
            yf_off: 0,
            emit: padlen,
            last: Vec::new(),
            n: 0,
            tmp: Vec::new(),
        })
    }

    fn forward(&mut self, x: &[f64]) {
        let (b, a) = (&self.b, &self.a);
        let yf = &mut self.yf;
        df2t(b, a, &mut self.zf, x.iter().copied(), |v| yf.push(v));
    }

    fn remember(&mut self, x: &[f64]) {
        let keep = self.padlen + 1;
        if x.len() >= keep {
            self.last.clear();
            self.last.extend_from_slice(&x[x.len() - keep..]);
        } else {
            self.last.extend_from_slice(x);
            if self.last.len() > keep {
                let d = self.last.len() - keep;
                self.last.drain(..d);
            }
        }
    }

    /// Backward pass over ext indices [lo, hi) starting from state `z`; writes outputs for
    /// [lo, upto) to `out`.
    fn backward(&mut self, lo: usize, hi: usize, upto: usize, mut z: Vec<f64>, out: &mut Vec<f64>) {
        let s = &self.yf[lo - self.yf_off..hi - self.yf_off];
        self.tmp.clear();
        self.tmp.resize(hi - lo, 0.0);
        let tmp = &mut self.tmp;
        let mut k = hi - lo;
        df2t(&self.b, &self.a, &mut z, s.iter().rev().copied(), |v| {
            k -= 1;
            tmp[k] = v;
        });
        out.extend_from_slice(&self.tmp[..upto - lo]);
    }

    fn start(&mut self) {
        let p = self.padlen;
        let x0 = self.head[0];
        let left: Vec<f64> = (0..p).map(|k| x0 * 2.0 - self.head[p - k]).collect();
        let e0 = if p > 0 { left[0] } else { x0 };
        self.zf = self.zi.iter().map(|v| v * e0).collect();
        self.forward(&left);
        let head = std::mem::take(&mut self.head);
        self.forward(&head);
        self.started = true;
    }

    /// Push samples; appends finalized outputs to `out`.
    pub fn push(&mut self, x: &[f64], out: &mut Vec<f64>) {
        self.n += x.len();
        self.remember(x);
        if !self.started {
            self.head.extend_from_slice(x);
            if self.head.len() > self.padlen {
                self.start();
            }
        } else {
            self.forward(x);
        }
        let end = self.yf_off + self.yf.len();
        if self.started && end >= self.emit + self.lookahead + self.min_emit {
            let upto = end - self.lookahead;
            let z = vec![0.0; self.b.len() - 1];
            self.backward(self.emit, end, upto, z, out);
            self.emit = upto;
            let d = upto - self.yf_off;
            self.yf.drain(..d);
            self.yf_off = upto;
        }
    }

    /// End of stream: exact SciPy tail. Errors like SciPy when the total length <= padlen.
    pub fn finish(&mut self, out: &mut Vec<f64>) -> Result<(), ValueError> {
        if !self.started {
            return err(format!(
                "The length of the input vector x must be greater than padlen, which is {}.",
                self.padlen
            ));
        }
        let p = self.padlen;
        let l = self.last.len();
        let xl = self.last[l - 1];
        let right: Vec<f64> = (0..p).map(|k| xl * 2.0 - self.last[l - 2 - k]).collect();
        self.forward(&right);
        let end = self.yf_off + self.yf.len();
        let y0 = *self.yf.last().unwrap();
        let z = self.zi.iter().map(|v| v * y0).collect();
        self.backward(self.emit, end, self.n + p, z, out);
        self.emit = self.n + p;
        self.yf.clear();
        self.yf_off = end;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdiv_basic() {
        let q = cdiv(Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0));
        assert!((q - Complex64::new(-0.2, 0.4)).norm() < 1e-15);
    }

    #[test]
    fn stream_equals_whole() {
        let (b, a) = iirnotch(1000.0, 30.0, 48000.0).unwrap();
        for n in [10usize, 11, 100, 70000, 200_000] {
            let x: Vec<f64> = (0..n)
                .map(|i| ((i as f64) * 0.37).sin() + 0.1 * ((i * 13 % 7) as f64))
                .collect();
            let w = filtfilt(&b, &a, &x, PadType::Odd, None).unwrap();
            for blk in [1usize, 7, 4096, 100_000] {
                let mut s = FiltFiltStream::new(&b, &a, None, 16384).unwrap();
                let mut y = Vec::new();
                for c in x.chunks(blk) {
                    s.push(c, &mut y);
                }
                s.finish(&mut y).unwrap();
                assert_eq!(y.len(), n);
                let e = w
                    .iter()
                    .zip(&y)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0.0, f64::max);
                assert!(e < 1e-12, "n={n} blk={blk} e={e}");
            }
        }
        let mut s = FiltFiltStream::new(&b, &a, None, 16384).unwrap();
        let mut y = Vec::new();
        s.push(&[1.0; 9], &mut y);
        assert!(s.finish(&mut y).is_err());
        assert!(filtfilt(&b, &a, &[1.0; 9], PadType::Odd, None).is_err());
    }
}
