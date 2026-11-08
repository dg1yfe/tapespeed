//! Convolution and polyphase resampling: `fftconvolve`/`oaconvolve`
//! (full/same/valid), `upfirdn`, `resample_poly(x, 1, D, window=h)`.
//!
//! One engine serves all of them: the full linear convolution `y[k] = Σ_j h[j]·x[k−j]`,
//! evaluated only on the output indices a mode needs. Long kernels use overlap-save with a
//! real FFT (realfft/rustfft, NEON), short ones a direct dot product written for
//! auto-vectorisation (8 independent FMA accumulators). Complex signals with a real kernel
//! run as two real passes (SoA re/im). Results agree with scipy's pocketfft-based
//! convolution to ~1e-16 relative to the signal scale, not bit for bit (test tolerance 1e−10).
//!
//! Parallelism (rayon) splits the output into fixed blocks, each computed independently
//! with the same arithmetic as in a serial run → bit-identical for any thread count.

use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::{ValueError, verr};

/// `mode=` of `fftconvolve`/`oaconvolve`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvMode {
    Full,
    Same,
    Valid,
}

/// Output index range `[k0, k1)` in full-convolution coordinates (scipy `_centered`).
fn mode_range(n: usize, m: usize, mode: ConvMode) -> (usize, usize) {
    match mode {
        ConvMode::Full => (0, n + m - 1),
        ConvMode::Same => {
            let s = (m - 1) / 2;
            (s, s + n)
        }
        ConvMode::Valid => {
            let (lo, hi) = (n.min(m), n.max(m));
            (lo - 1, hi) // length hi - lo + 1
        }
    }
}

// ---------------------------------------------------------------- dot kernels --------

/// Σ a[i]·b[i] with 8 independent FMA accumulators (vectorises to 4×NEON f64x2).
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0f64; 8];
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] = x[i].mul_add(y[i], acc[i]);
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s = x.mul_add(*y, s);
    }
    s
}

/// Two dot products sharing `h` (SoA complex signal with a real kernel).
#[inline]
pub(crate) fn dot2(h: &[f64], xr: &[f64], xi: &[f64]) -> (f64, f64) {
    let n = h.len().min(xr.len()).min(xi.len());
    let (h, xr, xi) = (&h[..n], &xr[..n], &xi[..n]);
    let ch = h.chunks_exact(4);
    let cr = xr.chunks_exact(4);
    let ci = xi.chunks_exact(4);
    let (rh, rr, ri) = (ch.remainder(), cr.remainder(), ci.remainder());
    let mut ar = [0.0f64; 4];
    let mut ai = [0.0f64; 4];
    for ((hh, r), i) in ch.zip(cr).zip(ci) {
        for k in 0..4 {
            ar[k] = hh[k].mul_add(r[k], ar[k]);
            ai[k] = hh[k].mul_add(i[k], ai[k]);
        }
    }
    let mut sr = (ar[0] + ar[2]) + (ar[1] + ar[3]);
    let mut si = (ai[0] + ai[2]) + (ai[1] + ai[3]);
    for ((hh, r), i) in rh.iter().zip(rr).zip(ri) {
        sr = hh.mul_add(*r, sr);
        si = hh.mul_add(*i, si);
    }
    (sr, si)
}

/// Slices for full-convolution output `k`: `(hr_range, x_range)` with `hr` = reversed `h`,
/// or `None` if no tap overlaps the signal.
#[inline]
fn tap_ranges(k: usize, n: usize, m: usize) -> Option<(usize, usize, usize)> {
    // i ∈ [max(0, k-(n-1)), min(m-1, k)]
    let ilo = k.saturating_sub(n - 1);
    let ihi = k.min(m - 1);
    if ilo > ihi {
        return None;
    }
    // hr[m-1-ihi ..= m-1-ilo] · x[k-ihi ..= k-ilo]
    Some((m - 1 - ihi, k - ihi, ihi - ilo + 1))
}

// ---------------------------------------------------------------- direct engine ------

const PAR_BLOCK: usize = 1 << 13;

/// Direct evaluation of full-convolution outputs `k = k0 + step*q`, q in 0..count.
fn direct_strided(x: &[f64], h: &[f64], k0: usize, step: usize, count: usize) -> Vec<f64> {
    let (n, m) = (x.len(), h.len());
    let hr: Vec<f64> = h.iter().rev().copied().collect();
    let mut out = vec![0.0; count];
    out.par_chunks_mut(PAR_BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            for (q, o) in chunk.iter_mut().enumerate() {
                let k = k0 + step * (b * PAR_BLOCK + q);
                *o = match tap_ranges(k, n, m) {
                    Some((a, j, len)) => dot(&hr[a..a + len], &x[j..j + len]),
                    None => 0.0,
                };
            }
        });
    out
}

/// Complex (SoA) version of [`direct_strided`].
fn direct_strided_c(
    xr: &[f64],
    xi: &[f64],
    h: &[f64],
    k0: usize,
    step: usize,
    count: usize,
) -> Vec<Complex64> {
    let (n, m) = (xr.len(), h.len());
    let hr: Vec<f64> = h.iter().rev().copied().collect();
    let mut out = vec![Complex64::new(0.0, 0.0); count];
    out.par_chunks_mut(PAR_BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            for (q, o) in chunk.iter_mut().enumerate() {
                let k = k0 + step * (b * PAR_BLOCK + q);
                if let Some((a, j, len)) = tap_ranges(k, n, m) {
                    let (r, i) = dot2(&hr[a..a + len], &xr[j..j + len], &xi[j..j + len]);
                    *o = Complex64::new(r, i);
                }
            }
        });
    out
}

// ---------------------------------------------------------------- FFT engine ---------

struct OlsPlan {
    nfft: usize,
    m: usize,
    fwd: Arc<dyn RealToComplex<f64>>,
    inv: Arc<dyn ComplexToReal<f64>>,
    hspec: Vec<Complex64>,
}

impl OlsPlan {
    fn new(h: &[f64], nout: usize) -> Self {
        let m = h.len();
        let target = (8 * m).max(1024).next_power_of_two();
        let nfft = target
            .min((nout + m - 1).next_power_of_two())
            .max((2 * m).next_power_of_two());
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(nfft);
        let inv = planner.plan_fft_inverse(nfft);
        let mut buf = vec![0.0; nfft];
        buf[..m].copy_from_slice(h);
        let mut hspec = fwd.make_output_vec();
        fwd.process(&mut buf, &mut hspec).expect("fft length");
        // fold the 1/N normalisation into the kernel spectrum
        let s = 1.0 / nfft as f64;
        for c in &mut hspec {
            *c *= s;
        }
        Self {
            nfft,
            m,
            fwd,
            inv,
            hspec,
        }
    }

    /// Outputs per block.
    fn block(&self) -> usize {
        self.nfft - (self.m - 1)
    }
}

struct OlsScratch {
    seg: Vec<f64>,
    spec: Vec<Complex64>,
    out: Vec<f64>,
    sf: Vec<Complex64>,
    si: Vec<Complex64>,
}

impl OlsScratch {
    fn new(p: &OlsPlan) -> Self {
        Self {
            seg: vec![0.0; p.nfft],
            spec: p.fwd.make_output_vec(),
            out: vec![0.0; p.nfft],
            sf: p.fwd.make_scratch_vec(),
            si: p.inv.make_scratch_vec(),
        }
    }
}

/// One overlap-save block: full-convolution outputs `s .. s + dst.len()`.
fn ols_block(p: &OlsPlan, x: &[f64], s: usize, dst: &mut [f64], w: &mut OlsScratch) {
    let n = x.len() as isize;
    let start = s as isize - (p.m as isize - 1);
    for (q, v) in w.seg.iter_mut().enumerate() {
        let j = start + q as isize;
        *v = if (0..n).contains(&j) {
            x[j as usize]
        } else {
            0.0
        };
    }
    p.fwd
        .process_with_scratch(&mut w.seg, &mut w.spec, &mut w.sf)
        .expect("fft length");
    for (a, b) in w.spec.iter_mut().zip(&p.hspec) {
        *a *= *b;
    }
    // exact-zero imaginary parts at DC / Nyquist (realfft checks them)
    w.spec[0].im = 0.0;
    if p.nfft.is_multiple_of(2) {
        let l = w.spec.len() - 1;
        w.spec[l].im = 0.0;
    }
    p.inv
        .process_with_scratch(&mut w.spec, &mut w.out, &mut w.si)
        .expect("fft length");
    let l = dst.len();
    dst.copy_from_slice(&w.out[p.m - 1..p.m - 1 + l]);
}

fn ols_range(x: &[f64], h: &[f64], k0: usize, k1: usize) -> Vec<f64> {
    let nout = k1 - k0;
    let plan = OlsPlan::new(h, nout);
    let l = plan.block();
    let mut out = vec![0.0; nout];
    out.par_chunks_mut(l).enumerate().for_each_init(
        || OlsScratch::new(&plan),
        |w, (b, dst)| ols_block(&plan, x, k0 + b * l, dst, w),
    );
    out
}

/// Full-convolution outputs `k0..k1` of real `x` with real `h` (`x`, `h` non-empty).
fn conv_range(x: &[f64], h: &[f64], k0: usize, k1: usize) -> Vec<f64> {
    let m = x.len().min(h.len());
    if m <= 48 {
        if h.len() <= x.len() {
            direct_strided(x, h, k0, 1, k1 - k0)
        } else {
            direct_strided(h, x, k0, 1, k1 - k0)
        }
    } else if h.len() <= x.len() {
        ols_range(x, h, k0, k1)
    } else {
        ols_range(h, x, k0, k1)
    }
}

fn split(x: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
    (
        x.iter().map(|c| c.re).collect(),
        x.iter().map(|c| c.im).collect(),
    )
}

/// `scipy.signal.fftconvolve(a, v, mode)` (1-D, real). Empty input → empty output.
pub fn fftconvolve(a: &[f64], v: &[f64], mode: ConvMode) -> Vec<f64> {
    if a.is_empty() || v.is_empty() {
        return vec![];
    }
    let (k0, k1) = mode_range(a.len(), v.len(), mode);
    conv_range(a, v, k0, k1)
}

/// `fftconvolve` of a complex signal with a real kernel (two real passes, SoA).
pub fn fftconvolve_c(a: &[Complex64], v: &[f64], mode: ConvMode) -> Vec<Complex64> {
    if a.is_empty() || v.is_empty() {
        return vec![];
    }
    let (k0, k1) = mode_range(a.len(), v.len(), mode);
    let (ar, ai) = split(a);
    let (yr, yi) = (conv_range(&ar, v, k0, k1), conv_range(&ai, v, k0, k1));
    yr.into_iter()
        .zip(yi)
        .map(|(r, i)| Complex64::new(r, i))
        .collect()
}

/// `scipy.signal.oaconvolve` — same result as [`fftconvolve`] (to rounding).
pub fn oaconvolve(a: &[f64], v: &[f64], mode: ConvMode) -> Vec<f64> {
    fftconvolve(a, v, mode)
}

/// `scipy.signal.oaconvolve` of a complex signal with a real kernel.
pub fn oaconvolve_c(a: &[Complex64], v: &[f64], mode: ConvMode) -> Vec<Complex64> {
    fftconvolve_c(a, v, mode)
}

/// `fftconvolve(data, ker[:, None], mode, axes=0)` for a row-major `nrows × ncols` matrix:
/// every column convolved with `ker`. Output is row-major `nout × ncols`.
pub fn fftconvolve_axis0(
    data: &[f64],
    nrows: usize,
    ncols: usize,
    ker: &[f64],
    mode: ConvMode,
) -> Result<(Vec<f64>, usize), ValueError> {
    if data.len() != nrows * ncols {
        return verr("data length != nrows * ncols");
    }
    if nrows == 0 || ncols == 0 || ker.is_empty() {
        return Ok((vec![], 0));
    }
    let (k0, k1) = mode_range(nrows, ker.len(), mode);
    let nout = k1 - k0;
    let cols: Vec<Vec<f64>> = (0..ncols)
        .into_par_iter()
        .map(|c| {
            let col: Vec<f64> = (0..nrows).map(|r| data[r * ncols + c]).collect();
            // columns are short: evaluate serially inside the parallel column loop
            conv_range_serial(&col, ker, k0, k1)
        })
        .collect();
    let mut out = vec![0.0; nout * ncols];
    for (c, col) in cols.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            out[r * ncols + c] = v;
        }
    }
    Ok((out, nout))
}

/// Serial variant of [`conv_range`] (no nested parallelism).
fn conv_range_serial(x: &[f64], h: &[f64], k0: usize, k1: usize) -> Vec<f64> {
    let (x, h) = if h.len() <= x.len() { (x, h) } else { (h, x) };
    let (n, m) = (x.len(), h.len());
    if m <= 48 {
        let hr: Vec<f64> = h.iter().rev().copied().collect();
        (k0..k1)
            .map(|k| match tap_ranges(k, n, m) {
                Some((a, j, len)) => dot(&hr[a..a + len], &x[j..j + len]),
                None => 0.0,
            })
            .collect()
    } else {
        let plan = OlsPlan::new(h, k1 - k0);
        let l = plan.block();
        let mut w = OlsScratch::new(&plan);
        let mut out = vec![0.0; k1 - k0];
        for (b, dst) in out.chunks_mut(l).enumerate() {
            ols_block(&plan, x, k0 + b * l, dst, &mut w);
        }
        out
    }
}

// ---------------------------------------------------------------- upfirdn / resample -

fn upfirdn_len(nx: usize, nh: usize, up: usize, down: usize) -> usize {
    ((nx - 1) * up + nh - 1) / down + 1
}

/// `scipy.signal.upfirdn(h, x, up, down)` (mode 'constant', cval 0), real.
pub fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Result<Vec<f64>, ValueError> {
    if up < 1 || down < 1 {
        return verr("Both up and down must be >= 1");
    }
    if h.is_empty() {
        return verr("h must be 1-D with non-zero length");
    }
    if x.is_empty() {
        return Ok(vec![]);
    }
    let ny = upfirdn_len(x.len(), h.len(), up, down);
    if up == 1 {
        return Ok(direct_strided(x, h, 0, down, ny));
    }
    Ok((0..ny).map(|j| upfirdn_one(h, x, up, j * down)).collect())
}

/// `upfirdn` of a complex signal with a real filter.
pub fn upfirdn_c(
    h: &[f64],
    x: &[Complex64],
    up: usize,
    down: usize,
) -> Result<Vec<Complex64>, ValueError> {
    let (xr, xi) = split(x);
    let yr = upfirdn(h, &xr, up, down)?;
    let yi = upfirdn(h, &xi, up, down)?;
    Ok(yr
        .into_iter()
        .zip(yi)
        .map(|(r, i)| Complex64::new(r, i))
        .collect())
}

/// Output at upsampled index `q`: Σ_{m ≡ q (mod up)} h[m]·x[(q − m)/up].
fn upfirdn_one(h: &[f64], x: &[f64], up: usize, q: usize) -> f64 {
    let mut s = 0.0;
    let mut m = q % up;
    while m < h.len() && m <= q {
        let j = (q - m) / up;
        if j < x.len() {
            s += h[m] * x[j];
        }
        m += up;
    }
    s
}

/// Number of `resample_poly(x, 1, down)` outputs: `ceil(n / down)`.
pub fn resample_poly_len(n: usize, down: usize) -> usize {
    n.div_ceil(down)
}

/// `scipy.signal.resample_poly(x, up, down, window=h)` in general form (real, padtype
/// 'constant'): `up, down` reduced by their gcd (`up == down == 1` → copy), `h *= up`,
/// `n_pre_pad = down - half % down`, post-padding until enough outputs exist, output
/// `upfirdn(h, x, up, down)[n_pre_remove .. n_pre_remove + ceil(n·up/down)]`.
/// `window = None` designs scipy's default `firwin(2·10·max(up,down)+1, 1/max, kaiser 5.0)`.
pub fn resample_poly(
    x: &[f64],
    up: usize,
    down: usize,
    window: Option<&[f64]>,
) -> Result<Vec<f64>, ValueError> {
    if up < 1 || down < 1 {
        return verr("up and down must be >= 1");
    }
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        return Ok(x.to_vec());
    }
    let n_in = x.len();
    let n_out = (n_in * up).div_ceil(down);
    let mut h = match window {
        Some(w) => {
            if w.is_empty() {
                return verr("window must be non-empty");
            }
            w.to_vec()
        }
        None => {
            let m = up.max(down);
            crate::fir::firwin_kaiser(2 * 10 * m + 1, 1.0 / m as f64, 5.0, 2.0)?
        }
    };
    let half = (h.len() - 1) / 2;
    for v in &mut h {
        *v *= up as f64;
    }
    if n_in == 0 {
        return Ok(vec![]);
    }
    let n_pre_pad = down - half % down;
    let n_pre_remove = (half + n_pre_pad) / down;
    let mut n_post_pad = 0;
    while upfirdn_len(n_in, h.len() + n_pre_pad + n_post_pad, up, down) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut hp = vec![0.0; n_pre_pad];
    hp.extend_from_slice(&h);
    hp.resize(hp.len() + n_post_pad, 0.0);
    let y = upfirdn(&hp, x, up, down)?;
    Ok(y[n_pre_remove..n_pre_remove + n_out].to_vec())
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Window spec of scipy's default `resample_poly` filter for (up = 1, down):
/// `firwin(numtaps = 2·10·down + 1, cutoff = 1/down, window=('kaiser', 5.0))`.
/// Generate `h` with `fir::firwin` and pass it to [`resample_poly_down`].
pub fn resample_poly_default_spec(down: usize) -> (usize, f64, f64) {
    (2 * 10 * down + 1, 1.0 / down as f64, 5.0)
}

fn check_resample(down: usize, h: &[f64]) -> Result<(), ValueError> {
    if down < 1 {
        return verr("up and down must be >= 1");
    }
    if h.is_empty() {
        return verr("window must be non-empty");
    }
    Ok(())
}

/// `scipy.signal.resample_poly(x, 1, down, window=h)` with an array window (real).
///
/// `y[k] = Σ_i h[i]·x[k·down + half − i]`, `half = (len(h) − 1) // 2`, zeros outside,
/// `len(y) = ceil(n / down)`. This is exactly scipy's `upfirdn` with `n_pre_pad =
/// down − half % down` zeros (= down when `half % down == 0`) and `n_pre_remove =
/// (half + n_pre_pad) // down` dropped outputs. `down == 1` returns a copy of `x`
/// unfiltered (scipy short-circuits `up == down == 1`).
pub fn resample_poly_down(x: &[f64], down: usize, h: &[f64]) -> Result<Vec<f64>, ValueError> {
    check_resample(down, h)?;
    if down == 1 || x.is_empty() {
        return Ok(x.to_vec());
    }
    let half = (h.len() - 1) / 2;
    Ok(direct_strided(
        x,
        h,
        half,
        down,
        resample_poly_len(x.len(), down),
    ))
}

/// Complex `resample_poly(x, 1, down, window=h)` with a real window (SoA kernel).
pub fn resample_poly_down_c(
    x: &[Complex64],
    down: usize,
    h: &[f64],
) -> Result<Vec<Complex64>, ValueError> {
    check_resample(down, h)?;
    if down == 1 || x.is_empty() {
        return Ok(x.to_vec());
    }
    let half = (h.len() - 1) / 2;
    let (xr, xi) = split(x);
    Ok(direct_strided_c(
        &xr,
        &xi,
        h,
        half,
        down,
        resample_poly_len(x.len(), down),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_full(x: &[f64], h: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; x.len() + h.len() - 1];
        for (i, &a) in x.iter().enumerate() {
            for (j, &b) in h.iter().enumerate() {
                y[i + j] += a * b;
            }
        }
        y
    }

    #[test]
    fn ols_matches_naive() {
        let x: Vec<f64> = (0..5000)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) / 7.0)
            .collect();
        let h: Vec<f64> = (0..300)
            .map(|i| ((i * 13 % 29) as f64 - 14.0) / 3.0)
            .collect();
        let want = naive_full(&x, &h);
        let got = fftconvolve(&x, &h, ConvMode::Full);
        let scale = want.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-12 * scale);
        }
        let got = conv_range_serial(&x, &h, 0, want.len());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-12 * scale);
        }
    }

    #[test]
    fn upfirdn_up2() {
        let h = [1.0, 2.0, 3.0];
        let x = [1.0, 1.0];
        // xu = [1,0,1]; full conv = [1,2,4,2,3]
        assert_eq!(
            upfirdn(&h, &x, 2, 1).unwrap(),
            vec![1.0, 2.0, 4.0, 2.0, 3.0]
        );
        assert_eq!(upfirdn(&h, &x, 2, 2).unwrap(), vec![1.0, 4.0, 3.0]);
    }
}
