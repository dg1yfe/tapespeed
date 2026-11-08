//! FIR design and filtering with NumPy/SciPy-compatible definitions:
//! `np.i0` (Cephes Chebyshev), `np.sinc`, symmetric Kaiser window, `kaiserord`,
//! `firwin(..., window=("kaiser", beta), fs)` lowpass, and an overlap-save FIR
//! (`fftconvolve(x, h, mode="same")` alignment) usable as a stream or on a whole array.
//!
//! Shared helpers come from their canonical modules and are re-exported here for the
//! existing call sites: `ValueError` (crate root), `np_sum` (stats), `i0` (window).
//!
//! `OverlapSave` / [`fftconvolve_same`] intentionally stay separate from
//! `conv::fftconvolve`: the streaming correct chain needs stream ≡ whole-array bit for bit,
//! which fixes the block size (FFT ≥ 16384) and serial block order here, whereas
//! `conv::fftconvolve` picks direct/FFT and block sizes per call and runs blocks in
//! parallel. Both agree with scipy to ~1e-16 relative but not bitwise with each other, so
//! implementing one on top of the other would change either the correct step's reference
//! results or `conv`'s speed/parallelism.

use std::f64::consts::PI;
use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex64;

pub use crate::ValueError;
pub use crate::stats::np_sum;
pub(crate) use crate::verr as err;
pub use crate::window::i0;

/// `np.sinc(x)` = sin(pi x)/(pi x); x == 0 is replaced by eps as numpy does.
pub fn sinc(x: f64) -> f64 {
    let y = PI * x;
    let y = if y == 0.0 { f64::EPSILON } else { y };
    y.sin() / y
}

/// `scipy.signal.windows.kaiser(m, beta, sym=True)` (= `window::kaiser(m, beta, true)`).
pub fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    crate::window::kaiser(m, beta, true)
}

/// `scipy.signal.kaiser_beta`.
pub fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a > 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

/// `scipy.signal.kaiser_atten`.
pub fn kaiser_atten(numtaps: usize, width: f64) -> f64 {
    2.285 * (numtaps as f64 - 1.0) * PI * width + 7.95
}

/// `scipy.signal.kaiserord(ripple, width)` -> (numtaps, beta); width relative to Nyquist.
pub fn kaiserord(ripple: f64, width: f64) -> Result<(usize, f64), ValueError> {
    let a = ripple.abs();
    if a < 8.0 {
        return err(format!(
            "Requested maximum ripple attenuation {a:.6} is too small for the Kaiser formula."
        ));
    }
    let beta = kaiser_beta(a);
    let numtaps = (a - 7.95) / 2.285 / (PI * width) + 1.0;
    Ok((numtaps.ceil() as usize, beta))
}

/// `scipy.signal.firwin(numtaps, cutoff, window=("kaiser", beta), fs=fs)` (lowpass, scale=True).
pub fn firwin_kaiser(
    numtaps: usize,
    cutoff: f64,
    beta: f64,
    fs: f64,
) -> Result<Vec<f64>, ValueError> {
    let nyq = 0.5 * fs;
    let c = cutoff / nyq;
    if c <= 0.0 || c >= 1.0 {
        return err(
            "Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.",
        );
    }
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let win = kaiser(numtaps, beta);
    let mut h: Vec<f64> = (0..numtaps)
        .map(|i| {
            let m = i as f64 - alpha;
            // h = 0 + c*sinc(c*m) - 0*sinc(0*m), then *= win
            let v = c * sinc(c * m);
            (v - 0.0 * sinc(0.0 * m)) * win[i]
        })
        .collect();
    // scale_frequency 0 -> cos(0) = 1 -> s = sum(h)
    let s = np_sum(&h);
    for v in &mut h {
        *v /= s;
    }
    Ok(h)
}

/// FIR filter with `fftconvolve(x, h, mode="same")` alignment via overlap-save (f64),
/// fed in arbitrary pieces; outputs are emitted in order as they become final.
pub struct OverlapSave {
    l1: usize,
    hop: usize,
    spec: Vec<Complex64>,
    fwd: Arc<dyn RealToComplex<f64>>,
    inv: Arc<dyn ComplexToReal<f64>>,
    /// input queue; the first `l1` entries are history (zeros at start)
    inbuf: Vec<f64>,
    /// 'same' offset (h.len()-1)//2: leading 'full' outputs dropped
    delay: usize,
    n_in: usize,
    /// 'full' outputs computed so far
    n_full: usize,
    tbuf: Vec<f64>,
    fbuf: Vec<Complex64>,
    scratch_f: Vec<Complex64>,
    scratch_i: Vec<Complex64>,
}

impl OverlapSave {
    /// `h`: impulse response (non-empty). FFT size: power of two >= max(16384, 4 len(h)).
    pub fn new(h: &[f64]) -> Self {
        assert!(!h.is_empty(), "empty FIR");
        let l = h.len();
        let n_fft = (4 * l).next_power_of_two().max(16384);
        let mut planner = RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(n_fft);
        let inv = planner.plan_fft_inverse(n_fft);
        let mut tb = fwd.make_input_vec();
        tb[..l].copy_from_slice(h);
        let mut spec = fwd.make_output_vec();
        let mut sc = fwd.make_scratch_vec();
        fwd.process_with_scratch(&mut tb, &mut spec, &mut sc)
            .expect("fft");
        let norm = 1.0 / n_fft as f64;
        for v in &mut spec {
            *v *= norm;
        }
        Self {
            l1: l - 1,
            hop: n_fft - (l - 1),
            spec,
            scratch_f: fwd.make_scratch_vec(),
            scratch_i: inv.make_scratch_vec(),
            tbuf: fwd.make_input_vec(),
            fbuf: fwd.make_output_vec(),
            fwd,
            inv,
            inbuf: vec![0.0; l - 1],
            delay: (l - 1) / 2,
            n_in: 0,
            n_full: 0,
        }
    }

    /// One block over `inbuf` (exactly l1 + hop samples): emits `valid` full outputs.
    fn block(&mut self, out: &mut Vec<f64>, valid: usize) {
        self.tbuf.copy_from_slice(&self.inbuf);
        self.fwd
            .process_with_scratch(&mut self.tbuf, &mut self.fbuf, &mut self.scratch_f)
            .expect("fft");
        for (a, b) in self.fbuf.iter_mut().zip(&self.spec) {
            *a *= *b;
        }
        self.fbuf[0].im = 0.0;
        let last = self.fbuf.len() - 1;
        self.fbuf[last].im = 0.0;
        self.inv
            .process_with_scratch(&mut self.fbuf, &mut self.tbuf, &mut self.scratch_i)
            .expect("ifft");
        let full = &self.tbuf[self.l1..self.l1 + valid];
        let s = self.delay.saturating_sub(self.n_full).min(valid);
        out.extend_from_slice(&full[s..]);
        self.n_full += valid;
        self.inbuf.drain(..self.hop);
    }

    /// Push input samples; appends every output sample that became final.
    pub fn push(&mut self, x: &[f64], out: &mut Vec<f64>) {
        self.n_in += x.len();
        let mut rest = x;
        while !rest.is_empty() {
            let need = self.l1 + self.hop - self.inbuf.len();
            let k = need.min(rest.len());
            self.inbuf.extend_from_slice(&rest[..k]);
            rest = &rest[k..];
            if k == need {
                self.block(out, self.hop);
            }
        }
    }

    /// Flush with zeros: afterwards the total output length equals the total input length.
    pub fn finish(&mut self, out: &mut Vec<f64>) {
        let target = self.n_in + self.delay;
        while self.n_in > 0 && self.n_full < target {
            self.inbuf.resize(self.l1 + self.hop, 0.0);
            let valid = (target - self.n_full).min(self.hop);
            self.block(out, valid);
        }
    }

    /// Group delay removed by the 'same' alignment.
    pub fn delay(&self) -> usize {
        self.delay
    }
}

/// Whole-array `fftconvolve(x, h, mode="same")` (overlap-save; same numerics as the stream).
pub fn fftconvolve_same(x: &[f64], h: &[f64]) -> Vec<f64> {
    let mut os = OverlapSave::new(h);
    let mut out = Vec::with_capacity(x.len());
    os.push(x, &mut out);
    os.finish(&mut out);
    out
}

/// Direct-form 'same' convolution (reference for tests; O(n·L)).
pub fn convolve_same_direct(x: &[f64], h: &[f64]) -> Vec<f64> {
    let n = x.len() as isize;
    let d = ((h.len() - 1) / 2) as isize;
    (0..n)
        .map(|i| {
            let mut acc = 0.0;
            for (j, &hj) in h.iter().enumerate() {
                let k = i + d - j as isize;
                if k >= 0 && k < n {
                    acc += hj * x[k as usize];
                }
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_save_matches_direct_any_split() {
        for hl in [1usize, 2, 37, 38] {
            let h: Vec<f64> = (0..hl)
                .map(|i| ((i * 7 % 11) as f64 - 5.0) / 13.0)
                .collect();
            for n in [0usize, 1, 5, 36, 37, 100, 20000, 40001] {
                let x: Vec<f64> = (0..n)
                    .map(|i| ((i * 31 % 97) as f64 / 97.0) - 0.5)
                    .collect();
                let r = convolve_same_direct(&x, &h);
                let y = fftconvolve_same(&x, &h);
                assert_eq!(y.len(), n);
                let e = r
                    .iter()
                    .zip(&y)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(e < 1e-12, "hl={hl} n={n} e={e}");
                let mut os = OverlapSave::new(&h);
                let mut z = Vec::new();
                for c in x.chunks(777) {
                    os.push(c, &mut z);
                }
                os.finish(&mut z);
                assert_eq!(z, y);
            }
        }
    }

    #[test]
    fn np_sum_small() {
        assert_eq!(np_sum(&[1.0, 2.0, 3.0]), 6.0);
        let v: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        assert_eq!(np_sum(&v), 499500.0);
    }
}
