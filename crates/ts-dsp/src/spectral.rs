//! Spectral estimation: `scipy.signal.welch`, `scipy.signal.stft`
//! (`boundary=None, padded=False`), batched frame FFTs, `np.fft.fftfreq/rfftfreq/fftshift`.
//!
//! All frames of one call go through a single realfft plan; frames are processed in
//! parallel (rayon) in fixed chunks, each frame independently, and Welch's frame average is
//! merged in chunk order → bit-identical results for any thread count.
//!
//! Frame grid as scipy: frame `p` covers `x[p*hop .. p*hop + nperseg]`,
//! `nframes = (n - noverlap) // hop` (= `(n - nperseg) // hop + 1`).

use num_complex::Complex64;
use rayon::prelude::*;
use realfft::RealFftPlanner;

use crate::stats::{mean, np_sum};
use crate::window::hann_periodic;
use crate::{ValueError, verr};

/// `np.fft.fftfreq(n, d)`: `[0, 1, …, (n-1)//2, -(n//2), …, -1] * (1 / (n*d))`.
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    let npos = (n - 1) / 2 + 1;
    (0..n)
        .map(|i| {
            let k = if i < npos {
                i as f64
            } else {
                i as f64 - n as f64
            };
            k * val
        })
        .collect()
}

/// `np.fft.rfftfreq(n, d)`: `[0, 1, …, n//2] * (1 / (n*d))`.
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    (0..n / 2 + 1).map(|i| i as f64 * val).collect()
}

/// `np.fft.fftshift` (1-D): rotate right by `n // 2`.
pub fn fftshift<T: Clone>(x: &[T]) -> Vec<T> {
    let n = x.len();
    let s = n / 2;
    x[n - s..].iter().chain(&x[..n - s]).cloned().collect()
}

/// `scipy.signal.hilbert(x)`: analytic signal via FFT, `h = [1, 2, …, 2, (1), 0, …]`.
pub fn hilbert(x: &[f64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let mut v: Vec<Complex64> = x.iter().map(|&r| Complex64::new(r, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut v);
    let half = if n.is_multiple_of(2) {
        n / 2
    } else {
        n.div_ceil(2)
    };
    for (k, c) in v.iter_mut().enumerate() {
        let h = if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
            1.0
        } else if k < half {
            2.0
        } else {
            0.0
        };
        *c *= h;
    }
    planner.plan_fft_inverse(n).process(&mut v);
    let s = 1.0 / n as f64;
    v.iter().map(|c| c * s).collect()
}

/// Number of frames `x[p*hop .. p*hop+nper]` that fit in `n` samples.
pub fn n_frames(n: usize, nper: usize, hop: usize) -> usize {
    if n < nper || nper == 0 || hop == 0 {
        0
    } else {
        (n - nper) / hop + 1
    }
}

const FRAME_CHUNK: usize = 16;

/// Batched frame spectra: for each frame `p`, `prep(frame, buf)` fills the first `nper`
/// samples of the (zeroed, length-`nfft`) FFT buffer, then `rfft(buf)` is written to row `p`
/// of the frame-major output (`nframes × (nfft/2+1)`).
fn frames_rfft_with<F>(x: &[f64], nper: usize, hop: usize, nfft: usize, prep: F) -> Vec<Complex64>
where
    F: Fn(&[f64], &mut [f64]) + Sync,
{
    let nf = n_frames(x.len(), nper, hop);
    let nbin = nfft / 2 + 1;
    let mut out = vec![Complex64::new(0.0, 0.0); nf * nbin];
    if nf == 0 {
        return out;
    }
    let plan = RealFftPlanner::<f64>::new().plan_fft_forward(nfft);
    out.par_chunks_mut(nbin * FRAME_CHUNK)
        .enumerate()
        .for_each_init(
            || (vec![0.0; nfft], plan.make_scratch_vec()),
            |(buf, scratch), (c, rows)| {
                for (q, row) in rows.chunks_mut(nbin).enumerate() {
                    let s = (c * FRAME_CHUNK + q) * hop;
                    buf.iter_mut().for_each(|v| *v = 0.0);
                    prep(&x[s..s + nper], &mut buf[..nper]);
                    plan.process_with_scratch(buf, row, scratch)
                        .expect("fft length");
                }
            },
        );
    out
}

/// Frame spectra `rfft(frame * win, nfft)` for frames `x[p*hop .. p*hop+len(win)]`
/// (e.g. the survey's frame spectra: every `hop`-th length-`n` frame times `np.hanning(n)`).
/// Returns the frame-major matrix (`nframes × (nfft/2+1)`) and `nframes`.
pub fn rfft_frames(
    x: &[f64],
    win: &[f64],
    hop: usize,
    nfft: usize,
) -> Result<(Vec<Complex64>, usize), ValueError> {
    let nper = win.len();
    if nper == 0 || hop == 0 || nfft < nper {
        return verr("need len(win) >= 1, hop >= 1, nfft >= len(win)");
    }
    let out = frames_rfft_with(x, nper, hop, nfft, |fr, buf| {
        for ((b, &v), &w) in buf.iter_mut().zip(fr).zip(win) {
            *b = v * w;
        }
    });
    Ok((out, n_frames(x.len(), nper, hop)))
}

/// Result of [`stft`] / [`stft_frame_major`]: frequencies, frame times and `Z`.
/// Layout: scipy's row-major `nfreq × nframes` (`z[k * nframes + t]`) from [`stft`];
/// frame-major `nframes × nfreq` (`z[t * nfreq + k]`) from [`stft_frame_major`].
#[derive(Debug, Clone)]
pub struct Stft {
    pub f: Vec<f64>,
    pub t: Vec<f64>,
    pub z: Vec<Complex64>,
    pub nfreq: usize,
    pub nframes: usize,
    pub frame_major: bool,
}

impl Stft {
    /// `Z[k, t]` (scipy indexing) regardless of layout.
    #[inline]
    pub fn at(&self, k: usize, t: usize) -> Complex64 {
        if self.frame_major {
            self.z[t * self.nfreq + k]
        } else {
            self.z[k * self.nframes + t]
        }
    }
}

/// `scipy.signal.stft(x, fs, nperseg=nperseg, noverlap=noverlap, boundary=None,
/// padded=False)` with the default periodic Hann window, `nfft = nperseg`, no detrend,
/// one-sided, scaling 'spectrum': `Z = rfft(win * frame) * sqrt(1 / win.sum()**2)`.
///
/// `nperseg > len(x)` is clipped to `len(x)` (scipy warns); `noverlap` (default
/// `nperseg // 2`) `>= nperseg` → error, as scipy (this happens for a short tail chunk).
/// Input is f64. SciPy computes float32 input in complex64; here f32 samples are widened to
/// f64 instead (≈1e-8 rel. difference to SciPy's float32 path).
pub fn stft(
    x: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
) -> Result<Stft, ValueError> {
    let mut s = stft_frame_major(x, fs, nperseg, noverlap)?;
    s.z = transpose(&s.z, s.nframes, s.nfreq);
    s.frame_major = false;
    Ok(s)
}

/// [`stft`] without the final transpose: `z` is frame-major (`nframes × nfreq`), each
/// frame's spectrum contiguous. Same values.
pub fn stft_frame_major(
    x: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
) -> Result<Stft, ValueError> {
    if nperseg < 1 {
        return verr("nperseg must be a positive integer");
    }
    let n = x.len();
    if n == 0 {
        return Ok(Stft {
            f: vec![],
            t: vec![],
            z: vec![],
            nfreq: 0,
            nframes: 0,
            frame_major: true,
        });
    }
    let nperseg = nperseg.min(n);
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return verr("noverlap must be less than nperseg.");
    }
    let hop = nperseg - noverlap;
    let win = hann_periodic(nperseg);
    let ws = np_sum(&win);
    let scale = (1.0 / (ws * ws)).sqrt();
    let nbin = nperseg / 2 + 1;
    let mut z = frames_rfft_with(x, nperseg, hop, nperseg, |fr, buf| {
        for ((b, &v), &w) in buf.iter_mut().zip(fr).zip(&win) {
            *b = w * v;
        }
    });
    z.par_chunks_mut(nbin * FRAME_CHUNK)
        .for_each(|c| c.iter_mut().for_each(|v| *v *= scale));
    let nf = n_frames(n, nperseg, hop);
    let t = (0..nf)
        .map(|i| (nperseg as f64 / 2.0 + i as f64 * hop as f64) / fs)
        .collect();
    Ok(Stft {
        f: rfftfreq(nperseg, 1.0 / fs),
        t,
        z,
        nfreq: nbin,
        nframes: nf,
        frame_major: true,
    })
}

/// Row-major `rows × cols` → `cols × rows`, 32×32 tiles, parallel over output row bands.
fn transpose<T: Copy + Send + Sync + Default>(a: &[T], rows: usize, cols: usize) -> Vec<T> {
    const TB: usize = 32;
    let mut out = vec![T::default(); rows * cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    out.par_chunks_mut(rows * TB)
        .enumerate()
        .for_each(|(band, dst)| {
            let c0 = band * TB;
            let c1 = (c0 + TB).min(cols);
            for r0 in (0..rows).step_by(TB) {
                let r1 = (r0 + TB).min(rows);
                for c in c0..c1 {
                    let drow = &mut dst[(c - c0) * rows..(c - c0 + 1) * rows];
                    for r in r0..r1 {
                        drow[r] = a[r * cols + c];
                    }
                }
            }
        });
    out
}

/// `scipy.signal.welch(x, fs, nperseg=nperseg)` with all other defaults: periodic Hann,
/// `noverlap = nperseg // 2`, `nfft = nperseg`, detrend 'constant' per frame, one-sided,
/// scaling 'density' (`|X|² / (fs·Σw²)`, doubled except DC and — for even nfft — Nyquist),
/// mean over frames. `nperseg > len(x)` is clipped (scipy warns). Empty `x` → empty output.
pub fn welch(x: &[f64], fs: f64, nperseg: usize) -> Result<(Vec<f64>, Vec<f64>), ValueError> {
    if nperseg < 1 {
        return verr("nperseg must be a positive integer");
    }
    let n = x.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    let nperseg = nperseg.min(n);
    let noverlap = nperseg / 2;
    let hop = nperseg - noverlap;
    let nfft = nperseg;
    let win = hann_periodic(nperseg);
    let nbin = nfft / 2 + 1;
    let nf = n_frames(n, nperseg, hop);
    let w2 = np_sum(&win.iter().map(|w| w * w).collect::<Vec<_>>());
    let scale = 1.0 / (fs * w2);

    // per chunk of frames: Σ|X|² (frames in order), chunks merged in order afterwards
    let plan = RealFftPlanner::<f64>::new().plan_fft_forward(nfft);
    let nchunks = nf.div_ceil(FRAME_CHUNK);
    let partial: Vec<Vec<f64>> = (0..nchunks)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![0.0; nfft],
                    plan.make_output_vec(),
                    plan.make_scratch_vec(),
                )
            },
            |(buf, spec, scratch), c| {
                let mut acc = vec![0.0; nbin];
                for p in c * FRAME_CHUNK..((c + 1) * FRAME_CHUNK).min(nf) {
                    let fr = &x[p * hop..p * hop + nperseg];
                    let m = mean(fr);
                    for ((b, &v), &w) in buf.iter_mut().zip(fr).zip(&win) {
                        *b = (v - m) * w;
                    }
                    plan.process_with_scratch(buf, spec, scratch)
                        .expect("fft length");
                    for (a, s) in acc.iter_mut().zip(spec.iter()) {
                        *a += s.norm_sqr();
                    }
                }
                acc
            },
        )
        .collect();
    let mut p = vec![0.0; nbin];
    for part in &partial {
        for (a, b) in p.iter_mut().zip(part) {
            *a += b;
        }
    }
    let hi = if nfft.is_multiple_of(2) {
        nbin - 1
    } else {
        nbin
    };
    let inv_nf = 1.0 / nf as f64;
    for (k, v) in p.iter_mut().enumerate() {
        let d = if k >= 1 && k < hi { 2.0 } else { 1.0 };
        *v = *v * scale * d * inv_nf;
    }
    Ok((rfftfreq(nfft, 1.0 / fs), p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freqs() {
        assert_eq!(fftfreq(4, 0.5), vec![0.0, 0.5, -1.0, -0.5]);
        assert_eq!(fftfreq(5, 1.0), vec![0.0, 0.2, 0.4, -0.4, -0.2]);
        assert_eq!(rfftfreq(5, 1.0), vec![0.0, 0.2, 0.4]);
        assert_eq!(fftshift(&[0, 1, 2, -2, -1]), vec![-2, -1, 0, 1, 2]);
        assert_eq!(fftshift(&[0, 1, -2, -1]), vec![-2, -1, 0, 1]);
    }

    #[test]
    fn stft_noverlap_error() {
        let x = vec![0.0; 2000];
        assert!(stft(&x, 1.0, 4096, Some(2048)).is_err());
        let s = stft(&vec![1.0; 3000], 1.0, 4096, Some(2048)).unwrap();
        assert_eq!((s.nfreq, s.nframes), (1501, 1));
        let x: Vec<f64> = (0..5000).map(|i| ((i * 7) % 13) as f64).collect();
        let (a, b) = (
            stft(&x, 1.0, 256, None).unwrap(),
            stft_frame_major(&x, 1.0, 256, None).unwrap(),
        );
        for k in 0..a.nfreq {
            for t in 0..a.nframes {
                assert_eq!(a.at(k, t), b.at(k, t));
            }
        }
    }
}
