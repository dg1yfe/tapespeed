//! Stage 2: coarse track by Viterbi over a 0.25 s spectrogram.
//!
//! Spectrogram rows are independent and computed in parallel (rayon, fixed row blocks); the
//! Viterbi recursion and back-tracking are sequential, in time order, with a
//! first-occurrence `argmax`. The spectrogram is recomputed for the second pass
//! (peak interpolation) instead of being held in memory (1 h ≈ 0.5 GB).

use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use ts_dsp::ValueError;
use ts_dsp::ndimage::{Mode, medfilt, uniform_filter1d};
use ts_dsp::spectral::fftfreq;
use ts_dsp::stats::median;
use ts_dsp::window::hanning;

/// Keyword parameters of [`coarse`] (defaults via `Default`).
#[derive(Debug, Clone, Copy)]
pub struct CoarseParams {
    /// block length [s]
    pub tb: f64,
    /// hop [s]
    pub hop: f64,
    /// max slew as a fraction of `ref_hz` per s
    pub slew: f64,
    /// path cost per Hz moved
    pub lam: f64,
    /// soft cap of the per-bin score
    pub cap: f64,
}

impl Default for CoarseParams {
    fn default() -> Self {
        CoarseParams {
            tb: 0.25,
            hop: 0.05,
            slew: 0.01,
            lam: 0.3,
            cap: 12.0,
        }
    }
}

/// `(tc, f, score)` of [`coarse`], plus the raw Viterbi path (bin indices into the search
/// window) for regression checks.
#[derive(Debug, Clone, Default)]
pub struct Coarse {
    /// block-centre times [s]
    pub tc: Vec<f64>,
    /// coarse track [Hz offset] (after medfilt 5 + uniform filter)
    pub f: Vec<f64>,
    /// path score [dB above row median]
    pub score: Vec<f64>,
    /// Viterbi path (index into the selected bins `|fr| <= fmax`)
    pub path: Vec<usize>,
}

/// numpy's complex `absolute` (SIMD loop): `L·sqrt(fma(s/L, s/L, 1))`, `L = max(|re|,|im|)`.
#[inline]
pub fn np_cabs(c: Complex64) -> f64 {
    let (a, b) = (c.re.abs(), c.im.abs());
    if a.is_nan() || b.is_nan() {
        return if a.is_infinite() || b.is_infinite() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    let (l, s) = if a >= b { (a, b) } else { (b, a) };
    if l == 0.0 {
        return 0.0;
    }
    let r = s / l;
    r.mul_add(r, 1.0).sqrt() * l
}

/// numpy `tanh` stand-in (libm). numpy 2.x uses its own SIMD tanh on aarch64 that differs
/// from libm in the last bit for ~20 % of arguments; the Viterbi path is unaffected on all
/// reference vectors (scores are soft-capped, ties are not decided at the ulp level).
#[inline]
fn np_tanh(x: f64) -> f64 {
    x.tanh()
}

struct Spec {
    nb: usize,
    nfft: usize,
    /// unshifted FFT bin of each selected (shifted) bin
    bins: Vec<usize>,
    w: Vec<f64>,
    fft: Arc<dyn Fft<f64>>,
}

impl Spec {
    /// `|fftshift(fft(z[s:s+nb]*w, nfft))[sel]|**2` into `out`.
    fn row(
        &self,
        z: &[Complex64],
        s: usize,
        buf: &mut Vec<Complex64>,
        scr: &mut Vec<Complex64>,
        out: &mut [f64],
    ) {
        buf.clear();
        buf.extend(
            z[s..s + self.nb]
                .iter()
                .zip(&self.w)
                .map(|(c, &w)| Complex64::new(c.re * w, c.im * w)),
        );
        buf.resize(self.nfft, Complex64::new(0.0, 0.0));
        scr.resize(self.fft.get_inplace_scratch_len(), Complex64::new(0.0, 0.0));
        self.fft.process_with_scratch(buf, scr);
        for (o, &b) in out.iter_mut().zip(&self.bins) {
            let a = np_cabs(buf[b]);
            *o = a * a;
        }
    }

    /// Rows `starts[i0..i1]` → row-major `P` (`(i1-i0) × nsel`), rows in parallel.
    fn block(&self, z: &[Complex64], starts: &[usize]) -> Vec<f64> {
        let nsel = self.bins.len();
        let mut p = vec![0.0; starts.len() * nsel];
        p.par_chunks_mut(nsel).zip(starts.par_iter()).for_each_init(
            || (Vec::new(), Vec::new()),
            |(buf, scr), (out, &s)| self.row(z, s, buf, scr, out),
        );
        p
    }
}

/// Rows per parallel spectrogram block (the Viterbi runs sequentially between blocks).
const ROWS: usize = 2048;

/// `coarse(z, fs, fmax, ref_hz, tb, hop, slew, lam, cap)`.
pub fn coarse(
    z: &[Complex64],
    fs: f64,
    fmax: f64,
    ref_hz: f64,
    p: CoarseParams,
) -> Result<Coarse, ValueError> {
    if z.is_empty() {
        return Err(ValueError("_coarse: empty input".into()));
    }
    let nb = ((p.tb * fs) as usize).min(z.len());
    let nh = ((p.hop * fs) as usize).max(1);
    let nfft = 1usize << ((2 * nb) as f64).log2().ceil() as u32;
    let df = fs / nfft as f64;
    let ff = fftfreq(nfft, 1.0 / fs);
    let half = nfft / 2;
    // fr = fftshift(ff): fr[j] = ff[(j + half) % nfft]
    let sel: Vec<usize> = (0..nfft)
        .filter(|&j| ff[(j + half) % nfft].abs() <= fmax)
        .collect();
    let nsel = sel.len();
    if nsel == 0 {
        return Err(ValueError("_coarse: empty search window".into()));
    }
    let bins: Vec<usize> = sel.iter().map(|&j| (j + half) % nfft).collect();
    let fr_sel: Vec<f64> = bins.iter().map(|&b| ff[b]).collect();
    let starts: Vec<usize> = (0..=z.len().saturating_sub(nb)).step_by(nh).collect();
    let nr = starts.len();
    let kk = ((p.slew * ref_hz * p.hop / df).ceil() as i64).max(1) as usize;
    let pen: Vec<f64> = (0..=2 * kk)
        .map(|d| p.lam * df * (d as i64 - kk as i64).abs() as f64)
        .collect();
    let spec = Spec {
        nb,
        nfft,
        bins,
        w: hanning(nb),
        fft: FftPlanner::<f64>::new().plan_fft_forward(nfft),
    };

    // ---- pass 1: Viterbi
    let mut acc = vec![0.0f64; nsel];
    let mut nxt = vec![0.0f64; nsel];
    let mut back = vec![0u8; nr * nsel];
    let mut padded = vec![-1e30f64; nsel + 2 * kk];
    for i0 in (0..nr).step_by(ROWS) {
        let i1 = (i0 + ROWS).min(nr);
        let pm = spec.block(z, &starts[i0..i1]);
        let sc: Vec<f64> = pm
            .par_chunks(nsel)
            .flat_map_iter(|row| {
                let med = median(row);
                row.iter().map(move |&v| {
                    let db = (10.0 * (v / (med + 1e-30) + 1e-30).log10() - 3.0).max(0.0);
                    p.cap * np_tanh(db / p.cap) + 0.05 * db.min(100.0)
                })
            })
            .collect();
        for (r, srow) in sc.chunks(nsel).enumerate() {
            padded[kk..kk + nsel].copy_from_slice(&acc);
            let brow = &mut back[(i0 + r) * nsel..(i0 + r + 1) * nsel];
            for j in 0..nsel {
                let win = &padded[j..j + 2 * kk + 1];
                let mut best = win[0] - pen[0];
                let mut bk = 0usize;
                for d in 1..=2 * kk {
                    let v = win[d] - pen[d];
                    if v > best {
                        best = v;
                        bk = d;
                    }
                }
                brow[j] = bk as u8;
                nxt[j] = best + srow[j];
            }
            std::mem::swap(&mut acc, &mut nxt);
        }
    }
    let mut path = vec![0usize; nr];
    path[nr - 1] = ts_dsp::stats::argmax(&acc).unwrap_or(0);
    for i in (1..nr).rev() {
        path[i - 1] = (path[i] + back[i * nsel + path[i]] as usize) - kk;
    }

    // ---- pass 2: parabolic peak interpolation on log power, path score
    let mut f = vec![0.0; nr];
    let mut score = vec![0.0; nr];
    for i0 in (0..nr).step_by(ROWS) {
        let i1 = (i0 + ROWS).min(nr);
        let pm = spec.block(z, &starts[i0..i1]);
        let res: Vec<(f64, f64)> = pm
            .par_chunks(nsel)
            .zip(path[i0..i1].par_iter())
            .map(|(row, &pk)| {
                let k = pk.clamp(1, nsel.saturating_sub(2).max(1));
                let (a, b, c) = (
                    (row[k - 1] + 1e-30).ln(),
                    (row[k] + 1e-30).ln(),
                    (row[k + 1] + 1e-30).ln(),
                );
                let den = a - 2.0 * b + c;
                let off = if den < 0.0 { 0.5 * (a - c) / den } else { 0.0 };
                let fv = fr_sel[k] + off.clamp(-1.0, 1.0) * df;
                (fv, 10.0 * (row[k] / median(row)).log10())
            })
            .collect();
        for (q, (fv, sv)) in res.into_iter().enumerate() {
            f[i0 + q] = fv;
            score[i0 + q] = sv;
        }
    }
    let tc: Vec<f64> = starts
        .iter()
        .map(|&s| (s as f64 + nb as f64 / 2.0) / fs)
        .collect();
    let size = ((0.3 / p.hop) as usize).max(1);
    let f = uniform_filter1d(&medfilt(&f, 5)?, size, Mode::Nearest)?;
    Ok(Coarse { tc, f, score, path })
}

/// `fcd = np.interp(np.arange(n) / fs_d, tc, fc)`: coarse track on the stage-1 grid.
pub fn fcd(n: usize, fs_d: f64, tc: &[f64], fc: &[f64]) -> Result<Vec<f64>, ValueError> {
    let td: Vec<f64> = (0..n).map(|i| i as f64 / fs_d).collect();
    super::track::interp_mono(&td, tc, fc)
}
