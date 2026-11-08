//! Model-aided tracking: [`fir0`] (variant with a short-input guard), [`viterbi_track`],
//! [`narrow`], [`refine_amps`], [`track_aided`], `aided_iterate`, [`coarse_track`]. `pick` /
//! `lpf` / `het_decimate` are [`crate::hetdec`]; `integ`, `cdiff_freq` and the element-wise
//! kernels (fused NEON
//! complex multiply, SIMD `|z|`, pairwise complex sum, `exp(jθ)` as separate libm `cos`/`sin`)
//! are shared with [`crate::analyze::track`] (identical definitions).
//!
//! FFTs (rustfft vs pocketfft) and the sparse solve of [`refine_amps`] (faer Cholesky vs
//! SuperLU) differ at rounding level; on all stored reference cases the narrow outputs agree
//! to ~1e-13 relative with identical masks.
//!
//! Parallelism: Viterbi spectrogram frames (independent rows) and inside the primitives
//! (het_decimate, median filter); results do not depend on the thread count.

use std::sync::Arc;

use indexmap::IndexMap;
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use ts_dsp::ValueError;
use ts_dsp::conv::resample_poly_down_c;
use ts_dsp::iir::{PadType, butter_lowpass_sos, sosfiltfilt};
use ts_dsp::linalg::{SymCsc, spd_solve};
use ts_dsp::ndimage::{
    Mode, binary_dilation, binary_erosion, binary_opening, medfilt, median_filter, uniform_filter1d,
};
use ts_dsp::spectral::fftfreq;
use ts_dsp::spline::cubic_spline;
use ts_dsp::stats::{argmax, mean, median};
use ts_dsp::window::hanning;
use ts_io::np_round;

use super::gaps::pchip_fill;
use super::{Amp, DT, FitOpts, HierOpts, Model, fit, fit_hier, hat};
use crate::curve::{Curve, MetaValue};
use crate::hetdec::{Channel, cos_sin, het_decimate, lpf};

type Res<T> = Result<T, ValueError>;

const PI: f64 = std::f64::consts::PI;

// ------------------------------------------------------------------ numpy kernels

pub(crate) use crate::analyze::coarse::np_cabs as cabs;
pub(crate) use crate::analyze::track::np_cmul as cmul;
pub use crate::analyze::track::{cdiff_freq, integ};
use crate::analyze::track::{derot, interp_mono as interp_mono_r, np_csum};

/// `exp(1j·θ)` as numpy computes it for a purely imaginary argument.
#[inline]
fn cis(theta: f64) -> Complex64 {
    let (c, s) = cos_sin(theta);
    Complex64::new(c, s)
}

/// `np.arange(n) / fs`.
fn arange_div(n: usize, fs: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 / fs).collect()
}

/// `np.interp(x, xp, fp)` for monotone `x` (hinted search); `xp` non-empty.
fn interp_mono(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    interp_mono_r(x, xp, fp).expect("interp: xp non-empty, same length as fp")
}

/// Output grid `np.arange(int(np.ceil(n / sr / DT)) + 1) * DT`.
pub fn grid_t(n: usize, sr: f64) -> Vec<f64> {
    let m = (n as f64 / sr / DT).ceil() as usize + 1;
    (0..m).map(|i| i as f64 * DT).collect()
}

// ------------------------------------------------------------------- fir0

/// `fir0(a, h)` (real): [`crate::analyze::track::fir0`] plus a short-input guard
/// (`len(a) <= len(h)//2 + 1` → constant mean).
pub fn fir0(a: &[f64], h: &[f64]) -> Vec<f64> {
    if a.len() <= h.len() / 2 + 1 {
        return vec![mean(a); a.len()];
    }
    crate::analyze::track::fir0(a, h)
}

/// `fir0(a, h, phasor=True)` (same guard, complex mean).
pub fn fir0_phasor(a: &[Complex64], h: &[f64]) -> Vec<Complex64> {
    let n = a.len();
    if n <= h.len() / 2 + 1 {
        let s = np_csum(a);
        return vec![Complex64::new(s.re / n as f64, s.im / n as f64); n];
    }
    crate::analyze::track::fir0_phasor(a, h)
}

// ------------------------------------------------------------------- viterbi_track

/// `viterbi_track(u, fs, fmax, tb, hop, slew, lam, cap)`: coarse residual frequency track
/// `|f| <= fmax` by Viterbi over an STFT of `u`. Returns `(tc, f, score)`.
#[allow(clippy::too_many_arguments)]
pub fn viterbi_track(
    u: &[Complex64],
    fs: f64,
    fmax: f64,
    tb: f64,
    hop: f64,
    slew: f64,
    lam: f64,
    cap: f64,
) -> Res<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if u.is_empty() {
        return Err(ValueError("_viterbi_track: empty input".into()));
    }
    let nb = ((tb * fs) as usize).min(u.len());
    if nb == 0 {
        return Err(ValueError("_viterbi_track: zero-length window".into()));
    }
    let nh = ((hop * fs) as i64).max(1) as usize;
    let nfft = 1usize << ((4 * nb) as f64).log2().ceil() as u32;
    let df = fs / nfft as f64;
    let fr_un = fftfreq(nfft, 1.0 / fs);
    let half = nfft / 2;
    // shifted index s <-> unshifted bin (s + nfft/2) % nfft (nfft even)
    let sel: Vec<usize> = (0..nfft)
        .filter(|&s| fr_un[(s + half) % nfft].abs() <= fmax)
        .collect();
    if sel.is_empty() {
        return Err(ValueError("_viterbi_track: no bins within fmax".into()));
    }
    let bins: Vec<usize> = sel.iter().map(|&s| (s + half) % nfft).collect();
    let nsel = sel.len();
    let nframes = (u.len() - nb) / nh + 1;
    let w = hanning(nb);
    let fft: Arc<dyn Fft<f64>> = FftPlanner::new().plan_fft_forward(nfft);
    // P rows (frame-major); frames are independent → parallel, deterministic
    use rayon::prelude::*;
    let mut p = vec![0.0; nframes * nsel];
    p.par_chunks_mut(nsel).enumerate().for_each_init(
        || {
            (
                vec![Complex64::new(0.0, 0.0); nfft],
                vec![Complex64::new(0.0, 0.0); fft.get_inplace_scratch_len()],
            )
        },
        |(buf, scr), (i, out)| {
            let s = i * nh;
            for (j, b) in buf.iter_mut().enumerate() {
                *b = if j < nb {
                    let z = u[s + j];
                    Complex64::new(z.re * w[j], z.im * w[j])
                } else {
                    Complex64::new(0.0, 0.0)
                };
            }
            fft.process_with_scratch(buf, scr);
            for (o, &b) in out.iter_mut().zip(&bins) {
                let a = cabs(buf[b]);
                *o = a * a;
            }
        },
    );
    // S = clip(10 log10(P / median(P, 1) + 1e-30) - 3, 0, cap)
    let s: Vec<f64> = p
        .par_chunks(nsel)
        .flat_map_iter(|row| {
            let md = median(row);
            row.iter()
                .map(move |&v| (10.0 * (v / md + 1e-30).log10() - 3.0).clamp(0.0, cap))
                .collect::<Vec<_>>()
        })
        .collect();
    drop(p);
    let kk = ((slew * hop / df).ceil() as i64).max(1) as usize;
    let ld = lam * df;
    let pen: Vec<f64> = (0..=2 * kk)
        .map(|m| ld * (m as f64 - kk as f64).abs())
        .collect();
    let mut acc: Vec<f64> = s[..nsel].to_vec();
    let mut back = vec![0u16; nframes * nsel];
    let mut nacc = vec![0.0; nsel];
    for i in 1..nframes {
        let row = &s[i * nsel..(i + 1) * nsel];
        let bk = &mut back[i * nsel..(i + 1) * nsel];
        for j in 0..nsel {
            // win[j, m] = accpad[j + m] - pen[m], accpad = pad(acc, K, -1e30)
            let mut best = f64::NEG_INFINITY;
            let mut bm = 0usize;
            for (m, &pm) in pen.iter().enumerate() {
                let q = j + m;
                let a = if q >= kk && q - kk < nsel {
                    acc[q - kk]
                } else {
                    -1e30
                };
                let v = a - pm;
                if v.is_nan() {
                    // np.argmax: the first NaN wins
                    best = v;
                    bm = m;
                    break;
                }
                if v > best {
                    best = v;
                    bm = m;
                }
            }
            bk[j] = bm as u16;
            nacc[j] = best + row[j];
        }
        std::mem::swap(&mut acc, &mut nacc);
    }
    let mut path = vec![0usize; nframes];
    path[nframes - 1] = argmax(&acc).unwrap_or(0);
    for i in (1..nframes).rev() {
        path[i - 1] = path[i] + back[i * nsel + path[i]] as usize - kk;
    }
    let tc: Vec<f64> = (0..nframes)
        .map(|i| ((i * nh) as f64 + nb as f64 / 2.0) / fs)
        .collect();
    let f: Vec<f64> = path.iter().map(|&p| fr_un[bins[p]]).collect();
    let sc: Vec<f64> = path
        .iter()
        .enumerate()
        .map(|(i, &p)| s[i * nsel + p])
        .collect();
    Ok((tc, f, sc))
}

// ------------------------------------------------------------------- narrow

/// [`narrow`] outputs.
#[derive(Debug, Clone)]
pub struct Narrow {
    pub f: Vec<f64>,
    pub v: Vec<Complex64>,
    pub ud: Vec<Complex64>,
    pub snr_db: Vec<f64>,
    pub good: Vec<bool>,
    pub n0: Vec<f64>,
}

/// `narrow(up, fs_p, B, res_search, min_snr_db, acq_f)`: Viterbi acquisition (or `acq_f`),
/// three zero-phase discriminator refinements flat to B, narrowband SNR and validity.
pub fn narrow(
    up: &[Complex64],
    fs_p: f64,
    b: f64,
    res_search: f64,
    min_snr_db: f64,
    acq_f: Option<&[f64]>,
) -> Res<Narrow> {
    let n = up.len();
    let mut f = match acq_f {
        Some(a) => a.to_vec(),
        None => {
            let (tc, fc, _) = viterbi_track(
                up,
                fs_p,
                res_search,
                1.0f64.max(4.0 / b),
                0.25,
                1.0,
                0.5,
                15.0,
            )?;
            let sm = uniform_filter1d(&medfilt(&fc, 5)?, 3, Mode::Nearest)?;
            interp_mono(&arange_div(n, fs_p), &tc, &sm)
        }
    };
    let h = lpf(b, 1.55 * b, fs_p, 60.0)?;
    let derot = |f: &[f64]| derot(up, &integ(f, fs_p));
    for _ in 0..3 {
        let v = fir0_phasor(&derot(&f), &h);
        let fl = fir0(&f, &h);
        let cd = cdiff_freq(&v, fs_p);
        f = fl.iter().zip(&cd).map(|(a, b)| a + b).collect();
    }
    let ud = derot(&f);
    let v = fir0_phasor(&ud, &h);
    let w = 0.45 * fs_p;
    let lo = (3.0 * b).min(0.5 * res_search + 1.5 * b);
    let hi = (res_search + 3.0 * b).min(w);
    let sh = butter_lowpass_sos(4, hi, fs_p)?;
    let sl = butter_lowpass_sos(4, lo, fs_p)?;
    let nh = sosfiltfilt(&sh, &ud, PadType::Odd, None)?;
    let nl = sosfiltfilt(&sl, &ud, PadType::Odd, None)?;
    let sm = |a: &[f64], sec: f64| {
        uniform_filter1d(a, ((sec * fs_p) as i64).max(1) as usize, Mode::Nearest)
    };
    let nd2: Vec<f64> = nh
        .iter()
        .zip(&nl)
        .map(|(a, b)| {
            let q = cabs(a - b);
            q * q
        })
        .collect();
    let den = 2.0 * (hi - lo);
    let n0: Vec<f64> = sm(&nd2, 1.0)?.iter().map(|v| v / den + 1e-30).collect();
    let bm = 1.25 * b;
    let av: Vec<f64> = v.iter().map(|&z| cabs(z)).collect();
    let av2: Vec<f64> = av.iter().map(|a| a * a).collect();
    let s2 = sm(&av2, 0.2f64.max(0.5 / b))?;
    let snr_db: Vec<f64> = s2
        .iter()
        .zip(&n0)
        .map(|(&s, &n0)| {
            let s = (s - n0 * 2.0 * bm).max(0.0);
            10.0 * (s / (n0 * 2.0 * bm) + 1e-30).log10()
        })
        .collect();
    let mf = median_filter(&av, 3usize.max((2.0 * fs_p) as usize) | 1, Mode::Nearest)?;
    let dips: Vec<bool> = av.iter().zip(&mf).map(|(&a, &m)| a < 0.3 * m).collect();
    let dd = binary_dilation(&dips, ((1.0 / b * fs_p) as i64).max(1) as usize | 1)?;
    let good: Vec<bool> = snr_db
        .iter()
        .zip(&dd)
        .map(|(&s, &d)| s >= min_snr_db && !d)
        .collect();
    let l = ((0.3f64.max(0.6 / b) * fs_p) as i64).max(1) as usize;
    let good = binary_opening(&good, l)?;
    let good = binary_erosion(&good, l | 1)?;
    Ok(Narrow {
        f,
        v,
        ud,
        snr_db,
        good,
        n0,
    })
}

// ------------------------------------------------------------------- refine_amps

/// `refine_amps(model, ud, v, good, N0, tp, fs_p, ref_hz, B, wide, lam)`: signal-domain
/// update of the L3 amplitudes of components with `h·f > 1.5 B` (linear phase detector on
/// the residual, least squares onto the integrated bases). Returns `(model, dmax)`.
#[allow(clippy::too_many_arguments)]
pub fn refine_amps(
    model: &Model,
    nw: &Narrow,
    tp: &[f64],
    fs_p: f64,
    ref_hz: f64,
    b: f64,
    wide: f64,
    lam: f64,
) -> Res<(Model, f64)> {
    let h = lpf(wide, 1.3 * wide, fs_p, 60.0)?;
    let ul = fir0_phasor(&nw.ud, &h);
    let n = nw.v.len();
    let mut sel_all = Vec::new();
    let mut psi = vec![0.0; n];
    let mut wq = vec![0.0; n];
    for i in 0..n {
        let a = cabs(nw.v[i]);
        let av2 = a * a + 1e-30;
        psi[i] = cmul(ul[i], nw.v[i].conj()).im / av2;
        let g = (nw.good[i] as u8 as f64) * (av2 / (nw.n0[i] * 2.0 * wide)).clamp(0.0, 100.0);
        wq[i] = g * ((psi[i].abs() < 1.0) as u8 as f64);
        if wq[i] > 0.0 {
            sel_all.push(i);
        }
    }
    let step = ((fs_p / (2.5 * wide)) as i64).max(1) as usize;
    let sel: Vec<usize> = sel_all.into_iter().step_by(step).collect();
    if sel.len() < 100 {
        return Ok((model.clone(), 0.0));
    }
    let ts: Vec<f64> = sel.iter().map(|&i| tp[i]).collect();
    let ys: Vec<f64> = sel.iter().map(|&i| psi[i]).collect();
    let ws: Vec<f64> = sel.iter().map(|&i| wq[i]).collect();
    let ns = ts.len();

    // keys: ((k, h), M); per key two column blocks (re, im) of the sparse design matrix
    struct Key {
        key: (usize, u32),
        m: usize,
        mi: Vec<usize>,
        ui: Vec<f64>,
        gr: Vec<f64>,
        gi: Vec<f64>,
    }
    let mut keys: Vec<Key> = Vec::new();
    for (k, c) in model.comps.iter().enumerate() {
        let ph: Vec<f64> = ts.iter().map(|&t| 2.0 * PI * c.phase_at(t)).collect();
        let fk: Vec<f64> = ts.iter().map(|&t| c.freq_at(t)).collect();
        for &hh in &c.harmonics {
            let Some(Amp::L3(a)) = model.amp.get(&(k, hh)) else {
                continue;
            };
            if hh as f64 * c.f_mean <= 1.5 * b {
                continue;
            }
            let m = a.len();
            let (mut mi, mut ui) = (Vec::with_capacity(ns), Vec::with_capacity(ns));
            for &t in &ts {
                let (a, u) = hat(t, model.span.0, model.ta, m);
                mi.push(a);
                ui.push(u);
            }
            let hf = hh as f64;
            let mut gr = Vec::with_capacity(ns);
            let mut gi = Vec::with_capacity(ns);
            for i in 0..ns {
                let x = hf * ph[i];
                let (cx, sx) = cos_sin(x);
                let den = hf * fk[i];
                gr.push(ref_hz * sx / den);
                gi.push(ref_hz * cx / den);
            }
            keys.push(Key {
                key: (k, hh),
                m,
                mi,
                ui,
                gr,
                gi,
            });
        }
    }
    if keys.is_empty() {
        return Ok((model.clone(), 0.0));
    }
    // column blocks: block 2q = re of key q, 2q+1 = im
    let nb = 2 * keys.len();
    let bm: Vec<usize> = (0..nb).map(|p| keys[p / 2].m).collect();
    let mut boff = vec![0usize; nb + 1];
    for p in 0..nb {
        boff[p + 1] = boff[p] + bm[p];
    }
    let ncol = boff[nb];
    let same_m = bm.iter().all(|&m| m == bm[0]);
    // knot band half-width of the pattern: data couple |i - j| <= 1, the 2nd-difference
    // penalty |i - j| <= 2 (any pair when the knot counts differ)
    let hw = if same_m { 2 } else { *bm.iter().max().unwrap() };
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(ncol);
    // base[c * nb + q]: position in `val` of (block q, knot jlo(c, q)) within column c
    let mut jlo = vec![0usize; ncol * nb];
    for p in 0..nb {
        for i in 0..bm[p] {
            let c = boff[p] + i;
            let mut r = Vec::new();
            for q in p..nb {
                let lo = if q == p { i } else { i.saturating_sub(hw) };
                let hi = (i + hw + 1).min(bm[q]);
                jlo[c * nb + q] = lo;
                for j in lo..hi {
                    r.push((boff[q] + j) as u32);
                }
            }
            rows.push(r);
        }
    }
    let mut nmat = SymCsc::from_columns(rows);
    let mut base = vec![usize::MAX; ncol * nb];
    for p in 0..nb {
        for i in 0..bm[p] {
            let c = boff[p] + i;
            let mut pos = nmat.col_ptr[c] as usize;
            for q in p..nb {
                base[c * nb + q] = pos;
                let lo = jlo[c * nb + q];
                let hi = (i + hw + 1).min(bm[q]);
                pos += hi.saturating_sub(lo);
            }
        }
    }
    // N = Aᵀ (A·w), b = (A·w)ᵀ y: rows in ascending order (scipy's accumulation order)
    let mut bvec = vec![0.0; ncol];
    let mut ent: Vec<(usize, usize, usize, f64)> = Vec::with_capacity(4 * keys.len());
    for r in 0..ns {
        ent.clear();
        for (q, kq) in keys.iter().enumerate() {
            let (m, u) = (kq.mi[r], kq.ui[r]);
            for (part, g) in [(0usize, kq.gr[r]), (1, kq.gi[r])] {
                let blk = 2 * q + part;
                ent.push((blk, m, boff[blk] + m, (1.0 - u) * g));
                ent.push((blk, m + 1, boff[blk] + m + 1, u * g));
            }
        }
        let w = ws[r];
        for &(_, _, c, a) in &ent {
            bvec[c] += (a * w) * ys[r];
        }
        for &(p1, i1, c1, a1) in &ent {
            for &(p2, i2, c2, a2) in &ent {
                if c2 < c1 {
                    continue;
                }
                // lower triangle: row c2 >= column c1; N[c2, c1] = Σ A[k, c2] · (A[k, c1] · w)
                let q = c1 * nb + p2;
                nmat.val[base[q] + i2 - jlo[q]] += a2 * (a1 * w);
                let _ = (p1, i1);
            }
        }
    }
    let diag = nmat.diagonal();
    let scale = median(&diag);
    let ls = lam * scale;
    let eps = 1e-4 * scale;
    for p in 0..nb {
        let m = bm[p];
        // D2ᵀD2 of the (max(M-2, 1), M) second-difference matrix
        let mut dtd: IndexMap<(usize, usize), f64> = IndexMap::new();
        for r in 0..(m.saturating_sub(2)).max(1) {
            let nz: Vec<(usize, f64)> = [(r, 1.0), (r + 1, -2.0), (r + 2, 1.0)]
                .into_iter()
                .filter(|&(c, _)| c < m)
                .collect();
            for &(i, a) in &nz {
                for &(j, bb) in &nz {
                    if j >= i {
                        *dtd.entry((i, j)).or_insert(0.0) += a * bb;
                    }
                }
            }
        }
        for i in 0..m {
            let c = boff[p] + i;
            for j in i..(i + 3).min(m) {
                let mut rv = ls * dtd.get(&(i, j)).copied().unwrap_or(0.0);
                if j == i {
                    rv += eps;
                }
                let pos = base[c * nb + p] + j - jlo[c * nb + p];
                nmat.val[pos] += rv;
            }
        }
    }
    let x = spd_solve(&nmat, &bvec)?;
    let mut m_new = model.clone();
    let mut dmax = 0.0f64;
    let mut j = 0;
    for kq in &keys {
        let mm = kq.m;
        let Some(Amp::L3(a0)) = model.amp.get(&kq.key) else {
            unreachable!()
        };
        let a1: Vec<Complex64> = (0..mm)
            .map(|i| {
                let d = Complex64::new(x[j + i], x[j + mm + i]);
                dmax = dmax.max(cabs(d));
                a0[i] + d
            })
            .collect();
        m_new.amp.insert(kq.key, Amp::L3(a1));
        j += 2 * mm;
    }
    Ok((m_new, dmax))
}

// ------------------------------------------------------------------ track_aided

/// Keyword arguments of [`track_aided`] (with their defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AidedOpts {
    /// residual band [Hz]
    pub b: f64,
    /// acquisition search [Hz]
    pub res_search: f64,
    pub channel: Channel,
    pub min_snr_db: f64,
    /// [`refine_amps`] iterations
    pub refine: usize,
    /// residual band of the amplitude refinement [Hz]
    pub wide: f64,
}

impl Default for AidedOpts {
    fn default() -> Self {
        AidedOpts {
            b: 2.0,
            res_search: 6.0,
            channel: Channel::Mid,
            min_snr_db: 6.0,
            refine: 0,
            wide: 25.0,
        }
    }
}

/// `track_aided(..., return_debug=True)` debug dict.
#[derive(Debug, Clone)]
pub struct AidedDebug {
    pub tp: Vec<f64>,
    pub f: Vec<f64>,
    pub snr_db: Vec<f64>,
    pub good: Vec<bool>,
    pub rm: Vec<f64>,
    pub fs_p: f64,
    /// final (refined) model
    pub model: Model,
    pub valid_hist: Vec<f64>,
    /// [`refine_amps`] dmax per applied refinement
    pub dmax: Vec<f64>,
}

pub(crate) fn channel_str(c: Channel) -> &'static str {
    match c {
        Channel::Mid => "mid",
        Channel::S => "S",
        Channel::L => "L",
        Channel::R => "R",
    }
}

/// `track_aided(x, sr, ref_hz, model, B, res_search, channel, min_snr_db, refine, wide)`:
/// derotate the reference by the model phase `2π ref ∫r_model`, track only the residual
/// (`|f| <= res_search`) flat to B Hz; `refine` signal-domain L3 amplitude updates first.
/// `x`: interleaved f32 frames with `ch` channels.
pub fn track_aided(
    x: &[f32],
    ch: usize,
    sr: u32,
    ref_hz: f64,
    model: &Model,
    o: &AidedOpts,
) -> Res<(Curve, AidedDebug)> {
    let srf = sr as f64;
    let n = x.len() / ch;
    let t = grid_t(n, srf);
    let rm0 = model.predict(&t);
    let rbar = mean(&rm0);
    let w = (o.res_search + 3.0 * o.b).max(if o.refine > 0 { o.wide + 3.0 } else { 0.0 });
    let dev = rm0.iter().fold(0.0f64, |m, v| m.max((v - rbar).abs())) * ref_hz + w;
    drop(rm0);
    let fh1 = 1.15 * dev;
    let d1 = ((srf / (2.5 * fh1)) as i64).max(1) as usize;
    let fs_d = srf / d1 as f64;
    let h1 = lpf(fh1, (fs_d - fh1).min(1.3 * fh1), srf, 90.0)?;
    let z = het_decimate(x, ch, srf, ref_hz * rbar, d1, &h1, o.channel)?;
    let td = arange_div(z.len(), fs_d);
    let d2 = ((fs_d / (4.0 * w)) as i64).max(1) as usize;
    let fs_p = fs_d / d2 as f64;
    let h2 = lpf(w, (fs_p - w).min(1.5 * w), fs_d, 90.0)?;
    let mut tp = arange_div(z.len().div_ceil(d2), fs_p);
    let mut model = model.clone();
    let mut hist = Vec::new();
    let mut dmaxs = Vec::new();
    let t_end = t[t.len() - 1];
    let c = -2.0 * PI * ref_hz;
    let mut last: Option<(Vec<f64>, Narrow)> = None;
    for it in 0..=o.refine {
        let rm = model.predict(&t);
        let dr: Vec<f64> = rm.iter().map(|v| v - rbar).collect();
        let phi = cubic_spline(&t, &dr)?.antiderivative();
        let mut hint = 0;
        let u: Vec<Complex64> = z
            .iter()
            .zip(&td)
            .map(|(&zz, &tv)| cmul(zz, cis(c * phi.eval_hint(tv.clamp(0.0, t_end), &mut hint))))
            .collect();
        let up = if d2 > 1 {
            resample_poly_down_c(&u, d2, &h2)?
        } else {
            u
        };
        tp.truncate(up.len());
        let nw = narrow(&up, fs_p, o.b, o.res_search, o.min_snr_db, None)?;
        drop(up);
        hist.push(nw.good.iter().filter(|&&g| g).count() as f64 / nw.good.len() as f64);
        let stop = if it < o.refine && model.level >= 3 {
            let (m_new, dmax) = refine_amps(&model, &nw, &tp, fs_p, ref_hz, o.b, o.wide, 0.3)?;
            if m_new.predict(&t).iter().all(|v| v.is_finite()) {
                model = m_new;
                dmaxs.push(dmax);
                false
            } else {
                true // ill-conditioned update: keep the model
            }
        } else {
            false
        };
        last = Some((rm, nw));
        if stop {
            break;
        }
    }
    let (rm, nw) = last.expect("at least one iteration");
    let mut fres = interp_mono(&t, &tp, &nw.f);
    let ng = nw.good.len();
    let vg: Vec<bool> = t
        .iter()
        .map(|&tv| {
            let i = (tv * fs_p).round_ties_even().clamp(0.0, (ng - 1) as f64) as usize;
            nw.good[i]
        })
        .collect();
    let nvg = vg.iter().filter(|&&v| v).count();
    if nvg >= 2 {
        let (tv, fv): (Vec<f64>, Vec<f64>) = (0..t.len())
            .filter(|&i| vg[i])
            .map(|i| (t[i], fres[i]))
            .unzip();
        let fi = interp_mono(&t, &tv, &fv);
        for i in 0..t.len() {
            if !vg[i] {
                fres[i] = fi[i];
            }
        }
    } else {
        fres = vec![0.0; t.len()];
    }
    let si = interp_mono(&t, &tp, &nw.snr_db);
    let conf: Vec<f64> = si
        .iter()
        .zip(&vg)
        .map(|(&s, &v)| {
            np_round(
                (s / (2.0 * o.min_snr_db)).clamp(0.0, 1.0) * (v as u8 as f64),
                4,
            )
        })
        .collect();
    let ratio: Vec<f64> = rm.iter().zip(&fres).map(|(a, f)| a + f / ref_hz).collect();
    let mut meta = IndexMap::new();
    meta.insert("ref_hz".into(), MetaValue::Float(ref_hz));
    meta.insert("sr".into(), MetaValue::Int(sr as i64));
    meta.insert(
        "channel".into(),
        MetaValue::Str(channel_str(o.channel).into()),
    );
    meta.insert("aided_B".into(), MetaValue::Float(o.b));
    meta.insert("res_search".into(), MetaValue::Float(o.res_search));
    meta.insert("refine".into(), MetaValue::Int(o.refine as i64));
    let curve = Curve {
        t,
        ratio,
        confidence: conf,
        valid: vg,
        meta,
    };
    let dbg = AidedDebug {
        tp,
        f: nw.f,
        snr_db: nw.snr_db,
        good: nw.good,
        rm,
        fs_p,
        model,
        valid_hist: hist,
        dmax: dmaxs,
    };
    Ok((curve, dbg))
}

/// `aided_iterate(x, sr, ref_hz, curve0, iters, B, fmax, Td, Ta, comps, res_search, refine,
/// channel, ident_kw)`: fit on `curve0`, aided track, refit on the aided curve, repeat.
/// Returns `(curve, model, fit_hier result, curves per iteration)`.
#[allow(clippy::too_many_arguments)]
pub fn aided_iterate(
    x: &[f32],
    ch: usize,
    sr: u32,
    ref_hz: f64,
    curve0: &Curve,
    iters: usize,
    hier: &HierOpts,
    o: &AidedOpts,
) -> Res<(Curve, Model, super::Hier, Vec<Curve>)> {
    let w: Vec<f64> = curve0.valid.iter().map(|&v| v as u8 as f64).collect();
    let h = fit_hier(&curve0.t, &curve0.ratio, &w, hier)?;
    let comps = h.comps.clone();
    let mut m = h.m3.clone();
    let mut hist: Vec<Curve> = Vec::new();
    for _ in 0..iters {
        let (c, _) = track_aided(x, ch, sr, ref_hz, &m, o)?;
        let wc: Vec<f64> = c.valid.iter().map(|&v| v as u8 as f64).collect();
        m = fit(
            &c.t,
            &c.ratio,
            &wc,
            &comps,
            &FitOpts {
                level: 3,
                td: hier.td,
                ta: hier.ta,
                ..FitOpts::default()
            },
        )?;
        hist.push(c);
    }
    let c = hist
        .last()
        .cloned()
        .ok_or_else(|| ValueError("aided_iterate: iters must be >= 1".into()))?;
    Ok((c, m, h, hist))
}

/// Keyword arguments of [`coarse_track`] (with their defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoarseOpts {
    pub search: f64,
    pub tb: f64,
    pub hop: f64,
    /// max |dr/dt| [1/s]
    pub slew: f64,
    pub min_db: f64,
    pub channel: Channel,
}

impl Default for CoarseOpts {
    fn default() -> Self {
        CoarseOpts {
            search: 0.05,
            tb: 1.0,
            hop: 0.25,
            slew: 0.002,
            min_db: 10.0,
            channel: Channel::Mid,
        }
    }
}

/// `coarse_track(x, sr, ref_hz, search, tb, hop, slew, min_db, channel)`: bootstrap curve
/// from a Viterbi ridge of a `tb`-second STFT within ±search (slow speed only).
pub fn coarse_track(x: &[f32], ch: usize, sr: u32, ref_hz: f64, o: &CoarseOpts) -> Res<Curve> {
    let srf = sr as f64;
    let fh = o.search * ref_hz * 1.1 + 20.0;
    let d = ((srf / (2.5 * fh)) as i64).max(1) as usize;
    let fs = srf / d as f64;
    let h = lpf(fh, (fs - fh).min(1.3 * fh), srf, 90.0)?;
    let z = het_decimate(x, ch, srf, ref_hz, d, &h, o.channel)?;
    let (tc, fc, sc) = viterbi_track(
        &z,
        fs,
        o.search * ref_hz,
        o.tb,
        o.hop,
        o.slew * ref_hz,
        0.1,
        15.0,
    )?;
    let n = x.len() / ch;
    let t = grid_t(n, srf);
    let good = binary_opening(
        &sc.iter().map(|&s| s + 3.0 >= o.min_db).collect::<Vec<_>>(),
        3,
    )?;
    let fm = median_filter(&fc, 5, Mode::Nearest)?;
    let rc: Vec<f64> = fm.iter().map(|v| 1.0 + v / ref_hz).collect();
    let r = interp_mono(&t, &tc, &rc);
    let gf: Vec<f64> = good.iter().map(|&g| g as u8 as f64).collect();
    let vg: Vec<bool> = interp_mono(&t, &tc, &gf).iter().map(|&v| v > 0.5).collect();
    let mut meta = IndexMap::new();
    meta.insert("ref_hz".into(), MetaValue::Float(ref_hz));
    meta.insert("sr".into(), MetaValue::Int(sr as i64));
    meta.insert(
        "channel".into(),
        MetaValue::Str(channel_str(o.channel).into()),
    );
    meta.insert("coarse_tb".into(), MetaValue::Float(o.tb));
    Ok(Curve {
        ratio: pchip_fill(&t, &r, &vg)?,
        confidence: vg.iter().map(|&v| v as u8 as f64).collect(),
        valid: vg,
        t,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integ_trapezoid() {
        let p = integ(&[1.0, 1.0, 3.0], 2.0);
        assert_eq!(p[0], 0.0);
        assert!((p[1] - PI).abs() < 1e-15);
        assert!((p[2] - 3.0 * PI).abs() < 1e-15);
    }

    #[test]
    fn fir0_short_is_mean() {
        let h = [0.25, 0.5, 0.25, 0.0, 0.0];
        assert_eq!(fir0(&[1.0, 3.0, 5.0], &h), vec![3.0; 3]);
        // identity kernel keeps the signal (odd extension invisible)
        let a: Vec<f64> = (0..20).map(|i| (i * i) as f64).collect();
        let id = [0.0, 0.0, 1.0, 0.0, 0.0];
        let y = fir0(&a, &id);
        assert!(y.iter().zip(&a).all(|(p, q)| (p - q).abs() < 1e-9));
    }

    #[test]
    fn cdiff_of_tone() {
        let fs = 100.0;
        let v: Vec<Complex64> = (0..50)
            .map(|i| cis(2.0 * PI * 3.0 * i as f64 / fs))
            .collect();
        let f = cdiff_freq(&v, fs);
        assert!(f.iter().all(|x| (x - 3.0).abs() < 1e-9));
    }
}
