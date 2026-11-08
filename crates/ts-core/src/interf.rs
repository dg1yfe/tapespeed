//! interf: stationary narrowband interferers (CW, carriers) in
//! the corrected output — detect, classify keyed/steady, notch only where present.
//!
//! Whole-array API (reference semantics): [`detect`], [`keying`], [`is_keyed`], [`notch_spans`].
//! Streaming API for the 2-pass auto-notch: [`Detector`] (pass 1, O(60 s) memory),
//! [`KeyingStream`] (span level statistics, O(frame batch) memory), [`GatedNotch`] (a
//! time-gated notch stage for the pass-2 chain) and [`run_gated`] (correct → reference/manual
//! notches → gated notches → lowpass → sink).
//!
//! Numerics: the STFT is f64 (`ts_dsp::spectral`). A complex64/float32 STFT of the float32
//! mono mix differs at ~1e-6 relative (keying contrast ~3e-5 dB); with float64 input the
//! results match the stored reference vectors to ~1e-12.
//! Determinism: chunks / frames / spans are independent work units merged in index order;
//! every value is computed identically for any thread count.

use std::collections::{BTreeMap, VecDeque};

use rayon::prelude::*;
use ts_dsp::ValueError;
use ts_dsp::iir::{FiltFiltStream, iirnotch};
use ts_dsp::peaks::find_peaks;
use ts_dsp::spectral::{Stft, n_frames, stft_frame_major};
use ts_dsp::stats::{median, percentiles};

use crate::correct::{self, NOTCH_LOOKAHEAD, Params, Post, PostChain, Resampler, RunError};

/// Detection STFT length (~10.8 Hz bins at 44.1 kHz).
pub const NPER: usize = 4096;
/// Presence block [s].
pub const BLOCK: f64 = 5.0;
/// Line vs ±100..300 Hz neighbourhood [dB] (per frame).
pub const MIN_SNR: f64 = 15.0;
/// Share of frames in a block for "present".
pub const MIN_ON: f64 = 0.2;
/// Cumulative presence for a detection [s].
pub const MIN_TOTAL: f64 = 20.0;
/// p90-p10 of the line level for "keyed" [dB].
pub const KEY_CONTRAST: f64 = 25.0;
/// Max share of frames in the middle third.
pub const KEY_MID: f64 = 0.25;
/// Interquartile range of the frame peak frequency [Hz].
pub const MAX_SPREAD: f64 = 3.0;

type Res<T> = Result<T, ValueError>;

fn verr<T>(m: impl Into<String>) -> Res<T> {
    Err(ValueError(m.into()))
}

/// One detected line (`dict(hz, total_s, spans, snr_db)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub hz: f64,
    pub total_s: f64,
    /// presence spans [s] (multiples of [`BLOCK`])
    pub spans: Vec<(f64, f64)>,
    pub snr_db: f64,
}

/// Keyword parameters of [`detect`] (defaults; `fmin` = `--auto-notch-min`).
#[derive(Debug, Clone, PartialEq)]
pub struct DetectParams {
    pub fmin: f64,
    /// `None` = 0.47·sr
    pub fmax: Option<f64>,
    /// [s]
    pub chunk: f64,
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            fmin: 3000.0,
            fmax: None,
            chunk: 60.0,
        }
    }
}

/// `y.mean(axis=1)` of float32 interleaved frames (float32 sum in channel order, / ch).
pub fn mono_mean(y: &[f32], ch: usize) -> Vec<f32> {
    if ch == 1 {
        return y.to_vec();
    }
    let d = ch as f32;
    y.chunks_exact(ch)
        .map(|fr| {
            let mut s = -0.0f32;
            for &v in fr {
                s += v;
            }
            s / d
        })
        .collect()
}

// ------------------------------------------------------------------ detection

/// Candidate event: (bin, time [s], interpolated frequency [Hz], snr [dB]).
type Ev = (usize, f64, f64, f64);

/// Frames per parallel work unit inside a chunk.
const FRAME_UNIT: usize = 32;

/// All candidate peaks of one detection chunk `seg = y[k0 : k0 + step + NPER]`, in a fixed
/// order (frames, then peaks ascending). Frames are processed in parallel.
fn chunk_events(seg: &[f64], k0: usize, sr: f64, fmin: f64, fmax: f64) -> Res<Vec<Ev>> {
    let s = stft_frame_major(seg, sr, NPER, Some(NPER / 2))?;
    if s.nframes == 0 {
        return Ok(Vec::new());
    }
    let nf = s.nfreq;
    if nf < 2 {
        return verr("detect: chunk too short for a frequency grid");
    }
    let df = s.f[1] - s.f[0];
    let w_in = (100.0 / df) as usize;
    let w_out = (300.0 / df) as usize;
    let inv = 1.0 / (2 * w_out.saturating_sub(w_in)) as f64;
    let m: Vec<bool> = s.f.iter().map(|&f| f >= fmin && f <= fmax).collect();
    let dist = ((50.0 / df) as i64).max(1) as f64;
    let t_off = k0 as f64 / sr;
    let frame_events = |j: usize| -> Res<Vec<Ev>> {
        let row = &s.z[j * nf..(j + 1) * nf];
        let l: Vec<f64> = row
            .iter()
            .map(|z| 20.0 * (z.re.hypot(z.im) + 1e-12).log10())
            .collect();
        let nb = ring_mean(&l, w_in, w_out, inv);
        let snr: Vec<f64> = (0..nf)
            .map(|k| if m[k] { l[k] - nb[k] } else { 0.0 })
            .collect();
        let pk = find_peaks(&snr, Some(MIN_SNR), Some(dist))?;
        Ok(pk
            .into_iter()
            .map(|i| {
                let (a, b, c) = (l[i - 1], l[i], l[i + 1]);
                let den = a - 2.0 * b + c;
                let d = if den != 0.0 { 0.5 * (a - c) / den } else { 0.0 };
                (i, t_off + s.t[j], s.f[i] + d * df, snr[i])
            })
            .collect())
    };
    let units: Vec<Vec<Ev>> = (0..s.nframes.div_ceil(FRAME_UNIT))
        .into_par_iter()
        .map(|u| -> Res<Vec<Ev>> {
            let mut v = Vec::new();
            for j in u * FRAME_UNIT..((u + 1) * FRAME_UNIT).min(s.nframes) {
                v.extend(frame_events(j)?);
            }
            Ok(v)
        })
        .collect::<Res<_>>()?;
    Ok(units.concat())
}

/// Neighbourhood level `fftconvolve(L, ker / ker.sum(), "same")` with
/// `ker = [1]*(w_out-w_in) + [0]*(2 w_in + 1) + [1]*(w_out-w_in)`, evaluated directly:
/// `nb[k] = (sum of L[k ± d], w_in < d <= w_out, zero outside) / ker.sum()`. Same value as
/// scipy's FFT convolution to ~1e-15 relative (no FFT rounding), ~5x faster per frame.
fn ring_mean(l: &[f64], w_in: usize, w_out: usize, inv: f64) -> Vec<f64> {
    let n = l.len() as isize;
    let (wi, wo) = (w_in as isize, w_out as isize);
    (0..n)
        .map(|k| {
            let mut s = 0.0;
            for j in (k - wo).max(0)..(k - wi).clamp(0, n) {
                s += l[j as usize];
            }
            for j in (k + wi + 1).clamp(0, n)..(k + wo + 1).min(n) {
                s += l[j as usize];
            }
            s * inv
        })
        .collect()
}

/// Per-bin candidate lists → detections (presence blocks, span bridging, merge).
fn summarise(cand: BTreeMap<usize, Vec<(f64, f64, f64)>>, sr: f64) -> Res<Vec<Detection>> {
    let dt_fr = NPER as f64 / 2.0 / sr;
    let mut out = Vec::new();
    for ev in cand.values() {
        let total = ev.len() as f64 * dt_fr;
        if total < MIN_TOTAL {
            continue;
        }
        let fs: Vec<f64> = ev.iter().map(|e| e.1).collect();
        let q = percentiles(&fs, &[25.0, 75.0])?;
        if q[1] - q[0] > MAX_SPREAD {
            continue;
        }
        let blk: Vec<usize> = ev.iter().map(|e| (e.0 / BLOCK).floor() as usize).collect();
        let mut cnt = vec![0usize; blk.iter().max().map_or(0, |m| m + 1)];
        for &b in &blk {
            cnt[b] += 1;
        }
        let on: Vec<usize> = (0..cnt.len())
            .filter(|&b| cnt[b] as f64 * dt_fr / BLOCK >= MIN_ON)
            .collect();
        if on.is_empty() {
            continue;
        }
        let mut spans = Vec::new();
        let (mut s0, mut prev) = (on[0], on[0]);
        for &b in &on[1..] {
            if b > prev + 2 {
                spans.push((s0 as f64 * BLOCK, (prev + 1) as f64 * BLOCK));
                s0 = b;
            }
            prev = b;
        }
        spans.push((s0 as f64 * BLOCK, (prev + 1) as f64 * BLOCK));
        let snr: Vec<f64> = ev.iter().map(|e| e.2).collect();
        out.push(Detection {
            hz: median(&fs),
            total_s: total,
            spans,
            snr_db: median(&snr),
        });
    }
    out.sort_by(|a, b| a.hz.total_cmp(&b.hz));
    let mut merged: Vec<Detection> = Vec::new();
    for d in out {
        match merged.last_mut() {
            Some(last) if (d.hz - last.hz).abs() < 2.0 * MAX_SPREAD => {
                if d.total_s > last.total_s {
                    *last = d;
                }
            }
            _ => merged.push(d),
        }
    }
    Ok(merged)
}

fn add_events(cand: &mut BTreeMap<usize, Vec<(f64, f64, f64)>>, ev: Vec<Ev>) {
    for (i, t, f, s) in ev {
        cand.entry(i).or_default().push((t, f, s));
    }
}

fn fmax_of(p: &DetectParams, sr: f64) -> f64 {
    match p.fmax {
        Some(f) if f != 0.0 => f,
        _ => 0.47 * sr,
    }
}

/// `detect(y, sr, fmin, fmax, chunk)`: lines that stay at one frequency (IQR ≤ MAX_SPREAD)
/// for ≥ MIN_TOTAL s. `y`: mono (use [`mono_mean`] for the pipeline's `y.mean(axis=1)`).
/// Chunks are scanned in parallel (groups of `current_num_threads` chunks bound memory).
/// Errors like scipy when the last chunk holds ≤ NPER/2 samples (noverlap ≥ nperseg).
pub fn detect(y: &[f64], sr: u32, p: &DetectParams) -> Res<Vec<Detection>> {
    let srf = sr as f64;
    let fmax = fmax_of(p, srf);
    let n = y.len();
    let step = (p.chunk * srf) as usize;
    if step == 0 {
        return verr("detect: chunk must be >= 1 sample");
    }
    let starts: Vec<usize> = (0..n).step_by(step).collect();
    let g = rayon::current_num_threads().max(1);
    let mut cand = BTreeMap::new();
    for grp in starts.chunks(g) {
        let evs: Vec<Vec<Ev>> = grp
            .par_iter()
            .map(|&k0| chunk_events(&y[k0..(k0 + step + NPER).min(n)], k0, srf, p.fmin, fmax))
            .collect::<Res<_>>()?;
        for ev in evs {
            add_events(&mut cand, ev);
        }
    }
    summarise(cand, srf)
}

/// Streaming [`detect`]: push mono samples (any block size); each 60 s chunk (+NPER overlap)
/// is scanned as soon as it is complete, so memory stays O(chunk). Same result as
/// [`detect`] on the concatenated input, bit for bit.
pub struct Detector {
    sr: f64,
    fmin: f64,
    fmax: f64,
    step: usize,
    k0: usize,
    buf: Vec<f64>,
    cand: BTreeMap<usize, Vec<(f64, f64, f64)>>,
}

impl Detector {
    pub fn new(sr: u32, p: &DetectParams) -> Res<Self> {
        let srf = sr as f64;
        let step = (p.chunk * srf) as usize;
        if step == 0 {
            return verr("detect: chunk must be >= 1 sample");
        }
        Ok(Self {
            sr: srf,
            fmin: p.fmin,
            fmax: fmax_of(p, srf),
            step,
            k0: 0,
            buf: Vec::new(),
            cand: BTreeMap::new(),
        })
    }

    fn scan(&mut self, len: usize) -> Res<()> {
        let ev = chunk_events(&self.buf[..len], self.k0, self.sr, self.fmin, self.fmax)?;
        add_events(&mut self.cand, ev);
        let d = self.step.min(self.buf.len());
        self.buf.drain(..d);
        self.k0 += self.step;
        Ok(())
    }

    /// Feed mono samples.
    pub fn push(&mut self, y: impl IntoIterator<Item = f64>) -> Res<()> {
        self.buf.extend(y);
        while self.buf.len() >= self.step + NPER {
            self.scan(self.step + NPER)?;
        }
        Ok(())
    }

    /// Feed interleaved f32 frames as their float32 channel mean (pipeline `y.mean(axis=1)`).
    pub fn push_frames(&mut self, y: &[f32], ch: usize) -> Res<()> {
        let m = mono_mean(y, ch);
        self.push(m.into_iter().map(f64::from))
    }

    /// End of input: scan the remaining chunk(s) and summarise.
    pub fn finish(mut self) -> Res<Vec<Detection>> {
        while !self.buf.is_empty() {
            let len = self.buf.len().min(self.step + NPER);
            self.scan(len)?;
        }
        summarise(std::mem::take(&mut self.cand), self.sr)
    }
}

// ------------------------------------------------------------------ keying

/// Keying STFT length: `1 << round(log2(0.025 * sr))` (~25 ms).
pub fn keying_nper(sr: u32) -> usize {
    1usize << ((0.025 * sr as f64).log2().round_ties_even() as u32)
}

/// Line geometry of `keying` for one STFT grid.
struct KeyGrid {
    i: usize,
    hi: usize,
    nb: Vec<usize>,
}

fn key_grid(hz: f64, df: f64, nfreq: usize) -> Res<KeyGrid> {
    let i = (hz / df).round_ties_even() as i64;
    let (w3, w1) = ((300.0 / df) as i64, (100.0 / df) as i64);
    let n = nfreq as i64;
    if i < 1 || i >= n {
        return verr("keying: line outside the STFT grid");
    }
    let mut nb = Vec::new();
    for j in (i - w3..i - w1).chain(i + w1..i + w3) {
        let jj = if j < 0 { j + n } else { j }; // numpy negative fancy index
        if !(0..n).contains(&jj) {
            return verr(format!("keying: index {j} is out of bounds for size {n}"));
        }
        nb.push(jj as usize);
    }
    Ok(KeyGrid {
        i: i as usize,
        hi: (i as usize + 2).min(nfreq),
        nb,
    })
}

/// `line - median(neighbourhood)` [dB] for every frame of `s`.
fn key_vals(s: &Stft, hz: f64) -> Res<Vec<f64>> {
    if s.nframes == 0 {
        return Ok(Vec::new());
    }
    let nf = s.nfreq;
    let g = key_grid(hz, s.f[1] - s.f[0], nf)?;
    let db = |z: num_complex::Complex64| 20.0 * (z.re.hypot(z.im) + 1e-12).log10();
    Ok((0..s.nframes)
        .map(|j| {
            let row = &s.z[j * nf..(j + 1) * nf];
            let mut line = f64::NEG_INFINITY;
            for z in &row[g.i - 1..g.hi] {
                let v = db(*z);
                if v > line || v.is_nan() {
                    line = v;
                }
            }
            let nbv: Vec<f64> = g.nb.iter().map(|&k| db(row[k])).collect();
            line - median(&nbv)
        })
        .collect())
}

/// Contrast / mid share from the concatenated per-frame values.
fn key_stats(mut v: Vec<f64>) -> Res<(f64, f64)> {
    if v.is_empty() {
        v.push(0.0);
    }
    let q = percentiles(&v, &[10.0, 90.0])?;
    let (lo, hi) = (q[0], q[1]);
    let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
    let mid = v.iter().filter(|&&x| x > a && x < b).count() as f64 / v.len() as f64;
    Ok((hi - lo, mid))
}

fn span_range(t0: f64, t1: f64, sr: f64, nper: usize, n: usize) -> (usize, usize) {
    let lo = ((t0 * sr) as i64).max(0) as usize;
    let hi = (((t1 * sr) as i64).max(0) as usize + nper).min(n);
    (lo.min(n), hi)
}

/// `keying(y, sr, hz, spans)` → (contrast p90-p10 [dB], mid-third share). Spans in parallel.
pub fn keying(y: &[f64], sr: u32, hz: f64, spans: &[(f64, f64)]) -> Res<(f64, f64)> {
    let nper = keying_nper(sr);
    let srf = sr as f64;
    let vals: Vec<Vec<f64>> = spans
        .par_iter()
        .map(|&(t0, t1)| {
            let (lo, hi) = span_range(t0, t1, srf, nper, y.len());
            if lo >= hi {
                return verr("keying: empty span slice (tuple index out of range)");
            }
            key_vals(
                &stft_frame_major(&y[lo..hi], srf, nper, Some(nper / 2))?,
                hz,
            )
        })
        .collect::<Res<_>>()?;
    key_stats(vals.concat())
}

/// `is_keyed(contrast, mid)`.
pub fn is_keyed(contrast: f64, mid: f64) -> bool {
    contrast >= KEY_CONTRAST && mid <= KEY_MID
}

/// Frames per batch in [`KeyingStream`].
const KEY_BATCH: usize = 512;

struct KeySpan {
    det: usize,
    lo: usize,
    /// nominal end `int(t1*sr) + nper` (clipped by the end of input)
    end: usize,
    /// samples from `lo + done * hop`
    buf: Vec<f64>,
    done: usize,
    vals: Vec<f64>,
}

/// Streaming [`keying`] for several detections at once: push mono samples (global order);
/// each span's 25 ms frames are evaluated in batches as soon as their samples arrived.
/// Same values as [`keying`] on the whole signal.
pub struct KeyingStream {
    sr: f64,
    nper: usize,
    hz: Vec<f64>,
    spans: Vec<KeySpan>,
    pos: usize,
}

impl KeyingStream {
    /// `lines`: (hz, spans) per detection.
    pub fn new(sr: u32, lines: &[(f64, Vec<(f64, f64)>)]) -> Self {
        let nper = keying_nper(sr);
        let srf = sr as f64;
        let mut spans = Vec::new();
        for (d, (_, sp)) in lines.iter().enumerate() {
            for &(t0, t1) in sp {
                let (lo, end) = span_range(t0, t1, srf, nper, usize::MAX);
                spans.push(KeySpan {
                    det: d,
                    lo,
                    end,
                    buf: Vec::new(),
                    done: 0,
                    vals: Vec::new(),
                });
            }
        }
        Self {
            sr: srf,
            nper,
            hz: lines.iter().map(|l| l.0).collect(),
            spans,
            pos: 0,
        }
    }

    fn run_batch(sp: &mut KeySpan, hz: f64, sr: f64, nper: usize, frames: usize) -> Res<()> {
        let hop = nper - nper / 2;
        let len = nper + (frames - 1) * hop;
        let s = stft_frame_major(&sp.buf[..len], sr, nper, Some(nper / 2))?;
        sp.vals.extend(key_vals(&s, hz)?);
        sp.buf.drain(..frames * hop);
        sp.done += frames;
        Ok(())
    }

    /// Feed mono samples.
    pub fn push(&mut self, y: &[f64]) -> Res<()> {
        let (p0, p1) = (self.pos, self.pos + y.len());
        let (sr, nper) = (self.sr, self.nper);
        let hop = nper - nper / 2;
        let hz = &self.hz;
        self.spans.par_iter_mut().try_for_each(|sp| -> Res<()> {
            let start = sp.lo + sp.done * hop + sp.buf.len();
            let (a, b) = (start.max(p0), sp.end.min(p1));
            if a < b {
                sp.buf.extend_from_slice(&y[a - p0..b - p0]);
            }
            let ready = n_frames(sp.buf.len(), nper, hop);
            if ready >= KEY_BATCH {
                Self::run_batch(sp, hz[sp.det], sr, nper, ready)?;
            }
            Ok(())
        })?;
        self.pos = p1;
        Ok(())
    }

    /// Feed interleaved f32 frames as their float32 channel mean.
    pub fn push_frames(&mut self, y: &[f32], ch: usize) -> Res<()> {
        let m: Vec<f64> = mono_mean(y, ch).into_iter().map(f64::from).collect();
        self.push(&m)
    }

    /// End of input → (contrast, mid) per detection.
    pub fn finish(mut self) -> Res<Vec<(f64, f64)>> {
        let (sr, nper) = (self.sr, self.nper);
        let hop = nper - nper / 2;
        for sp in &mut self.spans {
            let hz = self.hz[sp.det];
            if sp.done == 0 && sp.buf.len() < nper {
                // whole slice shorter than nper: scipy clips nperseg (or raises)
                if sp.buf.is_empty() {
                    return verr("keying: empty span slice (tuple index out of range)");
                }
                let s = stft_frame_major(&sp.buf, sr, nper, Some(nper / 2))?;
                sp.vals.extend(key_vals(&s, hz)?);
                continue;
            }
            let ready = n_frames(sp.buf.len(), nper, hop);
            if ready > 0 {
                Self::run_batch(sp, hz, sr, nper, ready)?;
            }
        }
        let mut per: Vec<Vec<f64>> = vec![Vec::new(); self.hz.len()];
        for sp in self.spans {
            per[sp.det].extend(sp.vals);
        }
        per.into_iter().map(key_stats).collect()
    }
}

// ------------------------------------------------------------------ gated notch

/// Margin / crossfade parameters of [`notch_spans`] (with their defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gate {
    /// [s] added before / after each span
    pub margin: f64,
    /// [s] raised-cosine crossfade
    pub fade: f64,
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            margin: 1.0,
            fade: 0.2,
        }
    }
}

/// Sample geometry of one gated span: notch region [a, b), filtered segment [pa, pb).
#[derive(Debug, Clone, Copy, PartialEq)]
struct SpanGeom {
    a: usize,
    b: usize,
    pa: usize,
    pb: usize,
}

/// Spans kept by `notch_spans` (those with `b - a >= 2 nf`), in order, and `nf`.
fn geometry(spans: &[(f64, f64)], sr: u32, n: usize, g: &Gate) -> (Vec<SpanGeom>, usize) {
    let srf = sr as f64;
    let pad = (0.5 * srf) as usize;
    let nf = ((g.fade * srf) as i64).max(1) as usize;
    let n_i = n as i64;
    let mut out = Vec::new();
    for &(t0, t1) in spans {
        let a = (((t0 - g.margin) * srf) as i64).max(0);
        let b = (((t1 + g.margin) * srf) as i64).min(n_i);
        if b - a < 2 * nf as i64 {
            continue;
        }
        let (a, b) = (a as usize, b as usize);
        out.push(SpanGeom {
            a,
            b,
            pa: a.saturating_sub(pad),
            pb: (b + pad).min(n),
        });
    }
    (out, nf)
}

/// Raised-cosine ramp `0.5 - 0.5 cos(pi k / nf)` as float32 (`g` is a float32 array).
fn ramp(nf: usize) -> Vec<f32> {
    (0..nf)
        .map(|k| (0.5 - 0.5 * (std::f64::consts::PI * k as f64 / nf as f64).cos()) as f32)
        .collect()
}

/// Gain of frame `k` (global) in span `s`.
#[inline]
fn gain(s: &SpanGeom, k: usize, n: usize, nf: usize, rp: &[f32]) -> f32 {
    let i = k - s.a;
    let len = s.b - s.a;
    if s.b < n && i >= len - nf {
        rp[nf - 1 - (i - (len - nf))]
    } else if s.a > 0 && i < nf {
        rp[i]
    } else {
        1.0
    }
}

/// `notch_spans(y, sr, hz, q, spans, margin, fade)` on interleaved f32 frames: notch `hz`
/// inside `spans` (None = whole file, `apply_notch`) with raised-cosine crossfades. Each
/// span's filtfilt segment is computed in parallel; spans are then applied in order (a later
/// span overwrites an overlap).
pub fn notch_spans(
    y: &[f32],
    ch: usize,
    sr: u32,
    hz: f64,
    q: f64,
    spans: Option<&[(f64, f64)]>,
    g: &Gate,
) -> Res<Vec<f32>> {
    let mut out = y.to_vec();
    let Some(spans) = spans else {
        correct::apply_notch(&mut out, ch, sr, hz, q)?;
        return Ok(out);
    };
    let n = y.len() / ch;
    let (geo, nf) = geometry(spans, sr, n, g);
    let rp = ramp(nf);
    let zs: Vec<Vec<f32>> = geo
        .par_iter()
        .map(|s| {
            let mut z = y[s.pa * ch..s.pb * ch].to_vec();
            correct::apply_notch(&mut z, ch, sr, hz, q)?;
            Ok(z)
        })
        .collect::<Res<_>>()?;
    for (s, z) in geo.iter().zip(&zs) {
        for k in s.a..s.b {
            let gk = gain(s, k, n, nf, &rp);
            for c in 0..ch {
                let yv = y[k * ch + c];
                let zv = z[(k - s.pa) * ch + c];
                out[k * ch + c] = yv + gk * (zv - yv);
            }
        }
    }
    Ok(out)
}

/// Backward warm-up of a notch filter so that the truncation error is < 1e-35 relative:
/// `max(NOTCH_LOOKAHEAD, ln(1e-35) / ln(pole radius))`.
fn notch_lookahead(a: &[f64; 3]) -> usize {
    let r = a[2].abs().sqrt();
    if r <= 0.0 || r >= 1.0 {
        return NOTCH_LOOKAHEAD;
    }
    let l = (-80.6 / r.ln()).ceil();
    NOTCH_LOOKAHEAD.max(l as usize)
}

struct SpanRun {
    g: SpanGeom,
    filt: Vec<FiltFiltStream>,
    finished: bool,
    /// interleaved z (as f32) for global frames [z0, z0 + len/ch)
    z: VecDeque<f32>,
    z0: usize,
    tmp: Vec<Vec<f64>>,
}

/// Streaming `notch_spans` stage over interleaved f32 blocks (for the pass-2 chain).
///
/// Every span filters its own segment `[pa, pb)` with an exact-edge streaming filtfilt (SciPy
/// odd extension + zi at both segment ends; interior backward pass from a zero state started
/// ≥ 16384 samples ahead, error < 1e-35); frames outside all spans pass through unchanged.
/// Output = [`notch_spans`] on the whole signal (bit-identical in practice: the backward
/// truncation is far below f32 resolution). Latency ≤ ~82 k frames inside spans.
pub struct GatedNotch {
    ch: usize,
    n: usize,
    nf: usize,
    rp: Vec<f32>,
    spans: Vec<SpanRun>,
    pending: VecDeque<f32>,
    head: usize,
    pos: usize,
    out: Vec<f32>,
}

impl GatedNotch {
    /// `n`: total frames of the stream (pass 2: `Resampler::n_out`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ch: usize,
        sr: u32,
        n: usize,
        hz: f64,
        q: f64,
        spans: &[(f64, f64)],
        g: &Gate,
    ) -> Res<Self> {
        let (geo, nf) = geometry(spans, sr, n, g);
        let (b, a) = iirnotch(hz, q, sr as f64)?;
        let la = notch_lookahead(&a);
        let spans = geo
            .into_iter()
            .map(|g| -> Res<SpanRun> {
                Ok(SpanRun {
                    g,
                    filt: (0..ch)
                        .map(|_| FiltFiltStream::new(&b, &a, None, la))
                        .collect::<Res<_>>()?,
                    finished: false,
                    z: VecDeque::new(),
                    z0: g.pa,
                    tmp: vec![Vec::new(); ch],
                })
            })
            .collect::<Res<_>>()?;
        Ok(Self {
            ch,
            n,
            nf,
            rp: ramp(nf),
            spans,
            pending: VecDeque::new(),
            head: 0,
            pos: 0,
            out: Vec::new(),
        })
    }

    /// Feed one interleaved block (frames `[pos, pos + len)`) into the span filters.
    fn feed(&mut self, y: &[f32]) -> Res<()> {
        let ch = self.ch;
        let (p0, p1) = (self.pos, self.pos + y.len() / ch);
        self.spans.par_iter_mut().try_for_each(|s| -> Res<()> {
            let (a, b) = (s.g.pa.max(p0), s.g.pb.min(p1));
            for c in 0..ch {
                s.tmp[c].clear();
            }
            for c in 0..ch {
                if a < b {
                    let xin: Vec<f64> = (a..b).map(|k| y[(k - p0) * ch + c] as f64).collect();
                    let o = &mut s.tmp[c];
                    s.filt[c].push(&xin, o);
                }
                if !s.finished && p1 >= s.g.pb {
                    let o = &mut s.tmp[c];
                    s.filt[c].finish(o)?;
                }
            }
            if !s.finished && p1 >= s.g.pb {
                s.finished = true;
            }
            let m = s.tmp[0].len();
            debug_assert!(s.tmp.iter().all(|t| t.len() == m));
            for j in 0..m {
                for c in 0..ch {
                    s.z.push_back(s.tmp[c][j] as f32);
                }
            }
            Ok(())
        })?;
        self.pending.extend(y.iter().copied());
        self.pos = p1;
        Ok(())
    }

    /// Emit every pending frame whose value is final.
    fn emit(&mut self) {
        let ch = self.ch;
        self.out.clear();
        while self.head < self.pos {
            let k = self.head;
            // last span (in order) containing k overrides earlier ones
            let mut hit = None;
            for (i, s) in self.spans.iter().enumerate() {
                if s.g.a <= k && k < s.g.b {
                    hit = Some(i);
                }
            }
            match hit {
                None => {
                    for _ in 0..ch {
                        self.out.push(self.pending.pop_front().unwrap());
                    }
                }
                Some(i) => {
                    let s = &mut self.spans[i];
                    // drop z before k
                    while s.z0 < k && !s.z.is_empty() {
                        s.z.drain(..ch);
                        s.z0 += 1;
                    }
                    if s.z0 != k || s.z.is_empty() {
                        break; // not filtered yet
                    }
                    let gk = gain(&s.g, k, self.n, self.nf, &self.rp);
                    for c in 0..ch {
                        let yv = self.pending.pop_front().unwrap();
                        let zv = s.z[c];
                        self.out.push(yv + gk * (zv - yv));
                    }
                }
            }
            self.head += 1;
        }
        // release z that no longer can be used
        for s in &mut self.spans {
            while s.z0 < self.head && !s.z.is_empty() {
                s.z.drain(..ch);
                s.z0 += 1;
            }
        }
    }

    /// Feed an interleaved block; returns the frames that became final.
    pub fn push(&mut self, y: &[f32]) -> Res<&[f32]> {
        if self.pos + y.len() / self.ch > self.n {
            return verr("GatedNotch: more frames than announced");
        }
        self.feed(y)?;
        self.emit();
        Ok(&self.out)
    }

    /// End of stream (all `n` frames pushed): remaining frames.
    pub fn finish(&mut self) -> Res<&[f32]> {
        if self.pos != self.n {
            return verr(format!(
                "GatedNotch: stream ended at {} of {} frames",
                self.pos, self.n
            ));
        }
        self.emit();
        if self.head != self.n {
            return verr("GatedNotch: unfiltered frames left");
        }
        Ok(&self.out)
    }
}

/// A gated notch for [`run_gated`]: (hz, q, spans).
pub type GatedSpec = (f64, f64, Vec<(f64, f64)>);

/// Pass 2 with gated notches: correct → `post.notches` (reference + whole-file notches) →
/// each gated notch in order → `post.lowpass` → `sink`, all streaming (same order as the
/// whole-array path: correct(notch) → extra_notches → apply_lowpass). Returns the output frame count.
#[allow(clippy::too_many_arguments)]
pub fn run_gated<E: Send>(
    x: &[f32],
    ch: usize,
    sr: u32,
    t: &[f64],
    ratio: &[f64],
    p: &Params,
    post: &Post,
    gated: &[GatedSpec],
    gate: &Gate,
    mut sink: impl FnMut(&[f32]) -> Result<(), E> + Send,
) -> Result<usize, RunError<E>> {
    let n_out = Resampler::new(x, ch, sr, t, ratio, p)?.n_out();
    let mut stages = gated
        .iter()
        .map(|(hz, q, sp)| GatedNotch::new(ch, sr, n_out, *hz, *q, sp, gate))
        .collect::<Res<Vec<_>>>()?;
    let mut lp = match &post.lowpass {
        Some(l) => Some(PostChain::new(
            ch,
            sr,
            &Post {
                notches: vec![],
                lowpass: Some(l.clone()),
            },
        )?),
        None => None,
    };
    let first = Post {
        notches: post.notches.clone(),
        lowpass: None,
    };
    // pipe a block through the gated stages and the lowpass into the sink
    fn pipe<E>(
        blk: &[f32],
        stages: &mut [GatedNotch],
        lp: &mut Option<PostChain>,
        sink: &mut dyn FnMut(&[f32]) -> Result<(), E>,
        finish: bool,
    ) -> Result<(), RunError<E>> {
        let mut cur: Vec<f32> = blk.to_vec();
        for st in stages.iter_mut() {
            let mut o = st.push(&cur)?.to_vec();
            if finish {
                o.extend_from_slice(st.finish()?);
            }
            cur = o;
        }
        let mut emit = |b: &[f32]| -> Result<(), RunError<E>> {
            if !b.is_empty() {
                sink(b).map_err(RunError::Sink)?;
            }
            Ok(())
        };
        match lp {
            Some(c) => {
                if !cur.is_empty() {
                    emit(c.push(&cur))?; // PostChain::push panics on an empty block
                }
                if finish {
                    emit(c.finish()?)?;
                }
            }
            None => emit(&cur)?,
        }
        Ok(())
    }
    let mut err = None;
    let n = correct::run_arrays(x, ch, sr, t, ratio, p, &first, |blk: &[f32]| {
        pipe(blk, &mut stages, &mut lp, &mut sink, false).map_err(|e| {
            err = Some(e);
        })
    });
    match n {
        Ok(_) => {}
        Err(RunError::Sink(())) => return Err(err.take().expect("stage error")),
        Err(RunError::Value(v)) => return Err(RunError::Value(v)),
    }
    pipe(&[], &mut stages, &mut lp, &mut sink, true)?;
    Ok(n_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nper_and_geometry() {
        assert_eq!(keying_nper(44100), 1024);
        assert_eq!(keying_nper(16000), 512);
        assert_eq!(keying_nper(48000), 1024);
        let (g, nf) = geometry(
            &[(2.0, 3.0), (4.0, 4.1)],
            1000,
            5000,
            &Gate {
                margin: 0.1,
                fade: 0.2,
            },
        );
        assert_eq!(nf, 200);
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].a, g[0].b, g[0].pa, g[0].pb), (1900, 3100, 1400, 3600));
        assert_eq!(mono_mean(&[1.0, 2.0, 3.0, 5.0], 2), vec![1.5, 4.0]);
    }
}
