//! Stage-1 complex baseband shared by analyze, model, evidence and absolute
//! ([`pick`], [`lpf`], [`het_decimate`]).
//!
//! The result is ONE global `resample_poly(xh, 1, D, window=h)` of the heterodyned signal
//! `xh[i] = pick(x)[i] · exp(-2πj·mod(i·f0/sr, 1))`, zero outside `[0, n)`; any chunking with
//! chunk starts at multiples of D and pads covering the FIR support gives the same values.
//! The work is therefore partitioned freely:
//! output blocks (at most [`OUT_BLOCK`] samples, and about [`SPAN`] input samples) run in
//! parallel (rayon), each heterodynes just the input span it needs into SoA scratch (re/im)
//! owned by its pool thread (one slot per thread for the whole call, so the scratch memory is
//! bounded by threads × SPAN × 16 B, not by the number of rayon jobs) and evaluates its outputs with the
//! 4×2-accumulator dot kernel of `ts_dsp::conv` (same summation order as
//! `resample_poly_down_c`). Each output is computed identically for any thread count.
//!
//! NCO: exact per-sample phase: `ph = mod(i·w, 1)` (`a - trunc(a)`, exact like
//! `fmod`), `θ = -2π·ph`, then separate libm `cos`/`sin` ([`cos_sin`]). A recursive phasor
//! was rejected: the defined phase carries the rounding jitter of `i·w`
//! (≈ ulp(i·w), 4e-10 rad at 60 s, 2e-8 rad at 1 h), which a smooth recursion does not
//! reproduce → it would miss the 1e-10 stage-1 tolerance on long files.
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::Mutex;
use ts_dsp::ValueError;
use ts_dsp::fir::{firwin_kaiser, kaiserord};

type Res<T> = Result<T, ValueError>;

fn verr<T>(m: impl Into<String>) -> Res<T> {
    Err(ValueError(m.into()))
}

/// `(cos x, sin x)` as two separate libm calls (as numpy's complex `exp` computes
/// them). LLVM would otherwise fuse a `cos`/`sin` pair into Apple's `__sincos_stret`,
/// whose `sin` differs from libm `sin` in the last bit for some arguments.
#[inline]
pub fn cos_sin(x: f64) -> (f64, f64) {
    (x.cos(), std::hint::black_box(x).sin())
}

/// Channel selection of [`pick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Mid,
    S,
    L,
    R,
}

impl std::str::FromStr for Channel {
    type Err = ValueError;
    fn from_str(s: &str) -> Res<Self> {
        match s {
            "mid" => Ok(Channel::Mid),
            "S" => Ok(Channel::S),
            "L" => Ok(Channel::L),
            "R" => Ok(Channel::R),
            _ => verr(format!("unknown channel {s:?}")),
        }
    }
}

/// `pick(x[a:b], channel)` for interleaved f32 frames (`ch` channels), as f64.
pub fn pick(x: &[f32], ch: usize, a: usize, b: usize, channel: Channel) -> Vec<f64> {
    if ch == 1 {
        return x[a..b].iter().map(|&v| v as f64).collect();
    }
    (a..b)
        .map(|i| {
            let (l, r) = (x[i * ch] as f64, x[i * ch + 1] as f64);
            match channel {
                Channel::Mid => (l + r) / 2.0,
                Channel::S => (l - r) / 2.0,
                Channel::L => l,
                Channel::R => r,
            }
        })
        .collect()
}

/// `lpf(fpass, fstop, fs, atten)`: Kaiser lowpass, odd length, cutoff mid-transition.
pub fn lpf(fpass: f64, fstop: f64, fs: f64, atten: f64) -> Res<Vec<f64>> {
    let (n, beta) = kaiserord(atten, (fstop - fpass) / (fs / 2.0))?;
    firwin_kaiser(n | 1, (fpass + fstop) / 2.0, beta, fs)
}

/// Max outputs per parallel work item.
pub const OUT_BLOCK: usize = 4096;
/// Target input span per work item (bounds the per-thread scratch for large D).
pub const SPAN: usize = 1 << 18;

/// Two dot products sharing `h` (SoA complex signal, real kernel): 4 independent FMA
/// accumulators per part, remainder sequential. Same order as `ts_dsp::conv::dot2`.
#[inline]
fn dot2(h: &[f64], xr: &[f64], xi: &[f64]) -> (f64, f64) {
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

/// numpy `np.mod(a, 1.0)` (fmod, then shifted into `[0, 1)`; exact).
#[inline]
fn mod1(a: f64) -> f64 {
    let r = a - a.trunc();
    if r < 0.0 {
        r + 1.0
    } else if r == 0.0 {
        0.0
    } else {
        r
    }
}

/// Heterodyne absolute frames `lo..hi` into `re`/`im`: `pick(x)[i] · exp(-2πj·mod(i·w, 1))`.
/// `x[0]` holds absolute frame `x0` (0 for a whole signal, the window start for a stream).
#[allow(clippy::too_many_arguments)]
fn heterodyne(
    x: &[f32],
    x0: usize,
    ch: usize,
    channel: Channel,
    w: f64,
    lo: usize,
    hi: usize,
    re: &mut Vec<f64>,
    im: &mut Vec<f64>,
) {
    re.clear();
    im.clear();
    re.reserve_exact(hi - lo);
    im.reserve_exact(hi - lo);
    let tau = std::f64::consts::TAU;
    let mut put = |i: usize, v: f64| {
        let th = -tau * mod1(i as f64 * w);
        let (c, s) = cos_sin(th);
        re.push(v * c);
        im.push(v * s);
    };
    let (a, b) = (lo - x0, hi - x0);
    if ch == 1 {
        for (i, &v) in (lo..hi).zip(&x[a..b]) {
            put(i, v as f64);
        }
        return;
    }
    for (i, f) in (lo..hi).zip(x[a * ch..b * ch].chunks_exact(ch)) {
        let (l, r) = (f[0] as f64, f[1] as f64);
        let v = match channel {
            Channel::Mid => (l + r) / 2.0,
            Channel::S => (l - r) / 2.0,
            Channel::L => l,
            Channel::R => r,
        };
        put(i, v);
    }
}

/// `het_decimate(x, sr, f0, D, h, channel)`: complex heterodyne at `f0` and
/// `resample_poly(·, 1, D, window=h)`; `y[k] <-> x[k*D]`, `len = ceil(n / D)`. `x` holds
/// interleaved f32 frames with `ch` channels. See the module doc for the partitioning.
pub fn het_decimate(
    x: &[f32],
    ch: usize,
    sr: f64,
    f0: f64,
    d: usize,
    h: &[f64],
    channel: Channel,
) -> Res<Vec<Complex64>> {
    if d < 1 {
        return verr("up and down must be >= 1");
    }
    if h.is_empty() {
        return verr("window must be non-empty");
    }
    let n = x.len() / ch;
    let w = f0 / sr;
    if n == 0 {
        return Ok(vec![]);
    }
    if d == 1 {
        // scipy resample_poly short-circuits up == down == 1: unfiltered copy
        let (mut re, mut im) = (vec![], vec![]);
        heterodyne(x, 0, ch, channel, w, 0, n, &mut re, &mut im);
        return Ok(re
            .iter()
            .zip(&im)
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect());
    }
    let m = h.len();
    let half = (m - 1) / 2;
    let hr: Vec<f64> = h.iter().rev().copied().collect();
    let mut out = vec![Complex64::new(0.0, 0.0); n.div_ceil(d)];
    // block size changes the partition only; every output is computed identically
    let blk_n = (SPAN / d).clamp(64, OUT_BLOCK);
    let slots: Vec<Mutex<(Vec<f64>, Vec<f64>)>> = (0..rayon::current_num_threads())
        .map(|_| Mutex::new((Vec::new(), Vec::new())))
        .collect();
    out.par_chunks_mut(blk_n).enumerate().for_each(|(b, blk)| {
        let slot = rayon::current_thread_index().map(|i| &slots[i % slots.len()]);
        // a slot is only held inside this closure, which never yields to rayon; try_lock is
        // a safety net (a foreign caller pool) that falls back to a local buffer
        let mut local = (Vec::new(), Vec::new());
        let mut guard = slot.and_then(|s| s.try_lock().ok());
        let (re, im) = match guard.as_deref_mut() {
            Some(g) => (&mut g.0, &mut g.1),
            None => (&mut local.0, &mut local.1),
        };
        let q0 = b * blk_n;
        // full-convolution index of output q: k = half + d*q, taps x[k-(m-1) ..= k]
        let k_first = half + d * q0;
        let k_last = half + d * (q0 + blk.len() - 1);
        let lo = k_first.saturating_sub(m - 1);
        let hi = (k_last + 1).min(n);
        heterodyne(x, 0, ch, channel, w, lo, hi, re, im);
        outputs(blk, &hr, d, k_first, lo, n, re, im);
    });
    Ok(out)
}

/// Outputs `k = k_first + d·q` (q over `blk`) from the heterodyned span starting at absolute
/// frame `lo`, for a signal of `n` frames (zero outside `[0, n)`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn outputs(
    blk: &mut [Complex64],
    hr: &[f64],
    d: usize,
    k_first: usize,
    lo: usize,
    n: usize,
    re: &[f64],
    im: &[f64],
) {
    let m = hr.len();
    for (q, o) in blk.iter_mut().enumerate() {
        let k = k_first + d * q;
        // i in [max(0, k-(n-1)), min(m-1, k)]: hr[m-1-ihi ..= m-1-ilo] · x[k-ihi ..= k-ilo]
        let ilo = k.saturating_sub(n - 1);
        let ihi = k.min(m - 1);
        if ilo > ihi {
            continue;
        }
        let (a, j, len) = (m - 1 - ihi, k - ihi - lo, ihi - ilo + 1);
        let (r, i) = dot2(&hr[a..a + len], &re[j..j + len], &im[j..j + len]);
        *o = Complex64::new(r, i);
    }
}

/// Streaming [`het_decimate`]: the input arrives in chunks of interleaved frames (e.g. straight
/// from the decoder), the output grows as soon as its windows are complete. Output `q` needs
/// frames `k-(m-1) ..= k`, `k = half + d·q`; frames below the next window are dropped, so the
/// state is `m - 1` frames plus the not yet emitted tail. Every output is computed exactly as
/// in `het_decimate` (absolute-index NCO, same spans and dot kernel), so the result is
/// bit-identical for any chunking and thread count. Emission is batched (`min_emit` outputs,
/// split over the rayon pool); the last outputs, whose windows reach past the end, are emitted
/// by [`HetStream::finish`] once `n` is known.
pub struct HetStream {
    ch: usize,
    channel: Channel,
    w: f64,
    d: usize,
    hr: Vec<f64>,
    half: usize,
    /// frames from absolute frame `base` on (interleaved)
    buf: Vec<f32>,
    base: usize,
    next_q: usize,
    min_emit: usize,
    par: Option<usize>,
    out: Vec<Complex64>,
}

impl HetStream {
    /// Same arguments as [`het_decimate`] minus the signal.
    pub fn new(ch: usize, sr: f64, f0: f64, d: usize, h: &[f64], channel: Channel) -> Res<Self> {
        if d < 1 {
            return verr("up and down must be >= 1");
        }
        if h.is_empty() {
            return verr("window must be non-empty");
        }
        if ch == 0 {
            return verr("no channels");
        }
        Ok(HetStream {
            ch,
            channel,
            w: f0 / sr,
            d,
            hr: h.iter().rev().copied().collect(),
            half: (h.len() - 1) / 2,
            buf: Vec::new(),
            base: 0,
            next_q: 0,
            min_emit: 16 * OUT_BLOCK,
            par: None,
            out: Vec::new(),
        })
    }

    /// Emit once at least `n` outputs are ready (default `16 · OUT_BLOCK`; any value gives the
    /// same result).
    pub fn min_emit(mut self, n: usize) -> Self {
        self.min_emit = n.max(1);
        self
    }

    /// At most `n` work items of one emission run concurrently (`None`, the default: the pool
    /// size). A stream fed by a decoder leaves the other cores to it; any value gives the same
    /// result, and it may change between pushes.
    pub fn set_par(&mut self, n: Option<usize>) {
        self.par = n.map(|v| v.max(1));
    }

    /// Reserve the output for `frames` input frames (an estimate; the output grows past it).
    pub fn reserve_frames(&mut self, frames: usize) {
        self.out
            .reserve_exact(frames.div_ceil(self.d).saturating_sub(self.out.len()));
    }

    fn avail(&self) -> usize {
        self.base + self.buf.len() / self.ch
    }

    /// Append interleaved frames (a whole number of frames).
    pub fn push(&mut self, frames: &[f32]) {
        debug_assert!(frames.len().is_multiple_of(self.ch), "partial frame");
        self.buf.extend_from_slice(frames);
        let a = self.avail();
        if self.d == 1 {
            // resample_poly short-circuit: unfiltered heterodyne (as het_decimate)
            self.emit_d1(a);
            return;
        }
        // outputs with k = half + d·q <= a - 1 have all their taps
        let q_end = if a > self.half {
            (a - 1 - self.half) / self.d + 1
        } else {
            0
        };
        if q_end >= self.next_q + self.min_emit {
            self.emit(q_end, usize::MAX);
        }
    }

    /// End of input: the remaining outputs (`ceil(n / d)` in total, as `het_decimate`).
    pub fn finish(mut self) -> Vec<Complex64> {
        let n = self.avail();
        if self.d == 1 {
            self.emit_d1(n);
        } else if n > 0 {
            self.emit(n.div_ceil(self.d), n);
        }
        self.out.shrink_to_fit();
        self.out
    }

    fn emit_d1(&mut self, a: usize) {
        let (mut re, mut im) = (vec![], vec![]);
        let lo = self.next_q;
        heterodyne(
            &self.buf,
            self.base,
            self.ch,
            self.channel,
            self.w,
            lo,
            a,
            &mut re,
            &mut im,
        );
        self.out
            .extend(re.iter().zip(&im).map(|(&r, &i)| Complex64::new(r, i)));
        self.next_q = a;
        self.buf.clear();
        self.base = a;
    }

    /// Outputs `next_q .. q_end`. `n`: total frames once known, else `usize::MAX` (then every
    /// window emitted lies inside the signal, as `push` guarantees).
    fn emit(&mut self, q_end: usize, n: usize) {
        if q_end > self.next_q {
            let (m, d, half) = (self.hr.len(), self.d, self.half);
            let q0 = self.next_q;
            let have = self.avail();
            let start = self.out.len();
            self.out
                .resize(start + (q_end - q0), Complex64::new(0.0, 0.0));
            // partition only: every output is computed identically
            // `par` parts run concurrently, each walks its blocks in order
            let par = self.par.unwrap_or_else(rayon::current_num_threads).max(1);
            let cap = (SPAN / d).clamp(64, OUT_BLOCK);
            let blk_n = (q_end - q0).div_ceil(4 * par).clamp(64, cap);
            let part = (q_end - q0).div_ceil(blk_n).div_ceil(par) * blk_n;
            let (buf, base, hr) = (&self.buf, self.base, &self.hr);
            let (ch, channel, w) = (self.ch, self.channel, self.w);
            self.out[start..]
                .par_chunks_mut(part)
                .enumerate()
                .for_each(|(p, pt)| {
                    let (mut re, mut im) = (Vec::new(), Vec::new());
                    for (b, blk) in pt.chunks_mut(blk_n).enumerate() {
                        let k_first = half + d * (q0 + p * part + b * blk_n);
                        let k_last = k_first + d * (blk.len() - 1);
                        let lo = k_first.saturating_sub(m - 1);
                        let hi = (k_last + 1).min(have);
                        heterodyne(buf, base, ch, channel, w, lo, hi, &mut re, &mut im);
                        outputs(blk, hr, d, k_first, lo, n, &re, &im);
                    }
                });
            self.next_q = q_end;
        }
        // drop frames no future window needs
        let keep = (self.half + self.d * self.next_q).saturating_sub(self.hr.len() - 1);
        if keep > self.base {
            let k = ((keep - self.base) * self.ch).min(self.buf.len());
            self.buf.drain(..k);
            self.base += k / self.ch;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ts_dsp::conv::resample_poly_down_c;

    /// The chunked algorithm, literally (reference for the block-partitioned version).
    #[allow(clippy::too_many_arguments)]
    fn het_decimate_chunked(
        x: &[f32],
        ch: usize,
        sr: f64,
        f0: f64,
        d: usize,
        h: &[f64],
        channel: Channel,
        chunk: usize,
    ) -> Vec<Complex64> {
        let n = x.len() / ch;
        let pad = h.len().div_ceil(2 * d) * d + d;
        let chunk = (chunk / d).max(1) * d;
        let w = f0 / sr;
        let mut out = vec![];
        for s in (0..n).step_by(chunk) {
            let (a, b, e) = (
                s.saturating_sub(pad),
                (s + chunk + pad).min(n),
                (s + chunk).min(n),
            );
            let seg = pick(x, ch, a, b, channel);
            let z: Vec<Complex64> = seg
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let ph = ((a + i) as f64 * w) % 1.0;
                    let (c, s) = cos_sin(-std::f64::consts::TAU * ph);
                    Complex64::new(v * c, v * s)
                })
                .collect();
            let y = resample_poly_down_c(&z, d, h).unwrap();
            let k0 = (s - a) / d;
            out.extend_from_slice(&y[k0..k0 + (e - s).div_ceil(d)]);
        }
        out
    }

    fn sig(n: usize, ch: usize) -> Vec<f32> {
        (0..n * ch)
            .map(|i| ((i as f64 * 0.37).sin() * 0.5 + ((i * 7919) % 1000) as f64 * 1e-4) as f32)
            .collect()
    }

    #[test]
    fn global_equals_chunked_bitwise() {
        let h = lpf(400.0, 1200.0, 48000.0, 90.0).unwrap();
        for &(n, ch, d, chunk, channel) in &[
            (100_000usize, 2usize, 29usize, 1usize << 14, Channel::Mid),
            (100_003, 1, 20, 1 << 13, Channel::L),
            (77_777, 2, 7, 1 << 12, Channel::S),
            (5_000, 2, 29, 1 << 21, Channel::R),
            // large D: blocks of SPAN / D < OUT_BLOCK outputs, several of them
            (1_000_003, 2, 117, 1 << 21, Channel::Mid),
            (700_001, 1, 4099, 1 << 18, Channel::L),
            (300, 1, 29, 1 << 21, Channel::Mid),
            (10, 1, 3, 4, Channel::Mid),
        ] {
            let x = sig(n, ch);
            let a = het_decimate(&x, ch, 48000.0, 9000.3, d, &h, channel).unwrap();
            let b = het_decimate_chunked(&x, ch, 48000.0, 9000.3, d, &h, channel, chunk);
            assert_eq!(a.len(), b.len());
            for (p, q) in a.iter().zip(&b) {
                assert_eq!(
                    (p.re.to_bits(), p.im.to_bits()),
                    (q.re.to_bits(), q.im.to_bits()),
                    "n={n} d={d}"
                );
            }
        }
    }

    #[test]
    fn stream_equals_global_bitwise() {
        let h = lpf(400.0, 1200.0, 48000.0, 90.0).unwrap();
        for &(n, ch, d, chunk, emit, channel) in &[
            (
                100_000usize,
                2usize,
                29usize,
                4096usize,
                1usize,
                Channel::Mid,
            ),
            (100_003, 1, 20, 1, 7, Channel::L),
            (77_777, 2, 7, 333, 100_000, Channel::S),
            (5_000, 2, 29, 1 << 20, 64, Channel::R),
            (1_000_003, 2, 117, 65_536, 16 * OUT_BLOCK, Channel::Mid),
            (300, 1, 29, 7, 1, Channel::Mid),
            (10, 1, 3, 4, 1, Channel::Mid),
            (0, 2, 29, 4, 1, Channel::Mid),
            (5_001, 2, 1, 77, 1, Channel::Mid),
        ] {
            let x = sig(n, ch);
            let a = het_decimate(&x, ch, 48000.0, 9000.3, d, &h, channel).unwrap();
            let mut st = HetStream::new(ch, 48000.0, 9000.3, d, &h, channel)
                .unwrap()
                .min_emit(emit);
            st.reserve_frames(n / 2);
            for (i, c) in x.chunks(chunk * ch).enumerate() {
                st.set_par((i % 3 > 0).then_some(1 + i % 5));
                st.push(c);
            }
            let b = st.finish();
            assert_eq!(a.len(), b.len(), "n={n} d={d}");
            for (p, q) in a.iter().zip(&b) {
                assert_eq!(
                    (p.re.to_bits(), p.im.to_bits()),
                    (q.re.to_bits(), q.im.to_bits()),
                    "n={n} d={d} chunk={chunk}"
                );
            }
        }
    }

    #[test]
    fn d1_is_plain_heterodyne() {
        let x = sig(1000, 1);
        let a = het_decimate(&x, 1, 48000.0, 9000.0, 1, &[1.0, 2.0, 1.0], Channel::Mid).unwrap();
        let b = het_decimate_chunked(
            &x,
            1,
            48000.0,
            9000.0,
            1,
            &[1.0, 2.0, 1.0],
            Channel::Mid,
            1 << 21,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn mod1_matches_fmod() {
        for &a in &[
            0.0,
            1.0,
            3.75,
            1e6 + 0.123,
            123456.0 * 0.1875,
            0.999_999_999_999_9,
        ] {
            assert_eq!(mod1(a).to_bits(), (a % 1.0).to_bits());
        }
        assert_eq!(mod1(-0.25), 0.75);
    }
}
