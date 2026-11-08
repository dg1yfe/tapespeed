//! correct: pass 2 varying-rate resampler, reference notch and
//! lowpass, as a whole-array path ([`correct`], [`apply_notch`], [`apply_lowpass`]) and a
//! streaming chain ([`run`]: correct -> notches -> lowpass -> sink, no full-output buffer).
//!
//! Time map tau = T(t_in) = integral of r (cubic spline of the curve ratio, integrated
//! analytically); T^-1 by Newton x3 from a 200 Hz linear-interpolation guess. Output sample k
//! sits at tau_k = k/sr. Kernel: f32 Kaiser-windowed sinc table (`phases` steps per zero
//! crossing, lerp between steps), stretched by 1/s with s = min(1, r).
//!
//! Numerics: every output sample is computed independently from the time map and the input
//! (f32 table/lerp/scale); the f32 taps are accumulated in 16 interleaved lanes. Chunking
//! and thread count do not change any value, so RAYON_NUM_THREADS=1 and N threads give bit-identical output.
//! Layout: interleaved frames (`x[frame * ch + c]`), as ts-io decodes and encodes.

use std::sync::mpsc::sync_channel;

use rayon::prelude::*;
use ts_dsp::fir::{self, OverlapSave, ValueError};
use ts_dsp::iir::{self, FiltFiltStream, PadType};
use ts_dsp::spline::{PPoly, cubic_spline};
use ts_dsp::stats;

use crate::curve::{Curve, MetaValue};

/// Output samples per parallel work unit.
pub const CHUNK: usize = 65536;
/// Work units per streaming block.
pub const BLOCK_CHUNKS: usize = 8;
/// Frames per post-chain step (bounds the per-channel f64 buffers).
pub const POST_PIECE: usize = 131072;
/// Backward warm-up of the streaming notch (pole radius <= 0.99 -> error < 1e-30).
pub const NOTCH_LOOKAHEAD: usize = 16384;

/// Resampler parameters (`correct()` keyword defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// kernel length at rate 1 (2*half zero crossings)
    pub taps: usize,
    pub phases: usize,
    /// fraction of Nyquist
    pub cutoff: f64,
    pub beta: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            taps: 64,
            phases: 1024,
            cutoff: 0.94,
            beta: 10.0,
        }
    }
}

/// `apply_lowpass(y, sr, fc, width, atten)` parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Lowpass {
    pub fc: f64,
    pub width: f64,
    pub atten: f64,
}

impl Lowpass {
    pub fn new(fc: f64) -> Self {
        Self {
            fc,
            width: 500.0,
            atten: 80.0,
        }
    }
}

/// Post-processing after resampling: whole-file notches `(hz, q)` in order (reference notch
/// first, then `--notch-at`), then an optional lowpass.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Post {
    pub notches: Vec<(f64, f64)>,
    pub lowpass: Option<Lowpass>,
}

/// Error of [`run`]: a SciPy-style ValueError or the sink's own error.
#[derive(Debug)]
pub enum RunError<E> {
    Value(ValueError),
    Sink(E),
}

impl<E> From<ValueError> for RunError<E> {
    fn from(e: ValueError) -> Self {
        RunError::Value(e)
    }
}

impl<E: std::fmt::Display> std::fmt::Display for RunError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Value(e) => write!(f, "{e}"),
            RunError::Sink(e) => write!(f, "{e}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for RunError<E> {}

/// Default reference-notch frequency: meta `abs_ref_true_hz`, else `ref_hz`.
pub fn default_notch_hz(c: &Curve) -> Option<f64> {
    let v = c.meta.get("abs_ref_true_hz").or(c.meta.get("ref_hz"))?;
    match v {
        MetaValue::Float(x) => Some(*x),
        MetaValue::Int(i) => Some(*i as f64),
        MetaValue::Str(s) => s.trim().parse().ok(),
        MetaValue::Bool(_) => None,
    }
}

// ------------------------------------------------------------------ time map

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// `np.arange(start, stop, step)` for floats (numpy fill: start + i*((start+step)-start)).
fn np_arange(start: f64, stop: f64, step: f64) -> Result<Vec<f64>, ValueError> {
    if step == 0.0 {
        return Err(ValueError("division by zero (np.arange step 0)".into()));
    }
    let len = ((stop - start) / step).ceil();
    if !len.is_finite() {
        return Err(ValueError("arange: cannot compute length".into()));
    }
    if len <= 0.0 {
        return Ok(vec![]);
    }
    let len = len as usize;
    let mut v = Vec::with_capacity(len);
    v.push(start);
    if len > 1 {
        let second = start + step;
        v.push(second);
        let delta = second - start;
        for i in 2..len {
            v.push(start + i as f64 * delta);
        }
    }
    Ok(v)
}

/// `time_map(curve, t_end)` -> (T, r) as PPolys; curve end values held up to t_end.
pub fn time_map(t: &[f64], ratio: &[f64], t_end: f64) -> Result<(PPoly, PPoly), ValueError> {
    if t.is_empty() || t.len() != ratio.len() {
        return Err(ValueError(
            "curve t/ratio empty or of different length".into(),
        ));
    }
    let dt = if t.len() > 1 {
        let d: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
        median(&d)
    } else {
        0.005
    };
    let pre = usize::from(t[0] > 0.0);
    let tl = *t.last().unwrap();
    let ext = if tl < t_end + 4.0 * dt {
        np_arange(tl + dt, t_end + 8.0 * dt, dt)?
    } else {
        vec![]
    };
    // exact capacities: no growth copies of curve-sized arrays
    let cap = pre + t.len() + ext.len();
    let mut tt = Vec::with_capacity(cap);
    let mut r = Vec::with_capacity(cap);
    if pre == 1 {
        tt.push(0.0);
        r.push(ratio[0]);
    }
    tt.extend_from_slice(t);
    r.extend_from_slice(ratio);
    let rl = *ratio.last().unwrap();
    r.extend(std::iter::repeat_n(rl, ext.len()));
    tt.extend(ext);
    let t = tt;
    let rs = cubic_spline(&t, &r)?;
    Ok((rs.antiderivative(), rs))
}

/// `np.linspace(0, stop, num)` value i (numpy: i*step, last = stop), not stored.
#[derive(Debug, Clone, Copy)]
struct Linspace {
    step: f64,
    stop: f64,
    num: usize,
}

impl Linspace {
    #[inline]
    fn at(&self, i: usize) -> f64 {
        if i == self.num - 1 {
            self.stop
        } else {
            i as f64 * self.step
        }
    }
}

/// `np.interp(x, xp, fp)` (clamped), xp increasing, fp a linspace (`ts_dsp::stats` kernel).
#[inline]
fn np_interp_lin(x: f64, xp: &[f64], g: &Linspace, hint: &mut usize) -> f64 {
    stats::interp_one_by(x, xp, |j| g.at(j), hint)
}

/// `sinc_table(half, phases, cutoff, beta)`: h(u), u = i/phases, i = 0..half*phases+1 (f32).
pub fn sinc_table(half: usize, phases: usize, cutoff: f64, beta: f64) -> Vec<f32> {
    let i0b = fir::i0(beta);
    (0..half * phases + 2)
        .map(|i| {
            let u = i as f64 / phases as f64;
            if u >= half as f64 {
                return 0.0;
            }
            let q = u / half as f64;
            let a = 1.0 - q * q;
            let a = if a < 0.0 { 0.0 } else { a };
            let w = fir::i0(beta * a.sqrt()) / i0b;
            (cutoff * fir::sinc(cutoff * u) * w) as f32
        })
        .collect()
}

// ------------------------------------------------------------------ resampler

const LANES: usize = 16;

/// Pass-2 resampler bound to one input (interleaved f32, `ch` channels) and one curve.
pub struct Resampler<'a> {
    x: &'a [f32],
    n: usize,
    ch: usize,
    sr: f64,
    tmap: PPoly,
    rs: PPoly,
    t_grid: Linspace,
    tg_t: Vec<f64>,
    n_out: usize,
    m: usize,
    offs: Vec<f32>,
    /// (tab[i], dtab[i]) pairs: one 8-byte gather per tap
    td: Vec<[f32; 2]>,
    last: i32,
    phases: f32,
}

/// Outputs whose time-map inversion is computed together (independent Newton chains interleave).
const SUB: usize = 128;

struct Scratch {
    ui: Vec<i32>,
    uf: Vec<f32>,
    tt: Vec<f32>,
    dd: Vec<f32>,
    h: Vec<f32>,
    hh: Vec<f32>,
    xw: Vec<f32>,
    ti: [f64; SUB],
    frac: [f32; SUB],
    s: [f32; SUB],
    row: [i64; SUB],
}

impl<'a> Resampler<'a> {
    /// Set up as `correct()` does before its chunk loop.
    pub fn new(
        x: &'a [f32],
        ch: usize,
        sr: u32,
        t: &[f64],
        ratio: &[f64],
        p: &Params,
    ) -> Result<Self, ValueError> {
        if ch == 0 || !x.len().is_multiple_of(ch) {
            return Err(ValueError("input is not a whole number of frames".into()));
        }
        let n = x.len() / ch;
        let srf = sr as f64;
        let t_end = n as f64 / srf;
        let (tmap, rs) = time_map(t, ratio, t_end)?;
        let num = 2usize.max((t_end * 200.0) as usize + 2);
        let stop = t_end + 4e-3;
        let step = stop / (num - 1) as f64;
        let t_grid = Linspace { step, stop, num };
        let mut h = 0;
        let tg_t: Vec<f64> = (0..num)
            .map(|i| tmap.eval_hint(t_grid.at(i), &mut h))
            .collect();
        let n_out = (tmap.eval(t_end) * srf).round_ties_even();
        let n_out = if n_out > 0.0 { n_out as usize } else { 0 };
        let half = p.taps / 2;
        let tab = sinc_table(half, p.phases, p.cutoff, p.beta);
        let mut td: Vec<[f32; 2]> = tab.windows(2).map(|w| [w[0], w[1] - w[0]]).collect();
        td.push([tab[tab.len() - 1], 0.0]);
        let last = (tab.len() - 2) as i32;
        let mut h = 0;
        let rmin = (0..num)
            .map(|i| rs.eval_hint(t_grid.at(i), &mut h))
            .fold(f64::INFINITY, f64::min);
        let s_min = 1.0f64.min(rmin);
        let m = (half as f64 / s_min).ceil() as usize + 1;
        let offs = (0..2 * m)
            .map(|j| (j as f64 - m as f64 + 1.0) as f32)
            .collect();
        Ok(Self {
            x,
            n,
            ch,
            sr: srf,
            tmap,
            rs,
            t_grid,
            tg_t,
            n_out,
            m,
            offs,
            td,
            last,
            phases: p.phases as f32,
        })
    }

    /// Number of output frames, `int(round(T(t_end) * sr))`.
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Channels.
    pub fn channels(&self) -> usize {
        self.ch
    }

    /// Half window m (taps per side incl. stretch margin).
    pub fn half_window(&self) -> usize {
        self.m
    }

    /// Time map T and ratio spline r.
    pub fn maps(&self) -> (&PPoly, &PPoly) {
        (&self.tmap, &self.rs)
    }

    fn scratch(&self) -> Scratch {
        let l = 2 * self.m;
        Scratch {
            ui: vec![0; l],
            uf: vec![0.0; l],
            tt: vec![0.0; l],
            dd: vec![0.0; l],
            h: vec![0.0; l],
            hh: vec![0.0; l * self.ch],
            xw: vec![0.0; l * self.ch],
            ti: [0.0; SUB],
            frac: [0.0; SUB],
            s: [0.0; SUB],
            row: [0; SUB],
        }
    }

    /// `invert()` + per-sample kernel parameters for outputs k0..k0+cnt (cnt <= SUB).
    fn positions(&self, k0: usize, cnt: usize, sc: &mut Scratch, gh: &mut usize) {
        let ti = &mut sc.ti[..cnt];
        for (j, v) in ti.iter_mut().enumerate() {
            let tau = (k0 + j) as f64 / self.sr;
            *v = np_interp_lin(tau, &self.tg_t, &self.t_grid, gh);
        }
        for _ in 0..3 {
            let mut ph = 0usize;
            for (j, v) in ti.iter_mut().enumerate() {
                let tau = (k0 + j) as f64 / self.sr;
                let (tv, rv) = match self.tmap.locate(*v, ph) {
                    Some(i) => {
                        ph = i;
                        (self.tmap.eval_piece(i, *v), self.rs.eval_piece(i, *v))
                    }
                    None => (f64::NAN, f64::NAN),
                };
                *v -= (tv - tau) / rv;
            }
        }
        let mut ph = 0usize;
        for (j, &t) in ti.iter().enumerate() {
            let r = match self.rs.locate(t, ph) {
                Some(i) => {
                    ph = i;
                    self.rs.eval_piece(i, t)
                }
                None => f64::NAN,
            };
            let p = t * self.sr;
            let base = p.floor();
            sc.frac[j] = (p - base) as f32;
            sc.s[j] = (if r.is_nan() { r } else { r.min(1.0) }) as f32;
            sc.row[j] = (base as i64 + 1).clamp(0, self.n as i64 + 2);
        }
    }

    /// Output frames [k0, k0 + out.len()/ch) into `out` (interleaved), sequentially.
    pub fn render(&self, k0: usize, out: &mut [f32]) {
        let ch = self.ch;
        let l = 2 * self.m;
        let mut sc = self.scratch();
        let mut gh = 0usize;
        for (bi, ob) in out.chunks_mut(SUB * ch).enumerate() {
            let kb = k0 + bi * SUB;
            let cnt = ob.len() / ch;
            self.positions(kb, cnt, &mut sc, &mut gh);
            for (j, yo) in ob.chunks_exact_mut(ch).enumerate() {
                let (frac, s, row) = (sc.frac[j], sc.s[j], sc.row[j]);
                // kernel taps: u = |offs - frac| * s * phases, table index + lerp fraction
                let sp = s * self.phases;
                for ((ui, uf), &o) in sc.ui.iter_mut().zip(sc.uf.iter_mut()).zip(&self.offs) {
                    let u = (o - frac).abs() * sp;
                    let i = (u as i32).min(self.last);
                    *ui = i;
                    *uf = u - i as f32;
                }
                for ((t, d), &i) in sc.tt.iter_mut().zip(sc.dd.iter_mut()).zip(&sc.ui) {
                    let [a, b] = self.td[i as usize];
                    *t = a;
                    *d = b;
                }
                for (((h, &t), &d), &u) in sc.h.iter_mut().zip(&sc.tt).zip(&sc.dd).zip(&sc.uf) {
                    *h = (t + u * d) * s;
                }
                // window: xp[row + j] = x[row + j - m] (zeros outside)
                let i0 = row - self.m as i64;
                let xs: &[f32] = if i0 >= 0 && (i0 as usize + l) <= self.n {
                    let a = i0 as usize * ch;
                    &self.x[a..a + l * ch]
                } else {
                    for jj in 0..l {
                        let f = i0 + jj as i64;
                        let ok = f >= 0 && (f as usize) < self.n;
                        for c in 0..ch {
                            sc.xw[jj * ch + c] = if ok { self.x[f as usize * ch + c] } else { 0.0 };
                        }
                    }
                    &sc.xw
                };
                match ch {
                    1 => yo[0] = reduce_lanes(&dot_lanes(&sc.h, xs), 0, 1),
                    2 => {
                        let (a0, a1) = dot2(&sc.h, xs);
                        yo[0] = a0;
                        yo[1] = a1;
                    }
                    _ if LANES.is_multiple_of(ch) => {
                        for (hc, &h) in sc.hh.chunks_exact_mut(ch).zip(&sc.h) {
                            hc.fill(h);
                        }
                        let acc = dot_lanes(&sc.hh, xs);
                        for (c, yv) in yo.iter_mut().enumerate() {
                            *yv = reduce_lanes(&acc, c, ch);
                        }
                    }
                    _ => {
                        for (c, yv) in yo.iter_mut().enumerate() {
                            let mut a = [0.0f32; 4];
                            for jj in 0..l {
                                a[jj % 4] += sc.h[jj] * xs[jj * ch + c];
                            }
                            *yv = (a[0] + a[1]) + (a[2] + a[3]);
                        }
                    }
                }
            }
        }
    }

    /// Like [`render`](Self::render), split into [`CHUNK`]-frame units run with rayon.
    pub fn render_par(&self, k0: usize, out: &mut [f32]) {
        out.par_chunks_mut(CHUNK * self.ch)
            .enumerate()
            .for_each(|(i, o)| self.render(k0 + i * CHUNK, o));
    }
}

#[inline]
fn dot_lanes(h: &[f32], x: &[f32]) -> [f32; LANES] {
    let mut acc = [0.0f32; LANES];
    let hc = h.chunks_exact(LANES);
    let xc = x.chunks_exact(LANES);
    let (hr, xr) = (hc.remainder(), xc.remainder());
    for (a, b) in hc.zip(xc) {
        for l in 0..LANES {
            acc[l] += a[l] * b[l];
        }
    }
    for (l, (a, b)) in hr.iter().zip(xr).enumerate() {
        acc[l] += a * b;
    }
    acc
}

/// Stereo dot product over interleaved x (8 lanes per channel, fixed reduction order).
#[inline]
fn dot2(h: &[f32], x: &[f32]) -> (f32, f32) {
    let mut a0 = [0.0f32; 8];
    let mut a1 = [0.0f32; 8];
    let hc = h.chunks_exact(8);
    let xc = x.chunks_exact(16);
    let (hr, xr) = (hc.remainder(), xc.remainder());
    for (a, b) in hc.zip(xc) {
        for l in 0..8 {
            a0[l] += a[l] * b[2 * l];
            a1[l] += a[l] * b[2 * l + 1];
        }
    }
    for (l, (a, b)) in hr.iter().zip(xr.chunks_exact(2)).enumerate() {
        a0[l] += a * b[0];
        a1[l] += a * b[1];
    }
    let r = |v: [f32; 8]| ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
    (r(a0), r(a1))
}

/// Sum lanes c, c+ch, ... in a fixed pairwise order.
#[inline]
fn reduce_lanes(acc: &[f32; LANES], c: usize, ch: usize) -> f32 {
    let mut v = [0.0f32; LANES];
    let k = LANES / ch;
    for i in 0..k {
        v[i] = acc[c + i * ch];
    }
    let mut w = k;
    while w > 1 {
        let h = w / 2;
        for i in 0..h {
            v[i] = v[2 * i] + v[2 * i + 1];
        }
        w = h;
    }
    v[0]
}

// ------------------------------------------------------------------ whole-array path

/// `correct(x, sr, curve, taps, phases, cutoff, beta)` without notch: float32 (n_out, ch).
pub fn correct(
    x: &[f32],
    ch: usize,
    sr: u32,
    t: &[f64],
    ratio: &[f64],
    p: &Params,
) -> Result<Vec<f32>, ValueError> {
    let r = Resampler::new(x, ch, sr, t, ratio, p)?;
    let mut y = vec![0.0f32; r.n_out() * ch];
    r.render_par(0, &mut y);
    Ok(y)
}

fn map_channels(
    y: &mut [f32],
    ch: usize,
    f: impl Fn(&[f64]) -> Result<Vec<f64>, ValueError> + Sync,
) -> Result<(), ValueError> {
    let cols: Vec<Vec<f64>> = (0..ch)
        .into_par_iter()
        .map(|c| {
            f(&y.iter()
                .skip(c)
                .step_by(ch)
                .map(|&v| v as f64)
                .collect::<Vec<_>>())
        })
        .collect::<Result<_, _>>()?;
    for (c, col) in cols.iter().enumerate() {
        for (i, v) in col.iter().enumerate() {
            y[i * ch + c] = *v as f32;
        }
    }
    Ok(())
}

/// `apply_notch(y, sr, f0, q)`: zero-phase IIR notch (filtfilt in f64, back to f32), in place.
pub fn apply_notch(y: &mut [f32], ch: usize, sr: u32, f0: f64, q: f64) -> Result<(), ValueError> {
    let (b, a) = iir::iirnotch(f0, q, sr as f64)?;
    map_channels(y, ch, |v| iir::filtfilt(&b, &a, v, PadType::Odd, None))
}

/// FIR of `apply_lowpass`: kaiserord(atten, width/(sr/2)), odd length, firwin(kaiser, fs=sr).
pub fn lowpass_fir(sr: u32, lp: &Lowpass) -> Result<Vec<f64>, ValueError> {
    let fs = sr as f64;
    let (n, beta) = fir::kaiserord(lp.atten, lp.width / (fs / 2.0))?;
    fir::firwin_kaiser(n | 1, lp.fc, beta, fs)
}

/// `apply_lowpass(y, sr, fc, width, atten)`: linear-phase Kaiser FIR, 'same' alignment, in place.
pub fn apply_lowpass(y: &mut [f32], ch: usize, sr: u32, lp: &Lowpass) -> Result<(), ValueError> {
    let h = lowpass_fir(sr, lp)?;
    map_channels(y, ch, |v| Ok(fir::fftconvolve_same(v, &h)))
}

/// Whole-array equivalent of [`run`]: correct, then each notch, then the lowpass.
pub fn correct_post(
    x: &[f32],
    ch: usize,
    sr: u32,
    t: &[f64],
    ratio: &[f64],
    p: &Params,
    post: &Post,
) -> Result<Vec<f32>, ValueError> {
    let mut y = correct(x, ch, sr, t, ratio, p)?;
    for &(hz, q) in &post.notches {
        apply_notch(&mut y, ch, sr, hz, q)?;
    }
    if let Some(lp) = &post.lowpass {
        apply_lowpass(&mut y, ch, sr, lp)?;
    }
    Ok(y)
}

// ------------------------------------------------------------------ streaming chain

/// One channel: notch stages then lowpass, with per-stage buffers reused across blocks.
struct ChanChain {
    notches: Vec<FiltFiltStream>,
    lp: Option<OverlapSave>,
    /// bufs[0] = input, bufs[i + 1] = output of stage i
    bufs: Vec<Vec<f64>>,
}

impl ChanChain {
    fn run_stages(&mut self, finish: bool) -> Result<(), ValueError> {
        for i in 0..self.notches.len() {
            let (a, b) = self.bufs.split_at_mut(i + 1);
            let o = &mut b[0];
            o.clear();
            self.notches[i].push(&a[i], o);
            if finish {
                self.notches[i].finish(o)?;
            }
            for v in o.iter_mut() {
                *v = *v as f32 as f64; // apply_notch returns float32
            }
        }
        if let Some(lp) = &mut self.lp {
            let k = self.notches.len();
            let (a, b) = self.bufs.split_at_mut(k + 1);
            let o = &mut b[0];
            o.clear();
            lp.push(&a[k], o);
            if finish {
                lp.finish(o);
            }
        }
        Ok(())
    }

    fn output(&self) -> &[f64] {
        self.bufs.last().unwrap()
    }
}

/// Per-channel streaming post chain over interleaved f32 blocks.
pub struct PostChain {
    ch: usize,
    chans: Vec<ChanChain>,
    out: Vec<f32>,
}

impl PostChain {
    pub fn new(ch: usize, sr: u32, post: &Post) -> Result<Self, ValueError> {
        let mut coef = Vec::new();
        for &(hz, q) in &post.notches {
            coef.push(iir::iirnotch(hz, q, sr as f64)?);
        }
        let h = match &post.lowpass {
            Some(lp) => Some(lowpass_fir(sr, lp)?),
            None => None,
        };
        let nb = 1 + coef.len() + usize::from(h.is_some());
        let chans = (0..ch)
            .map(|_| -> Result<ChanChain, ValueError> {
                Ok(ChanChain {
                    notches: coef
                        .iter()
                        .map(|(b, a)| FiltFiltStream::new(b, a, None, NOTCH_LOOKAHEAD))
                        .collect::<Result<_, _>>()?,
                    lp: h.as_ref().map(|h| OverlapSave::new(h)),
                    bufs: vec![Vec::new(); nb],
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            ch,
            chans,
            out: Vec::new(),
        })
    }

    fn interleave(&mut self) -> &[f32] {
        let n = self.chans[0].output().len();
        assert!(self.chans.iter().all(|c| c.output().len() == n));
        self.out.clear();
        if n == 0 {
            return &self.out;
        }
        self.out.resize(n * self.ch, 0.0);
        for (c, cc) in self.chans.iter().enumerate() {
            for (o, v) in self.out[c..].iter_mut().step_by(self.ch).zip(cc.output()) {
                *o = *v as f32;
            }
        }
        &self.out
    }

    /// Feed an interleaved block; returns the interleaved frames that became final.
    pub fn push(&mut self, y: &[f32]) -> &[f32] {
        let ch = self.ch;
        self.chans.par_iter_mut().enumerate().for_each(|(c, cc)| {
            let b = &mut cc.bufs[0];
            b.clear();
            b.extend(y.iter().skip(c).step_by(ch).map(|&v| v as f64)); // skip: empty block safe
            cc.run_stages(false).expect("push never fails");
        });
        self.interleave()
    }

    /// End of stream: remaining frames (exact SciPy tails).
    pub fn finish(&mut self) -> Result<&[f32], ValueError> {
        self.chans.par_iter_mut().try_for_each(|cc| {
            cc.bufs[0].clear();
            cc.run_stages(true)
        })?;
        Ok(self.interleave())
    }
}

fn feed<E>(
    chain: &mut PostChain,
    buf: &[f32],
    sink: &mut dyn FnMut(&[f32]) -> Result<(), E>,
) -> Result<(), RunError<E>> {
    for piece in buf.chunks(POST_PIECE * chain.ch) {
        let out = chain.push(piece);
        if !out.is_empty() {
            sink(out).map_err(RunError::Sink)?;
        }
    }
    Ok(())
}

fn close<E>(
    chain: &mut PostChain,
    sink: &mut dyn FnMut(&[f32]) -> Result<(), E>,
) -> Result<(), RunError<E>> {
    let out = chain.finish()?;
    if !out.is_empty() {
        sink(out).map_err(RunError::Sink)?;
    }
    Ok(())
}

/// Streaming pass 2: correct -> notches -> lowpass -> `sink` (interleaved f32 blocks, in order).
/// Returns the number of output frames. Holds O(block) output memory (buffers recycled).
#[allow(clippy::too_many_arguments)]
pub fn run_arrays<E: Send>(
    x: &[f32],
    ch: usize,
    sr: u32,
    t: &[f64],
    ratio: &[f64],
    p: &Params,
    post: &Post,
    mut sink: impl FnMut(&[f32]) -> Result<(), E> + Send,
) -> Result<usize, RunError<E>> {
    let res = Resampler::new(x, ch, sr, t, ratio, p)?;
    let mut chain = PostChain::new(ch, sr, post)?;
    let n_out = res.n_out();
    let block = CHUNK * BLOCK_CHUNKS;
    let nblk = n_out.div_ceil(block);
    let range = |b: usize| (b * block, ((b + 1) * block).min(n_out));
    if rayon::current_num_threads() <= 1 {
        let mut buf = Vec::with_capacity(block * ch);
        for b in 0..nblk {
            let (k0, k1) = range(b);
            buf.clear();
            buf.resize((k1 - k0) * ch, 0.0);
            res.render_par(k0, &mut buf);
            feed(&mut chain, &buf, &mut sink)?;
        }
        close(&mut chain, &mut sink)?;
        return Ok(n_out);
    }
    // post chain on its own thread, overlapped with resampling of the next block;
    // two buffers circulate (filled -> worker -> back)
    std::thread::scope(|sc| {
        let (tx, rx) = sync_channel::<Vec<f32>>(1);
        let (back_tx, back_rx) = sync_channel::<Vec<f32>>(2);
        for _ in 0..2 {
            back_tx.send(Vec::with_capacity(block * ch)).unwrap();
        }
        let worker = sc.spawn(move || {
            for buf in rx {
                feed(&mut chain, &buf, &mut sink)?;
                let _ = back_tx.send(buf);
            }
            close(&mut chain, &mut sink)
        });
        for b in 0..nblk {
            let Ok(mut buf) = back_rx.recv() else { break };
            let (k0, k1) = range(b);
            buf.clear();
            buf.resize((k1 - k0) * ch, 0.0);
            res.render_par(k0, &mut buf);
            if tx.send(buf).is_err() {
                break;
            }
        }
        drop(tx);
        worker.join().expect("post-chain thread panicked")
    })?;
    Ok(n_out)
}

/// [`run_arrays`] for a [`Curve`] (uses `curve.t`, `curve.ratio`).
pub fn run<E: Send>(
    x: &[f32],
    ch: usize,
    sr: u32,
    curve: &Curve,
    p: &Params,
    post: &Post,
    sink: impl FnMut(&[f32]) -> Result<(), E> + Send,
) -> Result<usize, RunError<E>> {
    run_arrays(x, ch, sr, &curve.t, &curve.ratio, p, post, sink)
}

#[cfg(test)]
mod tests {
    #[test]
    fn post_chain_empty_push_stereo() {
        let post = Post {
            notches: vec![(9000.0, 30.0)],
            lowpass: None,
        };
        let mut chain = PostChain::new(2, 44100, &post).unwrap();
        assert!(chain.push(&[]).is_empty());
        let x: Vec<f32> = (0..2 * 4410).map(|i| (i as f32 * 0.01).sin()).collect();
        let _ = chain.push(&x);
        let _ = chain.push(&[]);
        assert!(chain.finish().is_ok());
    }

    use super::*;

    #[test]
    fn arange_matches_numpy_fill() {
        let v = np_arange(1.0, 1.1, 0.03).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v[1], 1.0 + 0.03);
        assert_eq!(v[3], 1.0 + 3.0 * ((1.0 + 0.03) - 1.0));
        assert!(np_arange(2.0, 1.0, 0.1).unwrap().is_empty());
    }

    #[test]
    fn interp_clamps() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 10.0, 30.0];
        let mut h = 0;
        assert_eq!(stats::interp_one_by(-1.0, &xp, |j| fp[j], &mut h), 0.0);
        assert_eq!(stats::interp_one_by(1.5, &xp, |j| fp[j], &mut h), 20.0);
        assert_eq!(stats::interp_one_by(2.0, &xp, |j| fp[j], &mut h), 30.0);
        assert_eq!(stats::interp_one_by(5.0, &xp, |j| fp[j], &mut h), 30.0);
    }
}
