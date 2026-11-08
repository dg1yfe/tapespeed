//! `--abs pair`: both adjacent-channel beats on the speed-normalised spectrum ([`Normalised`],
//! [`floor`], [`peak`], [`comb`], [`pair_measure`], [`pair_table`], [`pair_info`]).
//!
//! AM station at `fc = raster + d` between neighbours on the exact raster: beats `HZ + d` and
//! `HZ - d`, true midpoint `HZ` for any `d`; apparent frequencies `F = s * f_true`, so
//! `f_true = 2 HZ F_p / (F_p + F_q)` (ratios only: level errors of the normalising curve cancel).
//!
//! Numerics: the PSD is float32 (segment spectra are stored as float32, so the file / block
//! means, floors, logs and parabolic offsets run in float32), complex products
//! use numpy's fused NEON kernel (`re = fma(ar, br, -ai*bi)`, `im = fma(ar, bi, ai*br)`), `|z|`
//! numpy's SIMD `larger * sqrt(fma(r, r, 1))`. FFTs (rustfft vs pocketfft) differ at 1e-16
//! relative; segments run in parallel, means are accumulated in segment order (deterministic).

use std::sync::Arc;

use indexmap::IndexMap;
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use ts_dsp::ndimage::{Mode, median_filter};
use ts_dsp::peaks::find_peaks;
use ts_dsp::spectral::{fftfreq, fftshift};
use ts_dsp::stats::{interp_one_by, mean, median, ptp, std};
use ts_dsp::window::hanning;
use ts_io::{fmt_fixed, fmt_general, np_round, repr_shortest};

use super::{AbsError, Info};
use crate::curve::{Curve, MetaValue};
use crate::hetdec::{Channel, het_decimate, lpf};

/// |d| range [Hz true].
pub const PAIR_DMAX: f64 = 300.0;
pub const PAIR_DMIN: f64 = 5.0;
/// [dB] candidate: peak / median over +-10 Hz.
pub const PAIR_PROM: f64 = 10.0;
/// Partner needs >= 15 dB and presence >= 0.5.
pub const PAIR_ACCEPT: f64 = 15.0;
pub const PAIR_PRES: f64 = 0.5;
/// [dB] presence in a block.
pub const PAIR_BLK_PROM: f64 = 6.0;
/// [dB] runner-up within this of the best (and present) -> ambiguous.
pub const PAIR_MARGIN: f64 = 6.0;
/// Cross-check line agreement that resolves an ambiguity (0.01 %).
pub const PAIR_XTOL: f64 = 1e-4;
/// Comb spacings [Hz true] that mark a hum-modulated carrier.
pub const PAIR_HUM: [f64; 4] = [50.0, 60.0, 100.0, 120.0];
/// Modulation-match scan: warp with k x the measured variation.
pub const KS: [f64; 4] = [0.0, 0.5, 1.0, 1.5];

type Res<T> = Result<T, AbsError>;

// ------------------------------------------------------------------ numpy kernels

/// numpy complex multiply (NEON loop, fused).
#[inline]
fn cmul_np(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(
        a.re.mul_add(b.re, -(a.im * b.im)),
        a.re.mul_add(b.im, a.im * b.re),
    )
}

/// numpy complex `abs` (SIMD kernel): `larger * sqrt(fma(r, r, 1))`, `r = smaller / larger`.
#[inline]
fn cabs_np(z: Complex64) -> f64 {
    let (re, im) = (z.re.abs(), z.im.abs());
    if re.is_infinite() || im.is_infinite() {
        return f64::INFINITY;
    }
    if re.is_nan() || im.is_nan() {
        return f64::NAN;
    }
    let (l, s) = if re >= im { (re, im) } else { (im, re) };
    let ratio = if l == 0.0 { 0.0 } else { s / l };
    ratio.mul_add(ratio, 1.0).sqrt() * l
}

/// `np.argmax` of float32 values (first maximum, first NaN wins).
fn argmax32(v: &[f32]) -> usize {
    let mut bi = 0;
    for (i, &x) in v.iter().enumerate() {
        if x.is_nan() {
            return i;
        }
        if x > v[bi] {
            bi = i;
        }
    }
    bi
}

/// `interp4(z, p)`: 4-point Lagrange interpolation of uniform `z` at fractional index `p`.
#[inline]
fn interp4(z: &[Complex64], q: f64) -> Complex64 {
    let i = (q.floor() as i64).clamp(1, z.len() as i64 - 3) as usize;
    let m = q - i as f64;
    let c0 = -m * (m - 1.0) * (m - 2.0) / 6.0;
    let c1 = (m + 1.0) * (m - 1.0) * (m - 2.0) / 2.0;
    let c2 = (m + 1.0) * m * (m - 2.0) / 2.0;
    let c3 = (m + 1.0) * m * (m - 1.0) / 6.0;
    let (a, b, c, d) = (z[i - 1], z[i], z[i + 1], z[i + 2]);
    Complex64::new(
        c0 * a.re + c1 * b.re - c2 * c.re + c3 * d.re,
        c0 * a.im + c1 * b.im - c2 * c.im + c3 * d.im,
    )
}

// ------------------------------------------------------------------ Normalised

/// `Spectra`: frequencies [Hz apparent mean] of the kept bins, file PSD, block PSDs `(t0, PSD)`.
#[derive(Debug, Clone)]
pub struct Spectra {
    pub f: Vec<f64>,
    pub p: Vec<f32>,
    pub blocks: Vec<(f64, Vec<f32>)>,
}

/// Baseband of `x` around `fc` (+-half Hz), sampled on the speed-normalised time of curve `c`
/// (k x its variation).
pub struct Normalised<'a> {
    pub fs: f64,
    pub fc: f64,
    pub z: Vec<Complex64>,
    c: &'a Curve,
}

impl<'a> Normalised<'a> {
    /// `Normalised(x, sr, fc, half, c, channel)`; `x` interleaved frames of `ch` channels.
    pub fn new(
        x: &[f32],
        ch: usize,
        sr: f64,
        fc: f64,
        half: f64,
        c: &'a Curve,
        channel: Channel,
    ) -> Res<Self> {
        let d = ((sr / (6.0 * half)) as i64).max(1) as usize;
        let fs = sr / d as f64;
        let h = lpf(
            1.05 * half,
            (sr / d as f64 - 1.05 * half).min(1.6 * half),
            sr,
            90.0,
        )?;
        let z = het_decimate(x, ch, sr, fc, d, &h, channel)?;
        Ok(Normalised { fs, fc, z, c })
    }

    /// `warp(k)`: `z` resampled on `tau = int u dt`, `u = 1 + k (r / mean r - 1)`, mixed back by
    /// the time shift so lines sit at their mean apparent frequency.
    pub fn warp(&self, k: f64) -> Vec<Complex64> {
        let (t, r) = (&self.c.t, &self.c.ratio);
        let rm = mean(r);
        let u: Vec<f64> = r.iter().map(|&v| 1.0 + k * (v / rm - 1.0)).collect();
        let mut tg = Vec::with_capacity(t.len());
        tg.push(0.0);
        let mut acc = 0.0;
        for i in 1..t.len() {
            acc += (u[i] + u[i - 1]) / 2.0 * (t[i] - t[i - 1]);
            tg.push(acc);
        }
        let fs = self.fs;
        let tz_end = (self.z.len() - 1) as f64 / fs;
        let ntau = (interp_one_by(tz_end, t, |j| tg[j], &mut 0) * fs) as i64;
        let ntau = ntau.max(0) as usize;
        let w = -std::f64::consts::TAU * self.fc;
        const CH: usize = 1 << 16;
        let mut out = vec![Complex64::new(0.0, 0.0); ntau];
        out.par_chunks_mut(CH).enumerate().for_each(|(ci, dst)| {
            let mut hint = 0usize;
            for (o, v) in dst.iter_mut().enumerate() {
                let tau = (ci * CH + o) as f64 / fs;
                let tm = interp_one_by(tau, &tg, |j| t[j], &mut hint);
                let y = w * (tau - tm);
                *v = cmul_np(interp4(&self.z, tm * fs), Complex64::new(y.cos(), y.sin()));
            }
        });
        out
    }

    /// `spectra(k, seg_s, blk_s, half)`: Hann segments (50 %), file PSD (float32 mean), block
    /// PSDs over `blk_s`-s blocks (>= 3 segments each).
    pub fn spectra(&self, k: f64, seg_s: f64, blk_s: Option<f64>, half: Option<f64>) -> Spectra {
        let z = self.warp(k);
        let fs = self.fs;
        let n = (seg_s * fs) as usize;
        let f = fftshift(&fftfreq(n, 1.0 / fs));
        let hl = match half {
            Some(h) if h != 0.0 => h,
            _ => f64::INFINITY,
        };
        // fftshift index j <- unshifted (j + n - n/2) % n
        let sel: Vec<usize> = (0..n).filter(|&j| f[j].abs() <= hl).collect();
        let fsel: Vec<f64> = sel.iter().map(|&j| f[j] + self.fc).collect();
        let src: Vec<usize> = sel.iter().map(|&j| (j + n - n / 2) % n).collect();
        let win = hanning(n);
        let hop = n / 2;
        let st: Vec<usize> = if z.len() >= n && hop > 0 {
            (0..=z.len() - n).step_by(hop).collect()
        } else {
            vec![]
        };
        let ns = sel.len();
        let fft: Arc<dyn Fft<f64>> = FftPlanner::new().plan_fft_forward(n);
        let mut rows = vec![0f32; st.len() * ns];
        rows.par_chunks_mut(ns.max(1) * 8)
            .enumerate()
            .for_each(|(ci, dst)| {
                let mut buf = vec![Complex64::new(0.0, 0.0); n];
                let mut scratch = vec![Complex64::new(0.0, 0.0); fft.get_inplace_scratch_len()];
                for (ri, row) in dst.chunks_mut(ns.max(1)).enumerate() {
                    let s0 = st[ci * 8 + ri];
                    for (b, (zz, ww)) in buf.iter_mut().zip(z[s0..s0 + n].iter().zip(&win)) {
                        *b = Complex64::new(zz.re * ww, zz.im * ww);
                    }
                    fft.process_with_scratch(&mut buf, &mut scratch);
                    for (o, &j) in row.iter_mut().zip(&src) {
                        let a = cabs_np(buf[j]);
                        *o = (a * a) as f32;
                    }
                }
            });
        let mean_rows = |idx: &mut dyn Iterator<Item = usize>| -> Vec<f32> {
            let mut acc: Option<Vec<f32>> = None;
            let mut cnt = 0usize;
            for i in idx {
                let row = &rows[i * ns..(i + 1) * ns];
                match acc.as_mut() {
                    None => acc = Some(row.to_vec()),
                    Some(a) => a.iter_mut().zip(row).for_each(|(x, &y)| *x += y),
                }
                cnt += 1;
            }
            match acc {
                Some(a) => a.into_iter().map(|v| v / cnt as f32).collect(),
                None => vec![f32::NAN; ns],
            }
        };
        let p = mean_rows(&mut (0..st.len()));
        let mut blocks = Vec::new();
        if let Some(b) = blk_s.filter(|&b| b != 0.0) {
            let lz = z.len();
            let nb = ((lz as f64 / fs / b.max(2.0 * seg_s)) as usize).max(1);
            for j in 0..nb {
                let lo = (j * lz) as f64 / nb as f64;
                let hi = ((j + 1) * lz) as f64 / nb as f64 + (n / 2) as f64;
                let m: Vec<usize> = (0..st.len())
                    .filter(|&i| st[i] as f64 >= lo && (st[i] + n) as f64 <= hi)
                    .collect();
                if m.len() >= 3 {
                    blocks.push((st[m[0]] as f64 / fs, mean_rows(&mut m.iter().copied())));
                }
            }
        }
        Spectra { f: fsel, p, blocks }
    }
}

// ------------------------------------------------------------------ floor / peak / comb

/// `floor(P, df, ring)`: running median over +-ring Hz (float32) + 1e-30.
pub fn floor(p: &[f32], df: f64, ring: f64) -> Vec<f32> {
    let size = ((2.0 * ring / df) as usize) | 1;
    let v: Vec<f64> = p.iter().map(|&x| x as f64).collect();
    median_filter(&v, size, Mode::Nearest)
        .expect("median_filter size >= 1")
        .into_iter()
        .map(|m| m as f32 + 1e-30f32)
        .collect()
}

/// `peak(f, P, N, f0, tol)`: strongest bin within `f0 +- tol` → (parabolic frequency,
/// prominence dB) or (NaN, -inf).
pub fn peak(f: &[f64], p: &[f32], nf: &[f32], f0: f64, tol: f64) -> (f64, f64) {
    let m: Vec<usize> = (0..f.len()).filter(|&i| (f[i] - f0).abs() <= tol).collect();
    if m.is_empty() {
        return (f64::NAN, f64::NEG_INFINITY);
    }
    let pm: Vec<f32> = m.iter().map(|&i| p[i]).collect();
    let i = m[argmax32(&pm)].clamp(1, p.len() - 2);
    let (a, b, c) = (
        (p[i - 1] + 1e-30f32).ln(),
        (p[i] + 1e-30f32).ln(),
        (p[i + 1] + 1e-30f32).ln(),
    );
    let den = a - 2.0f32 * b + c;
    let off = if den < 0.0 {
        0.5f32 * (a - c) / den
    } else {
        0.0
    };
    (
        f[i] + (off.clamp(-1.0, 1.0) as f64) * (f[1] - f[0]),
        (10.0f32 * (p[i] / nf[i]).log10()) as f64,
    )
}

/// `10 * log10(P / N)` (float32) as float64 for `find_peaks`.
fn prom_db(p: &[f32], nf: &[f32]) -> Vec<f64> {
    p.iter()
        .zip(nf)
        .map(|(&a, &b)| (10.0f32 * (a / b).log10()) as f64)
        .collect()
}

/// `comb(F, fs_, tol)`: spacing if `F` is a member of an equally spaced comb among the line
/// frequencies (`F +- D`, or `F +- D, +- 2D` on one side; `D` = spacing to the nearest
/// neighbour), else None.
pub fn comb(ff: f64, lines: &[f64], tol: f64) -> Option<f64> {
    let o: Vec<f64> = lines
        .iter()
        .copied()
        .filter(|g| (g - ff).abs() > tol)
        .collect();
    if o.is_empty() {
        return None;
    }
    let d = o
        .iter()
        .map(|g| (g - ff).abs())
        .fold(
            f64::INFINITY,
            |a, b| if b < a || b.is_nan() { b } else { a },
        );
    let has = |v: f64| o.iter().any(|g| (g - v).abs() <= tol);
    ((has(ff - d) && has(ff + d))
        || (has(ff + d) && has(ff + 2.0 * d))
        || (has(ff - d) && has(ff - 2.0 * d)))
    .then_some(d)
}

// ------------------------------------------------------------------ pair_measure

/// [`pair_measure`] keyword arguments (with their defaults).
#[derive(Debug, Clone)]
pub struct PairParams {
    /// Channel spacing [Hz].
    pub hz: f64,
    pub dmax: f64,
    pub dmin: f64,
    pub channel: Channel,
    pub seg_s: f64,
    pub blk_s: Option<f64>,
    /// Cross-check lines (nominal Hz); zeros are skipped.
    pub xcheck: Vec<f64>,
    pub play: f64,
    pub smax: f64,
}

impl Default for PairParams {
    fn default() -> Self {
        PairParams {
            hz: 9000.0,
            dmax: PAIR_DMAX,
            dmin: PAIR_DMIN,
            channel: Channel::Mid,
            seg_s: 8.0,
            blk_s: Some(120.0),
            xcheck: vec![],
            play: 1.0,
            smax: 0.1,
        }
    }
}

/// Which side of the primary a line sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Lo,
    Hi,
}

impl Side {
    pub fn as_str(self) -> &'static str {
        match self {
            Side::Lo => "lo",
            Side::Hi => "hi",
        }
    }
}

/// One partner candidate (report entry, keys in insertion order).
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub f: f64,
    pub prom: f64,
    pub s: f64,
    pub l1_pct: f64,
    pub delta: f64,
    pub side: Side,
    /// Rejection reasons (empty = survivor).
    pub why: Vec<String>,
    pub pres: f64,
    pub drift: f64,
    pub k_hat: f64,
    pub gain_db: f64,
    pub k_contrast: f64,
    pub mirror_db: f64,
    /// `(f, prom)` of the co-channel sub-audio line.
    pub cochan: Option<(f64, f64)>,
    pub comb: Option<f64>,
    pub sub_d: Option<(f64, f64)>,
}

/// Cross-check line result (`xcheck[nom]`).
#[derive(Debug, Clone, PartialEq)]
pub struct XCheck {
    pub nom: f64,
    pub f: f64,
    pub prom: f64,
    pub s: Option<f64>,
    pub ppm: Option<f64>,
}

/// The chosen partner (keys added to the report when a best candidate exists).
#[derive(Debug, Clone, PartialEq)]
pub struct PairSolution {
    pub best: usize,
    pub s: f64,
    pub f_lo: f64,
    pub f_hi: f64,
    pub delta: f64,
    /// `lo`: the primary is the lower beat.
    pub side: Side,
    pub f_true: f64,
    pub offset_hz: f64,
    /// Per block `(t0 rounded to 0.1 s, f_true, d)`.
    pub blocks: Vec<(f64, f64, f64)>,
    /// `(blk_ftrue_ptp_ppm, blk_ftrue_std_ppm, blk_delta_ptp)` if >= 2 blocks.
    pub blk_stats: Option<(f64, f64, f64)>,
}

/// `pair_measure` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct PairResult {
    pub hz: f64,
    pub f_primary: f64,
    pub prom_primary: f64,
    pub r0: f64,
    pub candidates: Vec<Candidate>,
    pub xcheck: Vec<XCheck>,
    pub n_blocks: usize,
    pub sub_lines: Vec<(f64, f64)>,
    pub ambiguity: String,
    /// Set if no unambiguous partner.
    pub error: Option<String>,
    pub solution: Option<PairSolution>,
}

impl PairResult {
    /// `f_hi / f_lo` of the solution.
    pub fn rho(&self) -> Option<f64> {
        self.solution.as_ref().map(|s| s.f_hi / s.f_lo)
    }

    /// The report entry as a JSON value (`rep["pair"]`, key order is part of the format).
    pub fn to_value(&self) -> ts_io::json::Value {
        use ts_io::json::Value;
        let pair = |p: Option<(f64, f64)>| match p {
            Some((f, pr)) => Value::obj().with("f", f).with("prom", pr),
            None => Value::Null,
        };
        let opt = |v: Option<f64>| v.map(Value::Float).unwrap_or(Value::Null);
        let cands: Vec<Value> = self
            .candidates
            .iter()
            .map(|e| {
                Value::obj()
                    .with("f", e.f)
                    .with("prom", e.prom)
                    .with("s", e.s)
                    .with("l1_pct", e.l1_pct)
                    .with("delta", e.delta)
                    .with("side", e.side.as_str())
                    .with("why", e.why.clone())
                    .with("pres", e.pres)
                    .with("drift", e.drift)
                    .with("k_hat", e.k_hat)
                    .with("gain_db", e.gain_db)
                    .with("k_contrast", e.k_contrast)
                    .with("mirror_db", e.mirror_db)
                    .with("cochan", pair(e.cochan))
                    .with("comb", opt(e.comb))
                    .with("sub_d", pair(e.sub_d))
            })
            .collect();
        let mut xc = Value::obj();
        for x in &self.xcheck {
            xc.push(
                &repr_shortest(x.nom),
                Value::obj()
                    .with("f", x.f)
                    .with("prom", x.prom)
                    .with("s", opt(x.s))
                    .with("ppm", opt(x.ppm)),
            );
        }
        let subs: Vec<Value> = self
            .sub_lines
            .iter()
            .map(|&(f, p)| Value::Arr(vec![f.into(), p.into()]))
            .collect();
        let mut v = Value::obj()
            .with("hz", self.hz)
            .with("f_primary", self.f_primary)
            .with("prom_primary", self.prom_primary)
            .with("r0", self.r0)
            .with("candidates", Value::Arr(cands))
            .with("xcheck", xc)
            .with("n_blocks", self.n_blocks)
            .with("sub_lines", Value::Arr(subs))
            .with("ambiguity", self.ambiguity.as_str());
        if let Some(e) = &self.error {
            v.push("error", e.as_str());
        }
        if let Some(s) = &self.solution {
            v.push("best", s.best);
            v.push("s", s.s);
            v.push("f_lo", s.f_lo);
            v.push("f_hi", s.f_hi);
            v.push("delta", s.delta);
            v.push("side", s.side.as_str());
            v.push("f_true", s.f_true);
            v.push("offset_hz", s.offset_hz);
            let bl: Vec<Value> = s
                .blocks
                .iter()
                .map(|&(a, b, c)| Value::Arr(vec![a.into(), b.into(), c.into()]))
                .collect();
            v.push("blocks", Value::Arr(bl));
            if let Some((p, sd, d)) = s.blk_stats {
                v.push("blk_ftrue_ptp_ppm", p);
                v.push("blk_ftrue_std_ppm", sd);
                v.push("blk_delta_ptp", d);
            }
        }
        v
    }
}

/// `pair_measure(x, sr, c, ref_hz, ...)`: find the adjacent-channel partner of the primary
/// beat (tracked as curve `c`, ratio rel. `ref_hz`) and the absolute speed
/// `s = (F_lo + F_hi) / (2 hz)`. A candidate line is rejected when its offset `|d|` lies
/// outside `[dmin, dmax]`, the implied speed is beyond `play x (1 +- smax)`, it is weak (prominence
/// / block presence), its modulation index `k` is not the primary's tape modulation, or a mirror
/// line at `2 F_p - F` marks it as a sideband of the primary.
pub fn pair_measure(
    x: &[f32],
    ch: usize,
    sr: f64,
    c: &Curve,
    ref_hz: f64,
    o: &PairParams,
) -> Res<PairResult> {
    let (hz, dmax, dmin, smax, play, seg_s) = (o.hz, o.dmax, o.dmin, o.smax, o.play, o.seg_s);
    let r0 = mean(&c.ratio);
    let fp0 = ref_hz * r0;
    let half = 2.0 * dmax * (1.0 + smax) + 15.0;
    let nz = Normalised::new(x, ch, sr, fp0, half, c, o.channel)?;
    let Spectra { f, p, blocks } = nz.spectra(1.0, seg_s, o.blk_s, Some(half));
    let df = f[1] - f[0];
    let nf = floor(&p, df, 10.0);
    let (fp, prom_p) = peak(&f, &p, &nf, fp0, 2.0);
    let pk = find_peaks(
        &prom_db(&p, &nf),
        Some(PAIR_PROM),
        Some(((1.0 / df) as i64).max(1) as f64),
    )?;
    let lines: Vec<f64> = pk
        .iter()
        .map(|&i| peak(&f, &p, &nf, f[i], 1.5 * df).0)
        .collect();
    let nb: Vec<(f64, Vec<f32>, Vec<f32>)> = blocks
        .into_iter()
        .map(|(t0, pb)| {
            let n = floor(&pb, df, 10.0);
            (t0, pb, n)
        })
        .collect();
    // sub-audio lines, same normalisation
    let lo = Normalised::new(x, ch, sr, 0.0, 420.0, c, o.channel)?;
    let sl = lo.spectra(1.0, seg_s, None, Some(420.0));
    let keep: Vec<usize> = (0..sl.f.len()).filter(|&i| sl.f[i] > 0.0).collect();
    let fl: Vec<f64> = keep.iter().map(|&i| sl.f[i]).collect();
    let pl: Vec<f32> = keep.iter().map(|&i| sl.p[i]).collect();
    let dfl = fl[1] - fl[0];
    let nl = floor(&pl, dfl, 5.0);
    let sub: Vec<(f64, f64)> = find_peaks(
        &prom_db(&pl, &nl),
        Some(PAIR_PROM),
        Some(((1.0 / dfl) as i64).max(1) as f64),
    )?
    .into_iter()
    .filter(|&i| fl[i] >= 20.0)
    .map(|i| peak(&fl, &pl, &nl, fl[i], 1.5 * dfl))
    .collect();
    drop(lo);
    let kp: Vec<Vec<f32>> = KS
        .iter()
        .map(|&k| {
            if k == 1.0 {
                p.clone()
            } else {
                nz.spectra(k, seg_s, None, Some(half)).p
            }
        })
        .collect();
    drop(nz);
    let mut all_lines = lines.clone();
    all_lines.push(fp);
    let mut cands = Vec::new();
    for &ff in &lines {
        if (ff - fp).abs() < 2.0 * dmin * (1.0 - smax) {
            continue;
        }
        let pr = peak(&f, &p, &nf, ff, 0.2).1;
        let s = (fp + ff) / (2.0 * hz);
        let d = (ff - fp) / (2.0 * s);
        let mut why = Vec::new();
        if !(dmin <= d.abs() && d.abs() <= dmax) {
            why.push(format!(
                "|d| {} Hz outside {}-{}",
                fmt_fixed(d.abs(), 1),
                fmt_general(dmin, 6),
                fmt_general(dmax, 6)
            ));
        }
        if (s / play - 1.0).abs() > smax {
            why.push(format!(
                "s {} beyond play x (1 +- {})",
                fmt_fixed(s, 4),
                fmt_general(smax, 6)
            ));
        }
        let fb: Vec<f64> = nb
            .iter()
            .map(|(_, pb, n)| peak(&f, pb, n, ff, 1.0))
            .filter(|&(_, pp)| pp >= PAIR_BLK_PROM)
            .map(|(g, _)| g)
            .collect();
        let pres = fb.len() as f64 / nb.len().max(1) as f64;
        let drift = if fb.len() > 1 { ptp(&fb).unwrap() } else { 0.0 };
        if pr < PAIR_ACCEPT || pres < PAIR_PRES {
            why.push(format!(
                "weak: {} dB / presence {} (need {} / {})",
                fmt_fixed(pr, 0),
                fmt_fixed(pres, 2),
                fmt_general(PAIR_ACCEPT, 6),
                fmt_general(PAIR_PRES, 6)
            ));
        }
        let kd: Vec<f32> = kp
            .iter()
            .map(|pk_| {
                let mut mx = f32::NEG_INFINITY;
                let mut nan = false;
                for i in 0..f.len() {
                    if (f[i] - ff).abs() <= 0.5 {
                        let v = pk_[i];
                        nan |= v.is_nan();
                        if v > mx {
                            mx = v;
                        }
                    }
                }
                let mx = if nan { f32::NAN } else { mx };
                10.0f32 * (mx + 1e-30f32).log10()
            })
            .collect();
        let j = argmax32(&kd);
        let mut kh = KS[j];
        if 0 < j && j < KS.len() - 1 {
            let (a, b, cc) = (kd[j - 1], kd[j], kd[j + 1]);
            let den = a - 2.0f32 * b + cc;
            if den < 0.0 {
                // float32 offset * 0.5; the sum with the f64 grid step stays float32
                let off = 0.5f32 * (a - cc) / den;
                kh = (KS[j] as f32 + off * ((KS[1] - KS[0]) as f32)) as f64;
            }
        }
        let gain_db = (kd[2] - kd[0]) as f64;
        let (kmx, kmn) = kd
            .iter()
            .fold((f32::NEG_INFINITY, f32::INFINITY), |(a, b), &v| {
                (a.max(v), b.min(v))
            });
        let k_contrast = (kmx - kmn) as f64;
        if k_contrast >= 3.0 && !(0.6..=1.4).contains(&kh) {
            why.push(format!(
                "modulation k {}: not the primary's tape modulation",
                fmt_fixed(kh, 2)
            ));
        }
        let mirror_db = peak(&f, &p, &nf, 2.0 * fp - ff, 0.4).1;
        if mirror_db >= (pr - 10.0).max(PAIR_BLK_PROM) {
            why.push(format!(
                "mirror {} Hz at {} dB: sideband of the primary",
                fmt_fixed(2.0 * fp - ff, 1),
                fmt_fixed(mirror_db, 0)
            ));
        }
        let sp = (ff - fp).abs();
        let cochan = sub.iter().copied().find(|&(g, _)| (g - sp).abs() <= 0.4);
        if let Some((g, pp)) = cochan {
            why.push(format!(
                "spacing {} Hz = sub-audio line {} Hz ({} dB): co-channel carrier beating with the same neighbour",
                fmt_fixed(sp, 2),
                fmt_fixed(g, 2),
                fmt_fixed(pp, 0)
            ));
        }
        let cm = comb(ff, &all_lines, 0.3);
        if let Some(cv) = cm.filter(|&v| v != 0.0)
            && PAIR_HUM
                .iter()
                .map(|h| (cv / (h * play) - 1.0).abs())
                .fold(f64::INFINITY, f64::min)
                <= 0.04
        {
            why.push(format!(
                "comb member (spacing {} Hz): modulated carrier, not a clean beat",
                fmt_fixed(cv, 2)
            ));
        }
        let sub_d = sub
            .iter()
            .copied()
            .find(|&(g, _)| (g - sp / 2.0).abs() <= 0.4);
        cands.push(Candidate {
            f: ff,
            prom: pr,
            s,
            l1_pct: (s - 1.0) * 100.0,
            delta: d.abs(),
            side: if d > 0.0 { Side::Hi } else { Side::Lo },
            why,
            pres,
            drift,
            k_hat: kh,
            gain_db,
            k_contrast,
            mirror_db,
            cochan,
            comb: cm,
            sub_d,
        });
    }
    // cross-check lines (dict semantics: a repeated nominal keeps its first position)
    let mut xc: IndexMap<u64, XCheck> = IndexMap::new();
    for &nom in o.xcheck.iter().filter(|&&v| v != 0.0) {
        let cx = nom * fp / hz;
        let hx = nom * dmax / hz * (1.0 + smax) + 10.0;
        let sx =
            Normalised::new(x, ch, sr, cx, hx, c, o.channel)?.spectra(1.0, seg_s, None, Some(hx));
        let nx = floor(&sx.p, sx.f[1] - sx.f[0], 10.0);
        let (fx, px) = peak(&sx.f, &sx.p, &nx, cx, hx);
        let key = if nom == 0.0 { 0 } else { nom.to_bits() };
        xc.insert(
            key,
            XCheck {
                nom,
                f: fx,
                prom: px,
                s: (px >= PAIR_PROM).then_some(fx / nom),
                ppm: None,
            },
        );
    }
    let mut xc: Vec<XCheck> = xc.into_values().collect();
    let mut ok: Vec<usize> = (0..cands.len())
        .filter(|&i| cands[i].why.is_empty())
        .collect();
    ok.sort_by(|&a, &b| {
        cands[b]
            .prom
            .partial_cmp(&cands[a].prom)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut best = ok.first().copied();
    let mut amb = "no candidate".to_string();
    if let (Some(b0), true) = (best, ok.len() > 1) {
        let (bc, rc) = (&cands[b0], &cands[ok[1]]);
        if rc.prom >= bc.prom - PAIR_MARGIN {
            amb = format!(
                "ambiguous: {} Hz ({} dB, pres {}) within {} dB of {} Hz ({} dB)",
                fmt_fixed(rc.f, 2),
                fmt_fixed(rc.prom, 0),
                fmt_fixed(rc.pres, 2),
                fmt_general(PAIR_MARGIN, 6),
                fmt_fixed(bc.f, 2),
                fmt_fixed(bc.prom, 0)
            );
            let agree: Vec<usize> = ok[..2]
                .iter()
                .copied()
                .filter(|&i| {
                    xc.iter().any(|v| {
                        v.s.is_some_and(|s| s != 0.0 && (s / cands[i].s - 1.0).abs() <= PAIR_XTOL)
                    })
                })
                .collect();
            best = (agree.len() == 1).then(|| agree[0]);
            if best.is_some() {
                let keys: Vec<String> = xc.iter().map(|v| fmt_general(v.nom, 6)).collect();
                amb += &format!("; resolved by cross-check {} Hz", keys.join(", "));
            }
        } else {
            amb = format!(
                "none: runner-up {} Hz {} dB, pres {}",
                fmt_fixed(rc.f, 2),
                fmt_fixed(rc.prom, 0),
                fmt_fixed(rc.pres, 2)
            );
        }
    } else if best.is_some() {
        amb = "none: single survivor".into();
    }
    for v in xc.iter_mut() {
        v.ppm = match (best, v.s) {
            (Some(b), Some(s)) if s != 0.0 => Some((s / cands[b].s - 1.0) * 1e6),
            _ => None,
        };
    }
    let mut res = PairResult {
        hz,
        f_primary: fp,
        prom_primary: prom_p,
        r0,
        candidates: cands,
        xcheck: xc,
        n_blocks: nb.len(),
        sub_lines: sub,
        ambiguity: amb,
        error: None,
        solution: None,
    };
    let Some(bi) = best else {
        res.error = Some(format!(
            "--abs pair: {} within |d| {}-{} Hz of the primary at {} Hz ({})",
            if ok.is_empty() {
                "no partner line passing the criteria"
            } else {
                "no unambiguous partner"
            },
            fmt_general(dmin, 6),
            fmt_general(dmax, 6),
            fmt_fixed(fp, 2),
            res.ambiguity
        ));
        return Ok(res);
    };
    let bf = res.candidates[bi].f;
    let s = res.candidates[bi].s;
    let (a_, b_) = if fp <= bf { (fp, bf) } else { (bf, fp) };
    // time-resolved: per block true primary frequency and d from the two lines alone
    let mut bl = Vec::new();
    for (t0, pb, n) in &nb {
        let (ga, pa) = peak(&f, pb, n, fp, 1.0);
        let (gb, pbb) = peak(&f, pb, n, bf, 1.0);
        // min(pa, pb): first argument unless the second is smaller
        if (if pbb < pa { pbb } else { pa }) >= PAIR_BLK_PROM {
            bl.push((
                np_round(*t0, 1),
                hz * 2.0 * ga / (ga + gb),
                (gb - ga).abs() * hz / (ga + gb),
            ));
        }
    }
    let blk_stats = (bl.len() >= 2).then(|| {
        let ft: Vec<f64> = bl.iter().map(|b| b.1).collect();
        let dd: Vec<f64> = bl.iter().map(|b| b.2).collect();
        let mf = median(&ft);
        (
            ptp(&ft).unwrap() / mf * 1e6,
            std(&ft) / mf * 1e6,
            ptp(&dd).unwrap(),
        )
    });
    res.solution = Some(PairSolution {
        best: bi,
        s,
        f_lo: a_ / s,
        f_hi: b_ / s,
        delta: (b_ - a_) / (2.0 * s),
        side: if fp < bf { Side::Lo } else { Side::Hi },
        f_true: fp / s,
        offset_hz: fp / s - hz,
        blocks: bl,
        blk_stats,
    });
    Ok(res)
}

/// `pair_table(res)`: candidate list as text lines.
pub fn pair_table(res: &PairResult) -> Vec<String> {
    let subs: Vec<String> = res
        .sub_lines
        .iter()
        .map(|&(g, _)| fmt_fixed(g, 2))
        .collect();
    let subs = if subs.is_empty() {
        "none".to_string()
    } else {
        subs.join(", ")
    };
    let mut l = vec![
        format!(
            "  primary {} Hz ({} dB), {} blocks; sub-audio lines {}; candidates (apparent Hz, normalised):",
            fmt_fixed(res.f_primary, 2),
            fmt_fixed(res.prom_primary, 0),
            res.n_blocks,
            subs
        ),
        format!(
            "  {:>9} {:>9} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8} {:>7}  status",
            "f", "f true", "prom", "pres", "k", "gain", "drift", "L1 %", "d Hz"
        ),
    ];
    let w = |v: f64, width: usize, prec: usize| format!("{:>width$}", fmt_fixed(v, prec));
    for (i, e) in res.candidates.iter().enumerate() {
        let best = res.solution.as_ref().map(|s| s.best);
        let st = if best == Some(i) {
            "CHOSEN".to_string()
        } else if !e.why.is_empty() {
            format!("rejected: {}", e.why.join("; "))
        } else {
            "survivor".to_string()
        };
        let s = res.solution.as_ref().map_or(e.s, |x| x.s);
        let l1 = if e.l1_pct.is_nan() {
            "nan".to_string()
        } else if e.l1_pct.is_infinite() {
            if e.l1_pct > 0.0 { "+inf" } else { "-inf" }.to_string()
        } else {
            format!("{:+.4}", e.l1_pct)
        };
        l.push(format!(
            "  {} {} {} {} {} {} {} {:>8} {}  {}",
            w(e.f, 9, 2),
            w(e.f / s, 9, 2),
            w(e.prom, 5, 1),
            w(e.pres, 5, 2),
            w(e.k_hat, 5, 2),
            w(e.gain_db, 5, 1),
            w(e.drift, 5, 2),
            l1,
            w(e.delta, 7, 2),
            st
        ));
    }
    l
}

/// `pair_info(pm)`: curve-meta / report keys of a pair_measure result with a solution.
pub fn pair_info(pm: &PairResult) -> Info {
    let sol = pm.solution.as_ref().expect("pair_info needs a solution");
    let e = &pm.candidates[sol.best];
    let mut o = Info::new();
    let mut put = |k: &str, v: MetaValue| {
        o.insert(k.to_string(), v);
    };
    put("pair_hz", pm.hz.into());
    put("pair_lo_hz", sol.f_lo.into());
    put("pair_hi_hz", sol.f_hi.into());
    put("pair_delta_hz", sol.delta.into());
    put(
        "pair_side",
        if sol.side == Side::Lo {
            "primary is the lower beat"
        } else {
            "primary is the upper beat"
        }
        .into(),
    );
    put("pair_offset_hz", sol.offset_hz.into());
    put("pair_ambiguity", pm.ambiguity.clone().into());
    put(
        "pair_prom_db",
        format!("{}/{}", fmt_fixed(pm.prom_primary, 1), fmt_fixed(e.prom, 1)).into(),
    );
    put("pair_pres", e.pres.into());
    put("pair_k", e.k_hat.into());
    put("pair_n_cand", pm.candidates.len().into());
    put(
        "pair_n_rejected",
        pm.candidates
            .iter()
            .filter(|c| !c.why.is_empty())
            .count()
            .into(),
    );
    put("pair_n_blocks", sol.blocks.len().into());
    if let Some((ptp_, sd, dd)) = sol.blk_stats {
        put("pair_blk_ftrue_std_ppm", sd.into());
        put("pair_blk_ftrue_ptp_ppm", ptp_.into());
        put("pair_blk_delta_ptp", dd.into());
    }
    let xc: Vec<String> = pm
        .xcheck
        .iter()
        .filter_map(|v| {
            v.ppm
                .map(|p| format!("{}:{}", fmt_general(v.nom, 6), fmt_plus(p, 1)))
        })
        .collect();
    if !xc.is_empty() {
        put("pair_xcheck_ppm", xc.join(" ").into());
    }
    o
}

/// Fixed-point with explicit sign: `+1.23` / `-0.50` (`n` decimals).
pub(crate) fn fmt_plus(x: f64, n: usize) -> String {
    if x.is_nan() {
        "+nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "+inf" } else { "-inf" }.into()
    } else {
        format!("{x:+.n$}")
    }
}
