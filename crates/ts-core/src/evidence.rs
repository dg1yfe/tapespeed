//! evidence: is the measured FM tape speed?
//!
//! - [`fading_metric`]: Spearman rank correlation between line-envelope dips and residual FM
//!   of the line after derotation by the curve (file-wide and per segment).
//! - [`band_stats`]: band-wise coherence / rms of two curves (Welch segments with a validity mask).
//! - [`bandwidth_policy`] + [`apply_bandwidth`]: correction band limit, curve lowpass
//!   ([`lowpass_curve`]: Butterworth 4 sosfiltfilt, padtype even, explicit padlen) with the
//!   periodic model part kept above the limit.
//!
//! Results keep a fixed key order ([`Fading::to_value`], [`BandStats::to_value`]); it is part
//! of the report format.
//! Numerics follow numpy's operation order (pairwise sums, hypot for |z|, libm trig); the
//! stage-1 heterodyne/decimator is chunked and run in parallel per chunk, each chunk computed
//! identically, so thread count does not change any value.

use num_complex::Complex64;
use rayon::prelude::*;
use ts_dsp::ValueError;
use ts_dsp::conv::resample_poly_down_c;
use ts_dsp::iir::{PadType, butter_lowpass_sos, cmul, sosfiltfilt};
use ts_dsp::ndimage::{Mode, median_filter, uniform_filter1d};
use ts_dsp::spectral::{rfft_frames, rfftfreq};
use ts_dsp::spline::cubic_spline;
use ts_dsp::stats::{interp, mean, median, np_sum, spearmanr};
use ts_dsp::window::hanning;
use ts_io::json::Value;
use ts_io::{fmt_fixed, fmt_general, repr_shortest, round_dec};

use crate::curve::Curve;

/// Coherence bands [Hz] (`BANDS`).
pub const BANDS: [(f64, f64); 3] = [(0.0, 0.5), (0.5, 6.0), (6.0, 25.0)];
/// Fading-only limit threshold (`FADE_RHO`).
pub const FADE_RHO: f64 = 0.9;
/// Band counted as tape-caused at or above this coherence (`COH_MIN`).
pub const COH_MIN: f64 = 0.5;

type Res<T> = Result<T, ValueError>;

fn verr<T>(m: impl Into<String>) -> Res<T> {
    Err(ValueError(m.into()))
}

/// `min(a, b)` (first argument wins ties; NaN in `a` is kept).
#[inline]
pub(crate) fn min2(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `max(a, b)` (first argument wins ties; NaN in `a` is kept).
#[inline]
pub(crate) fn max2(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// numpy mean of a bool mask (NaN when empty).
fn bool_mean(it: impl Iterator<Item = bool>) -> f64 {
    let (mut s, mut n) = (0usize, 0usize);
    for b in it {
        s += b as usize;
        n += 1;
    }
    s as f64 / n as f64
}

fn count(m: &[bool]) -> usize {
    m.iter().filter(|v| **v).count()
}

fn sel<T: Copy>(a: &[T], m: &[bool]) -> Vec<T> {
    a.iter()
        .zip(m)
        .filter(|(_, b)| **b)
        .map(|(v, _)| *v)
        .collect()
}

/// `BANDS` key `f"{lo:g}-{hi:g}"`.
pub fn band_key(lo: f64, hi: f64) -> String {
    format!("{}-{}", fmt_general(lo, 6), fmt_general(hi, 6))
}

// stage-1 baseband: canonical in crate::hetdec (pick / lpf / het_decimate)
use crate::hetdec::het_decimate;
pub use crate::hetdec::{Channel, lpf};

/// `line_baseband(x, sr, ref_hz, c, channel, bw)`: stage-1 complex baseband of the line
/// around `ref_hz * mean(r)`. Returns `(t, z, fs)`.
pub fn line_baseband(
    x: &[f32],
    ch: usize,
    sr: f64,
    ref_hz: f64,
    c: &Curve,
    channel: Channel,
    bw: f64,
) -> Res<(Vec<f64>, Vec<Complex64>, f64)> {
    let r = &c.ratio;
    let rm = mean(r);
    let mut mx = f64::NEG_INFINITY;
    for v in r {
        let a = (v - rm).abs();
        if a > mx || a.is_nan() {
            mx = a;
        }
    }
    let dev = mx * ref_hz + 1.5 * bw;
    let d1 = ((sr / (2.5 * dev)) as i64).max(1) as usize;
    let fs1 = sr / d1 as f64;
    let h = lpf(dev, min2(fs1 - dev, 1.3 * dev), sr, 90.0)?;
    let z = het_decimate(x, ch, sr, ref_hz * rm, d1, &h, channel)?;
    let t = (0..z.len()).map(|i| i as f64 / fs1).collect();
    Ok((t, z, fs1))
}

/// `derot_lp(z, t, fs, ref_hz, c, 0.0, bw)`: derotate by the curve phase, filter to ±bw.
fn derot_lp(
    z: &[Complex64],
    t: &[f64],
    fs: f64,
    ref_hz: f64,
    c: &Curve,
    bw: f64,
) -> Res<(Vec<Complex64>, f64)> {
    let rm = mean(&c.ratio);
    let rr: Vec<f64> = c.ratio.iter().map(|v| v - rm).collect();
    let phi = cubic_spline(&c.t, &rr)?.antiderivative();
    let (t0, t1) = (c.t[0], c.t[c.t.len() - 1]);
    let d = ((fs / (4.0 * bw)) as i64).max(1) as usize;
    let fsd = fs / d as f64;
    let h = lpf(bw, min2(fsd - bw, 1.5 * bw), fs, 90.0)?;
    let k = -std::f64::consts::TAU * ref_hz;
    let mut hint = 0;
    let v: Vec<Complex64> = z
        .iter()
        .zip(t)
        .map(|(&zz, &tt)| {
            let th = k * phi.eval_hint(tt.clamp(t0, t1), &mut hint);
            cmul(zz, Complex64::new(th.cos(), th.sin()))
        })
        .collect();
    Ok((resample_poly_down_c(&v, d, &h)?, fsd))
}

// ------------------------------------------------------------------ fading metric

/// Result of [`fading_metric`] ([`Fading::to_value`] keeps the key order).
#[derive(Debug, Clone, PartialEq)]
pub struct Fading {
    pub rho: f64,
    pub fm_ratio: f64,
    pub deep_frac: f64,
    pub rho_curve: f64,
    pub curve_ratio: f64,
    pub valid_in_deep: f64,
    /// per extra curve: (name, rho_<name>, ratio_<name>)
    pub extra: Vec<(String, f64, f64)>,
    /// rounded to 3 decimals (correctly rounded, ties to even)
    pub seg_rho: Vec<f64>,
    pub seg_rho_median: f64,
    pub seg_high_frac: f64,
}

impl Fading {
    /// `(key, value)` of the scalar entries in report order (`seg_rho` excluded).
    pub fn scalars(&self) -> Vec<(String, f64)> {
        let mut out = vec![
            ("rho".into(), self.rho),
            ("fm_ratio".into(), self.fm_ratio),
            ("deep_frac".into(), self.deep_frac),
            ("rho_curve".into(), self.rho_curve),
            ("curve_ratio".into(), self.curve_ratio),
            ("valid_in_deep".into(), self.valid_in_deep),
        ];
        for (name, r, q) in &self.extra {
            out.push((format!("rho_{name}"), *r));
            out.push((format!("ratio_{name}"), *q));
        }
        out.push(("seg_rho_median".into(), self.seg_rho_median));
        out.push(("seg_high_frac".into(), self.seg_high_frac));
        out
    }

    /// The report entry as a JSON value, keys in report order.
    pub fn to_value(&self) -> Value {
        let sc = self.scalars();
        let (head, tail) = sc.split_at(sc.len() - 2);
        let mut v = Value::obj();
        for (k, x) in head {
            v.push(k, *x);
        }
        v.push(
            "seg_rho",
            Value::Arr(self.seg_rho.iter().map(|&x| Value::Float(x)).collect()),
        );
        for (k, x) in tail {
            v.push(k, *x);
        }
        v
    }
}

/// Keyword parameters of [`fading_metric`] (with their defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct FadingParams {
    pub blk: f64,
    pub seg: f64,
    pub deep_db: f64,
}

impl Default for FadingParams {
    fn default() -> Self {
        Self {
            blk: 0.1,
            seg: 30.0,
            deep_db: 6.0,
        }
    }
}

/// An extra curve for [`fading_metric`]: `(name, ratio, valid)` on `c.t`.
pub type ExtraCurve<'a> = (&'a str, &'a [f64], &'a [bool]);

/// `fading_metric(x, sr, ref_hz, c, channel, blk, seg, deep_db, extra)`; `x` interleaved f32
/// frames with `ch` channels. Errors on unusable input (the pipeline then records
/// `{"error": ...}`; evidence never blocks the correction).
#[allow(clippy::too_many_arguments)]
pub fn fading_metric(
    x: &[f32],
    ch: usize,
    sr: f64,
    ref_hz: f64,
    c: &Curve,
    channel: Channel,
    p: &FadingParams,
    extra: &[ExtraCurve<'_>],
) -> Res<Fading> {
    let (blk, deep_db) = (p.blk, p.deep_db);
    let (t1, z1, fs1) = line_baseband(x, ch, sr, ref_hz, c, channel, 25.0)?;
    let (u, fs) = derot_lp(&z1, &t1, fs1, ref_hz, c, 10.0)?;
    let nu = u.len();
    if nu < 3 {
        return verr("fading_metric: baseband too short");
    }
    let mut f = vec![0.0; nu];
    let k4 = 4.0 * std::f64::consts::PI;
    for i in 1..nu - 1 {
        let q = cmul(u[i + 1], u[i - 1].conj());
        f[i] = q.im.atan2(q.re) * fs / k4;
    }
    f[0] = f[1];
    f[nu - 1] = f[nu - 2];
    let k = ((blk * fs).round_ties_even() as i64).max(1) as usize;
    let nb = nu / k;
    let a2: Vec<f64> = u[..nb * k]
        .iter()
        .map(|z| {
            let a = z.re.hypot(z.im);
            a * a
        })
        .collect();
    let f2: Vec<f64> = f[..nb * k].iter().map(|v| v * v).collect();
    let env: Vec<f64> = a2
        .chunks_exact(k)
        .map(|r| 10.0 * (np_sum(r) / k as f64 + 1e-30).log10())
        .collect();
    let fm: Vec<f64> = f2
        .chunks_exact(k)
        .map(|r| (np_sum(r) / k as f64).sqrt())
        .collect();
    let msize = ((10.0 / blk) as i64).max(3) as usize | 1;
    let dip: Vec<f64> = median_filter(&env, msize, Mode::Nearest)?
        .iter()
        .zip(&env)
        .map(|(m, e)| m - e)
        .collect();
    let tb: Vec<f64> = (0..nb).map(|i| (i as f64 + 0.5) * blk).collect();
    let ok: Vec<bool> = dip.iter().map(|v| v.is_finite()).collect();

    let ufs = ((1.0 / blk) as i64).max(1) as usize;
    let fastpart = |r: &[f64], v: &[bool]| -> Res<(Vec<f64>, Vec<bool>)> {
        let rb: Vec<f64> = interp(&tb, &c.t, r)?.iter().map(|x| x * ref_hz).collect();
        let sm = uniform_filter1d(&rb, ufs, Mode::Nearest)?;
        let fast = rb.iter().zip(&sm).map(|(a, b)| (a - b).abs()).collect();
        let vf: Vec<f64> = v.iter().map(|&b| b as u8 as f64).collect();
        let vb = interp(&tb, &c.t, &vf)?.iter().map(|&x| x > 0.5).collect();
        Ok((fast, vb))
    };
    let rho = |a: &[f64], b: &[f64], m: &[bool]| -> Res<f64> {
        if count(m) > 20 {
            spearmanr(&sel(a, m), &sel(b, m))
        } else {
            Ok(f64::NAN)
        }
    };
    let and =
        |a: &[bool], b: &[bool]| -> Vec<bool> { a.iter().zip(b).map(|(x, y)| *x && *y).collect() };
    let ratio = |y: &[f64], m: &[bool]| -> f64 {
        let d: Vec<bool> = m
            .iter()
            .zip(&dip)
            .map(|(&mm, &v)| mm && v > deep_db)
            .collect();
        let q: Vec<bool> = m.iter().zip(&dip).map(|(&mm, &v)| mm && v < 2.0).collect();
        if count(&d) > 10 && count(&q) > 10 {
            median(&sel(y, &d)) / max2(median(&sel(y, &q)), 1e-12)
        } else {
            f64::NAN
        }
    };
    let (fast, vb) = fastpart(&c.ratio, &c.valid)?;
    let okv = and(&ok, &vb);
    let deep: Vec<bool> = ok
        .iter()
        .zip(&dip)
        .map(|(&o, &v)| o && v > deep_db)
        .collect();
    let mut out = Fading {
        rho: rho(&dip, &fm, &ok)?,
        fm_ratio: ratio(&fm, &ok),
        deep_frac: bool_mean(sel(&dip, &ok).iter().map(|&v| v > deep_db)),
        rho_curve: rho(&dip, &fast, &okv)?,
        curve_ratio: ratio(&fast, &okv),
        valid_in_deep: if deep.iter().any(|v| *v) {
            bool_mean(sel(&vb, &deep).into_iter())
        } else {
            f64::NAN
        },
        extra: Vec::new(),
        seg_rho: Vec::new(),
        seg_rho_median: f64::NAN,
        seg_high_frac: f64::NAN,
    };
    for &(name, r, v) in extra {
        let (fx, vx) = fastpart(r, v)?;
        let m = and(&ok, &vx);
        out.extra
            .push((name.to_string(), rho(&dip, &fx, &m)?, ratio(&fx, &m)));
    }
    let ns = ((p.seg / blk) as i64).max(1) as usize;
    let stop = nb as i64 - (ns / 2) as i64;
    let mut seg_r = Vec::new();
    let mut i = 0i64;
    while i < stop {
        let (a, b) = (i as usize, (i as usize + ns).min(nb));
        seg_r.push(rho(&dip[a..b], &fm[a..b], &ok[a..b])?);
        i += ns as i64;
    }
    let good: Vec<f64> = seg_r.iter().copied().filter(|v| v.is_finite()).collect();
    out.seg_rho = seg_r.iter().map(|&v| round_dec(v, 3)).collect();
    if !good.is_empty() {
        out.seg_rho_median = median(&good);
        out.seg_high_frac = bool_mean(good.iter().map(|&v| v > FADE_RHO));
    }
    Ok(out)
}

// ------------------------------------------------------------------ band coherence

/// Result of [`band_stats`]: per band `(key, value)` in band order (empty if < 3 segments).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BandStats {
    pub n_seg: usize,
    pub coh: Vec<(String, f64)>,
    pub rms1: Vec<(String, f64)>,
    pub rms2: Vec<(String, f64)>,
    pub rms_diff: Vec<(String, f64)>,
}

impl BandStats {
    /// `coh.get(key)`.
    pub fn coh_of(&self, key: &str) -> Option<f64> {
        self.coh.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    /// The report entry as a JSON value, keys in report order.
    pub fn to_value(&self) -> Value {
        let m = |v: &[(String, f64)]| {
            Value::Obj(
                v.iter()
                    .map(|(k, x)| (k.clone(), Value::Float(*x)))
                    .collect(),
            )
        };
        Value::obj()
            .with("n_seg", self.n_seg)
            .with("coh", m(&self.coh))
            .with("rms1", m(&self.rms1))
            .with("rms2", m(&self.rms2))
            .with("rms_diff", m(&self.rms_diff))
    }
}

/// numpy pairwise sum of a contiguous complex128 array (`CDOUBLE_pairwise_sum`, which counts
/// real components: blocks of 8 reals = 4 complex values, 4+4 accumulators).
fn csum(a: &[Complex64]) -> Complex64 {
    fn pw(a: &[Complex64]) -> (f64, f64) {
        let n = 2 * a.len();
        if n < 8 {
            let (mut r, mut i) = (-0.0, -0.0);
            for z in a {
                r += z.re;
                i += z.im;
            }
            (r, i)
        } else if n <= 128 {
            let mut r = [
                a[0].re, a[0].im, a[1].re, a[1].im, a[2].re, a[2].im, a[3].re, a[3].im,
            ];
            let body = (n - n % 8) / 2;
            let mut j = 4;
            while j < body {
                for q in 0..4 {
                    r[2 * q] += a[j + q].re;
                    r[2 * q + 1] += a[j + q].im;
                }
                j += 4;
            }
            let mut rr = (r[0] + r[2]) + (r[4] + r[6]);
            let mut ri = (r[1] + r[3]) + (r[5] + r[7]);
            for z in &a[body..] {
                rr += z.re;
                ri += z.im;
            }
            (rr, ri)
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            let (a1, a2) = pw(&a[..n2 / 2]);
            let (b1, b2) = pw(&a[n2 / 2..]);
            (a1 + b1, a2 + b2)
        }
    }
    let (r, i) = pw(a);
    Complex64::new(0.0 + r, 0.0 + i)
}

/// `band_stats(t, r1, r2, valid, bands, nper, min_valid)`.
pub fn band_stats(
    t: &[f64],
    r1: &[f64],
    r2: &[f64],
    valid: &[bool],
    bands: &[(f64, f64)],
    nper: usize,
    min_valid: f64,
) -> Res<BandStats> {
    let n = t.len();
    if n < 2 || r1.len() < n || r2.len() < n || valid.len() < n || nper < 2 {
        return verr("band_stats: inconsistent lengths");
    }
    let dt = t[1] - t[0];
    let mut st = Vec::new();
    let mut s = 0;
    while s + nper <= n {
        let cnt = count(&valid[s..s + nper]);
        if cnt as f64 / nper as f64 >= min_valid {
            st.push(s);
        }
        s += nper / 2;
    }
    let mut out = BandStats {
        n_seg: st.len(),
        ..Default::default()
    };
    if st.len() < 3 {
        return Ok(out);
    }
    let hw = hanning(nper);
    let ones = vec![1.0; nper];
    let nf = nper / 2 + 1;
    let fr = rfftfreq(nper, dt);
    // per-segment spectra in parallel; accumulation in segment order (numpy's `S += ...`)
    type Seg = (Vec<Complex64>, Vec<Complex64>, f64);
    let specs: Vec<Seg> = st
        .par_iter()
        .map(|&s| -> Res<Seg> {
            let w: Vec<f64> = (0..nper)
                .map(|i| hw[i] * valid[s + i] as u8 as f64)
                .collect();
            let sw = np_sum(&w);
            let spec = |r: &[f64]| -> Res<Vec<Complex64>> {
                let seg = &r[s..s + nper];
                let rw: Vec<f64> = seg.iter().zip(&w).map(|(a, b)| a * b).collect();
                let m = np_sum(&rw) / sw;
                let buf: Vec<f64> = seg.iter().zip(&w).map(|(a, b)| (a - m) * b).collect();
                Ok(rfft_frames(&buf, &ones, nper, nper)?.0)
            };
            let w2: Vec<f64> = w.iter().map(|v| v * v).collect();
            Ok((spec(r1)?, spec(r2)?, np_sum(&w2)))
        })
        .collect::<Res<_>>()?;
    let zero = Complex64::new(0.0, 0.0);
    let mut acc = vec![[zero; 4]; nf];
    let mut wsum = 0.0;
    let sq = |z: Complex64| {
        let a = z.re.hypot(z.im);
        Complex64::new(a * a, 0.0)
    };
    for (x, y, w2) in &specs {
        for k in 0..nf {
            let (a, b) = (x[k], y[k]);
            acc[k][0] += sq(a);
            acc[k][1] += sq(b);
            acc[k][2] += cmul(a, b.conj());
            acc[k][3] += sq(a - b);
        }
        wsum += w2;
    }
    let sc = 2.0 * dt / wsum;
    let sxx: Vec<f64> = acc.iter().map(|v| v[0].re * sc).collect();
    let syy: Vec<f64> = acc.iter().map(|v| v[1].re * sc).collect();
    let sxy: Vec<Complex64> = acc
        .iter()
        .map(|v| Complex64::new(v[2].re * sc, v[2].im * sc))
        .collect();
    let sdd: Vec<f64> = acc.iter().map(|v| v[3].re * sc).collect();
    let df = fr[1] - fr[0];
    for &(lo, hi) in bands {
        let lim = max2(lo, 1.5 * df);
        let m: Vec<bool> = fr.iter().map(|&f| f > lim && f <= hi).collect();
        let key = band_key(lo, hi);
        let (px, py, pd) = (
            np_sum(&sel(&sxx, &m)),
            np_sum(&sel(&syy, &m)),
            np_sum(&sel(&sdd, &m)),
        );
        let c = csum(&sel(&sxy, &m));
        let ac = c.re.hypot(c.im);
        out.coh.push((key.clone(), ac * ac / (px * py + 1e-300)));
        out.rms1.push((key.clone(), (px * df).sqrt()));
        out.rms2.push((key.clone(), (py * df).sqrt()));
        out.rms_diff.push((key, (pd * df).sqrt()));
    }
    Ok(out)
}

/// [`band_stats`] with the defaults (`BANDS`, nper 2048, min_valid 0.5).
pub fn band_stats_default(t: &[f64], r1: &[f64], r2: &[f64], valid: &[bool]) -> Res<BandStats> {
    band_stats(t, r1, r2, valid, &BANDS, 2048, 0.5)
}

// ------------------------------------------------------------------ bandwidth policy

/// `--corr-limit` (`--weak-rf` = `On`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrLimit {
    Auto,
    On,
    Off,
}

impl std::str::FromStr for CorrLimit {
    type Err = ValueError;
    fn from_str(s: &str) -> Res<Self> {
        match s {
            "auto" => Ok(CorrLimit::Auto),
            "on" => Ok(CorrLimit::On),
            "off" => Ok(CorrLimit::Off),
            _ => verr(format!("invalid choice: {s:?}")),
        }
    }
}

/// `bandwidth_policy(path, fading, coh, max_hz, mode)` → (limit [Hz] or None, reason).
///
/// `fading_rho`: the `rho` entry of the fading dict (`None` when the dict is missing or holds
/// only `error`). `coh`: the coherence dict (`None` = no second line).
pub fn bandwidth_policy(
    path: &str,
    fading_rho: Option<f64>,
    coh: Option<&BandStats>,
    max_hz: f64,
    mode: CorrLimit,
) -> (Option<f64>, String) {
    match mode {
        CorrLimit::Off => return (None, "off".into()),
        CorrLimit::On => return (Some(max_hz), "forced".into()),
        CorrLimit::Auto => {}
    }
    if let Some(c) = coh
        && !c.coh.is_empty()
    {
        let mut top = None;
        for &(lo, hi) in &BANDS[1..] {
            if c.coh_of(&band_key(lo, hi)).unwrap_or(0.0) >= COH_MIN {
                top = Some(hi);
            } else {
                break;
            }
        }
        return match top {
            None => (Some(BANDS[1].0), "no coherent band above 0.5 Hz".into()),
            Some(t) if t >= BANDS[2].1 => (None, "coherent to 25 Hz".into()),
            Some(t) => (Some(t), format!("coherent to {} Hz", fmt_general(t, 6))),
        };
    }
    if path == "plain"
        && let Some(rho) = fading_rho
        && rho.is_finite()
        && rho > FADE_RHO
    {
        return (
            Some(max_hz),
            format!(
                "fading rho {} > {}, no second line",
                fmt_fixed(rho, 2),
                repr_shortest(FADE_RHO)
            ),
        );
    }
    (None, "no evidence against tape origin".into())
}

/// `lowpass_curve(t, r, fc)`: zero-phase Butterworth-4 lowpass of the deviation from the mean
/// (sosfiltfilt, padtype "even", padlen `min(len - 1, int(5 / fc / dt))`).
pub fn lowpass_curve(t: &[f64], r: &[f64], fc: f64) -> Res<Vec<f64>> {
    if t.len() < 2 || r.is_empty() {
        return verr("lowpass_curve: need >= 2 samples");
    }
    let dt = t[1] - t[0];
    let m = mean(r);
    let sos = butter_lowpass_sos(4, fc, 1.0 / dt)?;
    let d: Vec<f64> = r.iter().map(|v| v - m).collect();
    let padlen = ((5.0 / fc / dt) as i64).max(0) as usize;
    let y = sosfiltfilt(&sos, &d, PadType::Even, Some((r.len() - 1).min(padlen)))?;
    Ok(y.iter().map(|v| m + v).collect())
}

/// Correction-band step of `run`: with a limit `fc`, lowpass the curve and, when a periodic
/// model prediction `model` (on `t`) exists, add back its part above `fc`
/// (`ratio + mp - lowpass_curve(t, mp, fc)`). Returns (ratio, corr_keep_model_hf).
pub fn apply_bandwidth(
    t: &[f64],
    ratio: &[f64],
    fc: Option<f64>,
    model: Option<&[f64]>,
) -> Res<(Vec<f64>, bool)> {
    let Some(fc) = fc else {
        return Ok((ratio.to_vec(), false));
    };
    let mut out = lowpass_curve(t, ratio, fc)?;
    let Some(mp) = model else {
        return Ok((out, false));
    };
    let lp = lowpass_curve(t, mp, fc)?;
    for ((o, m), l) in out.iter_mut().zip(mp).zip(&lp) {
        *o = *o + m - l;
    }
    Ok((out, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_helpers() {
        assert_eq!(band_key(0.0, 0.5), "0-0.5");
        assert_eq!(band_key(6.0, 25.0), "6-25");
        assert!(max2(f64::NAN, 1e-12).is_nan());
        assert_eq!(min2(1.0, f64::NAN), 1.0);
        let z: Vec<Complex64> = (0..300)
            .map(|i| Complex64::new(i as f64 * 0.5, -(i as f64)))
            .collect();
        let s = csum(&z);
        assert_eq!((s.re, s.im), (22425.0, -44850.0));
    }

    #[test]
    fn policy_modes() {
        assert_eq!(
            bandwidth_policy("plain", Some(0.95), None, 6.0, CorrLimit::Off),
            (None, "off".into())
        );
        assert_eq!(
            bandwidth_policy("aided", None, None, 6.0, CorrLimit::On),
            (Some(6.0), "forced".into())
        );
        assert_eq!(
            bandwidth_policy("plain", Some(0.95), None, 6.0, CorrLimit::Auto).1,
            "fading rho 0.95 > 0.9, no second line"
        );
    }
}
