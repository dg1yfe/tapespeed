//! Survey line detection: blocks, channels, frame spectra, `line_grid`, `frames`,
//! `detect_block`, `group`, `wow_corr`, `classify`, `cliff`, `level_db`, `sigma_ratio`.
//!
//! Numerics follow numpy's operation order: frame spectra `rfft(frame * np.hanning)`,
//! frame means accumulated in frame order (numpy axis-0 reduction), pairwise sums for
//! contiguous reductions, hypot for |X|, libm exp/log.

use num_complex::Complex64;
use rayon::prelude::*;
use ts_dsp::conv::{resample_poly_default_spec, resample_poly_down};
use ts_dsp::fir::firwin_kaiser;
use ts_dsp::ndimage::{Mode, median_filter};
use ts_dsp::peaks::find_peaks;
use ts_dsp::spectral::{rfft_frames, rfftfreq};
use ts_dsp::stats::{argmax, argsort, corrcoef, detrend_linear, interp, mean, median, np_sum, std};
use ts_dsp::window::hanning;

use super::{Res, SurveyError};

/// Log-frequency grid step.
pub const STEP: f64 = 5e-4;
pub const WIDTHS: [f64; 4] = [5e-4, 1.5e-3, 4e-3, 1e-2];
pub const RING_HI: (f64, f64) = (0.012, 0.03);
pub const RING_LO: (f64, f64) = (0.03, 0.10);
/// [dB] detection
pub const MIN_PROM: f64 = 6.0;
/// [dB] frame presence
pub const FRAME_SNR: f64 = 10.0;
/// grouping tolerance
pub const GROUP_TOL: f64 = 4e-3;
/// hum / low band upper edge [Hz]
pub const LOW_MAX: f64 = 1000.0;
/// |f / (k 50|60) - 1| for playback/transfer hum
pub const HUM_EXACT: f64 = 5e-4;
pub const TAPE_CORR: f64 = 0.5;
pub const FREE_CORR: f64 = 0.2;
pub const LOSSY: [&str; 24] = [
    "mp3",
    "mp2",
    "mp1",
    "aac",
    "vorbis",
    "opus",
    "wmav1",
    "wmav2",
    "wmapro",
    "ac3",
    "eac3",
    "atrac3",
    "atrac3p",
    "amr_nb",
    "amr_wb",
    "ra_144",
    "ra_288",
    "cook",
    "gsm",
    "qcelp",
    "speex",
    "musepack7",
    "musepack8",
    "dts",
];
/// (class, nominal Hz, label)
pub const NOMINAL: [(&str, f64, &str); 5] = [
    ("FM pilot", 19000.0, "FM stereo pilot"),
    ("AM beat", 9000.0, "AM adjacent-channel beat (EU 9 kHz)"),
    ("AM beat", 10000.0, "AM adjacent-channel beat (US 10 kHz)"),
    ("TV line", 15625.0, "TV line whistle (PAL)"),
    ("TV line", 15734.0, "TV line whistle (NTSC)"),
];
pub const SCALE_DB: f64 = 4.0;
pub const LOCK_DB: f64 = 21.0 - SCALE_DB;
pub const MARGIN_DB: f64 = 26.0 - SCALE_DB;
pub const TRIAL_MIN: f64 = 0.10;
pub const OWN_FM: f64 = 2.0;
pub const PB_FULL: f64 = 25.0;

/// `sigma_ratio(cn0_db, f, pb)`: measurement-noise rms of the ratio curve flat to `pb`.
pub fn sigma_ratio(cn0_db: f64, f: f64, pb: f64) -> f64 {
    ((1.27 * pb).powf(3.0) / (3.0 * 10f64.powf(cn0_db / 10.0))).sqrt() / f
}

/// `blocks(dur, n, L)`: `[(t0, len)]`.
pub fn blocks(dur: f64, n: usize, l: f64) -> Vec<(f64, f64)> {
    if dur <= n as f64 * l {
        let k = ((dur / l - 1e-9).ceil() as i64).max(1) as usize;
        return (0..k)
            .map(|i| (i as f64 * dur / k as f64, dur / k as f64))
            .collect();
    }
    (0..n)
        .map(|i| ((i as f64 + 0.5) * dur / n as f64 - l / 2.0, l))
        .collect()
}

/// `channels(x)`: `[("mono", x)]` or `[("L"), ("R"), ("mid"), ("S")]` as f64.
pub fn channels(x: &[f32], ch: usize) -> Vec<(String, Vec<f64>)> {
    if ch == 1 {
        return vec![("mono".into(), x.iter().map(|&v| v as f64).collect())];
    }
    let l: Vec<f64> = x.iter().step_by(ch).map(|&v| v as f64).collect();
    let r: Vec<f64> = x.iter().skip(1).step_by(ch).map(|&v| v as f64).collect();
    let mid = l.iter().zip(&r).map(|(a, b)| (a + b) / 2.0).collect();
    let s = l.iter().zip(&r).map(|(a, b)| (a - b) / 2.0).collect();
    vec![
        ("L".into(), l),
        ("R".into(), r),
        ("mid".into(), mid),
        ("S".into(), s),
    ]
}

/// Frame power spectra `|rfft(frame * hanning)|^2` (frames x bins, row-major).
pub struct Frames {
    pub s: Vec<f64>,
    pub nframes: usize,
    pub nbin: usize,
}

impl Frames {
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.s[i * self.nbin..(i + 1) * self.nbin]
    }

    /// `2 * np.mean(S, 0) / scale` (frame-order accumulation as numpy's axis-0 reduction).
    pub fn psd(&self, scale: f64) -> Vec<f64> {
        let mut acc = vec![0.0; self.nbin];
        for i in 0..self.nframes {
            for (a, v) in acc.iter_mut().zip(self.row(i)) {
                *a += v;
            }
        }
        let n = self.nframes as f64;
        acc.iter().map(|a| 2.0 * (a / n) / scale).collect()
    }
}

/// Frame spectra for one channel: frames `x[p*hop .. p*hop+nper]`, `np.hanning` window.
/// Returns the power frames and `sum(w**2)`.
pub fn spectra(x: &[f64], nper: usize, hop: usize) -> Res<(Frames, f64)> {
    let w = hanning(nper);
    let w2: Vec<f64> = w.iter().map(|v| v * v).collect();
    let nbin = nper / 2 + 1;
    let (z, nf): (Vec<Complex64>, usize) = if x.len() >= nper {
        rfft_frames(x, &w, hop, nper)?
    } else {
        (vec![], 0)
    };
    let s = z
        .iter()
        .map(|c| {
            let a = c.re.hypot(c.im);
            a * a
        })
        .collect();
    Ok((
        Frames {
            s,
            nframes: nf,
            nbin,
        },
        np_sum(&w2),
    ))
}

/// Result of [`line_grid`] on the log grid `fk`.
#[derive(Debug, Clone)]
pub struct Grid {
    pub fk: Vec<f64>,
    pub prom: Vec<f64>,
    pub cn0: Vec<f64>,
    pub hw: Vec<f64>,
    pub hc: Vec<f64>,
    pub n0: Vec<f64>,
    pub d: Vec<f64>,
}

/// numpy `pad(mode="reflect")` index (no edge repeat).
#[inline]
fn reflect(i: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    let p = 2 * (n - 1);
    let k = i.rem_euclid(p);
    (if k < n { k } else { p - k }) as usize
}

/// `line_grid(f, P, fmin, fmax, ring, widths)`.
pub fn line_grid(f: &[f64], p: &[f64], fmin: f64, fmax: f64, ring: (f64, f64)) -> Res<Grid> {
    let nb = f.len();
    if nb < 2 {
        return Err(SurveyError::Value("line_grid: spectrum too short".into()));
    }
    let df = f[1] - f[0];
    let mut edges: Vec<f64> = f.iter().map(|v| v - df / 2.0).collect();
    edges.push(f[nb - 1] + df / 2.0);
    let mut cs = Vec::with_capacity(nb + 1);
    cs.push(0.0);
    let mut acc = 0.0;
    for v in p {
        acc += v;
        cs.push(acc * df);
    }
    let e = |a: &[f64], b: &[f64]| -> Res<Vec<f64>> {
        let ib = interp(b, &edges, &cs)?;
        let ia = interp(a, &edges, &cs)?;
        Ok(ib.iter().zip(&ia).map(|(x, y)| x - y).collect())
    };
    let nk = ((fmax / fmin).ln() / STEP) as i64 + 1;
    let nk = nk.max(0) as usize;
    let fk: Vec<f64> = (0..nk).map(|i| fmin * (STEP * i as f64).exp()).collect();
    let h: Vec<f64> = fk.iter().map(|v| (v * STEP / 2.0).max(df / 2.0)).collect();
    let lo: Vec<f64> = fk.iter().zip(&h).map(|(a, b)| a - b).collect();
    let hi: Vec<f64> = fk.iter().zip(&h).map(|(a, b)| a + b).collect();
    let d: Vec<f64> = e(&lo, &hi)?
        .iter()
        .zip(&h)
        .map(|(v, hh)| v / (2.0 * hh))
        .collect();
    let j1 = ((1.0 + ring.0).ln() / STEP).ceil() as isize;
    let j2 = ((1.0 + ring.1).ln() / STEP) as isize;
    let md = median(&d);
    let n = nk as isize;
    let n0: Vec<f64> = (0..n)
        .into_par_iter()
        .with_min_len(256)
        .map(|i| {
            let mut v = Vec::with_capacity(2 * (j2 - j1 + 1) as usize);
            for o in (-j2..=-j1).chain(j1..=j2) {
                v.push(d[reflect(i + o, n)]);
            }
            median(&v) + 1e-12 * md + 1e-30
        })
        .collect();
    let mut b = Vec::with_capacity(WIDTHS.len());
    let mut p_ = Vec::with_capacity(WIDTHS.len());
    let mut ex = Vec::with_capacity(WIDTHS.len());
    for w in WIDTHS {
        let bw: Vec<f64> = fk.iter().map(|v| (w * v).max(1.5 * df)).collect();
        let lo: Vec<f64> = fk.iter().zip(&bw).map(|(a, b)| a - b).collect();
        let hi: Vec<f64> = fk.iter().zip(&bw).map(|(a, b)| a + b).collect();
        let ew = e(&lo, &hi)?;
        p_.push(
            (0..nk)
                .map(|i| 10.0 * (ew[i] / (n0[i] * 2.0 * bw[i]) + 1e-30).log10())
                .collect::<Vec<f64>>(),
        );
        ex.push(
            (0..nk)
                .map(|i| (ew[i] - n0[i] * 2.0 * bw[i]).max(1e-30))
                .collect::<Vec<f64>>(),
        );
        b.push(bw);
    }
    let mut prom = vec![0.0; nk];
    let mut hw = vec![0.0; nk];
    let mut cc = vec![0.0; nk];
    let mut hc = vec![0.0; nk];
    let mut cn0 = vec![0.0; nk];
    for i in 0..nk {
        let col = [p_[0][i], p_[1][i], p_[2][i], p_[3][i]];
        let jb = argmax(&col).unwrap();
        prom[i] = col[jb];
        hw[i] = b[jb][i];
        let mut c = ex[jb][i];
        let mut hcv = hw[i];
        for j in 1..WIDTHS.len() {
            if j > jb && p_[j][i] >= MIN_PROM && ex[j][i] > 1.26 * c {
                c = ex[j][i];
                hcv = b[j][i];
            }
        }
        cc[i] = c;
        hc[i] = hcv;
        cn0[i] = 10.0 * (c / n0[i]).log10();
    }
    Ok(Grid {
        fk,
        prom,
        cn0,
        hw,
        hc,
        n0,
        d,
    })
}

/// `frames(S, fc, f0, hw, ring)` → (persistence, frame peak frequencies (NaN = absent)).
pub fn frames(s: &Frames, fc: &[f64], f0: f64, hw: f64, ring: (f64, f64)) -> (f64, Vec<f64>) {
    let df = fc[1] - fc[0];
    let b = hw.max(2e-3 * f0).max(1.5 * df);
    let band: Vec<usize> = (0..fc.len()).filter(|&i| (fc[i] - f0).abs() <= b).collect();
    let rlo = (ring.0 * f0).max(b + 2.0 * df);
    let rhi = (ring.1 * f0).max(b + 12.0 * df);
    let rng: Vec<usize> = (0..fc.len())
        .filter(|&i| {
            let a = (fc[i] - f0).abs();
            a >= rlo && a <= rhi
        })
        .collect();
    let n = s.nframes;
    if band.is_empty() || rng.len() < 4 || n == 0 {
        return (0.0, vec![f64::NAN; n]);
    }
    let mut on_n = 0usize;
    let mut fr = vec![f64::NAN; n];
    let last = fc.len() - 2;
    let mut rv = vec![0.0; rng.len()];
    for (j, out) in fr.iter_mut().enumerate() {
        let row = s.row(j);
        let a: Vec<f64> = band.iter().map(|&i| row[i]).collect();
        let k = argmax(&a).unwrap();
        let pk = a[k];
        for (r, &i) in rv.iter_mut().zip(&rng) {
            *r = row[i];
        }
        let on = 10.0 * (pk / (median(&rv) + 1e-30) + 1e-30).log10() >= FRAME_SNR;
        let i = band[k].clamp(1, last);
        let (a_, b_, c_) = (
            (row[i - 1] + 1e-30).ln(),
            (row[i] + 1e-30).ln(),
            (row[i + 1] + 1e-30).ln(),
        );
        let den = a_ - 2.0 * b_ + c_;
        let off = if den < 0.0 {
            0.5 * (a_ - c_) / den
        } else {
            0.0
        };
        if on {
            on_n += 1;
            *out = fc[i] + off.clamp(-1.0, 1.0) * df;
        }
    }
    (on_n as f64 / n as f64, fr)
}

/// One line detection in one block (`dict(ch, f, prom, cn0, hw, pers, wow, low, fr)` + `blk`).
#[derive(Debug, Clone)]
pub struct Det {
    /// channel index into the survey channel list
    pub ch: usize,
    pub f: f64,
    pub prom: f64,
    pub cn0: f64,
    pub hw: f64,
    pub pers: f64,
    pub wow: f64,
    pub low: bool,
    /// frame peak frequencies (high band only)
    pub fr: Option<Vec<f64>>,
    pub blk: usize,
}

/// Per-channel grid of a block (`spec[ch]`): `f`, `prom`, `D`.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    pub f: Vec<f64>,
    pub prom: Vec<f64>,
    pub d: Vec<f64>,
}

fn pow2_round(x: f64) -> usize {
    1usize << (x.log2().round_ties_even() as u32)
}

/// `detect_block(x, sr)`: detections and per-channel grids of one block (channels in
/// parallel, results in channel order). Detections carry `blk = 0` (set by the caller).
pub fn detect_block(x: &[f32], ch: usize, sr: u32, fmin: f64) -> Res<(Vec<Det>, Vec<BlockSpec>)> {
    let xs = channels(x, ch);
    let srf = sr as f64;
    let nf = pow2_round(0.75 * srf);
    let nc = pow2_round(0.093 * srf);
    let ff = rfftfreq(nf, 1.0 / srf);
    let fc = rfftfreq(nc, 1.0 / srf);
    let d = ((srf / 2400.0).floor() as i64).max(1) as usize;
    let (nt, cut, beta) = resample_poly_default_spec(d);
    let hlow = firwin_kaiser(nt, cut, beta, 2.0)?;
    let srl = srf / d as f64;
    let nlen = ts_dsp::conv::resample_poly_len(xs[0].1.len(), d);
    let nl = pow2_round(4.0 * srl).min(1usize << ((((nlen / 2).max(64)) as f64).log2() as u32));
    let fl = rfftfreq(nl, 1.0 / srl);
    let dist = ((3e-3 / STEP) as i64).max(1) as f64;
    type ChOut = Option<(Vec<Det>, BlockSpec)>;
    let per: Vec<ChOut> = xs
        .par_iter()
        .enumerate()
        .map(|(ci, (_, v))| -> Res<ChOut> {
            let (xf, wf) = spectra(v, nf, nf / 2)?;
            if xf.nframes == 0 {
                return Ok(None);
            }
            let (sc, _) = spectra(v, nc, nc / 2)?;
            let xl = if d == 1 {
                v.clone()
            } else {
                resample_poly_down(v, d, &hlow)?
            };
            let (sl, wl) = spectra(&xl, nl, nl / 2)?;
            let p = xf.psd(srf * wf);
            let pl = sl.psd(srl * wl);
            let g = line_grid(&ff, &p, fmin, 0.49 * srf, RING_HI)?;
            let gl = line_grid(&fl, &pl, 40.0, LOW_MAX.min(0.4 * srl), RING_LO)?;
            let mut out = Vec::new();
            for (gr, low) in [(&g, false), (&gl, true)] {
                let pr: Vec<f64> = gr
                    .prom
                    .iter()
                    .map(|&v| if v.is_finite() { v } else { -99.0 })
                    .collect();
                let pk = find_peaks(&pr, Some(MIN_PROM), Some(dist))?;
                for k in pk {
                    let (fq, pq) = if low { (&fl, &pl) } else { (&ff, &p) };
                    let m: Vec<usize> = (0..fq.len())
                        .filter(|&i| (fq[i] - gr.fk[k]).abs() <= gr.hc[k])
                        .collect();
                    let ex: Vec<f64> = m.iter().map(|&i| (pq[i] - gr.n0[k]).max(0.0)).collect();
                    let fx: Vec<f64> = m.iter().zip(&ex).map(|(&i, e)| fq[i] * e).collect();
                    let sx = np_sum(&ex);
                    let f0 = if sx > 0.0 { np_sum(&fx) / sx } else { gr.fk[k] };
                    let (pers, fr) = if low {
                        (frames(&sl, &fl, f0, gr.hw[k], RING_LO).0, None)
                    } else {
                        let (pp, fr) = frames(&sc, &fc, f0, gr.hw[k], RING_HI);
                        (pp, Some(fr))
                    };
                    let wow = match &fr {
                        Some(fr) => {
                            let on: Vec<f64> =
                                fr.iter().copied().filter(|v| v.is_finite()).collect();
                            if on.len() >= 10 {
                                std(&detrend_linear(&on)?) / f0 * 100.0
                            } else {
                                f64::NAN
                            }
                        }
                        None => f64::NAN,
                    };
                    out.push(Det {
                        ch: ci,
                        f: f0,
                        prom: gr.prom[k],
                        cn0: gr.cn0[k],
                        hw: gr.hw[k],
                        pers,
                        wow,
                        low,
                        fr,
                        blk: 0,
                    });
                }
            }
            Ok(Some((
                out,
                BlockSpec {
                    f: g.fk,
                    prom: g.prom,
                    d: g.d,
                },
            )))
        })
        .collect::<Res<_>>()?;
    let mut dets = Vec::new();
    let mut specs = Vec::new();
    for (ci, o) in per.into_iter().enumerate() {
        match o {
            Some((d, s)) => {
                dets.extend(d);
                specs.push(s);
            }
            None => {
                let _ = ci;
                specs.push(BlockSpec {
                    f: vec![],
                    prom: vec![],
                    d: vec![],
                });
            }
        }
    }
    Ok((dets, specs))
}

/// `group(dets, tol)`: single-linkage clusters by frequency (argsort order inside a cluster).
pub fn group(f: &[f64], tol: f64) -> Vec<Vec<usize>> {
    if f.is_empty() {
        return vec![];
    }
    let o = argsort(f);
    let mut out = vec![vec![o[0]]];
    for w in o.windows(2) {
        if f[w[1]] - f[w[0]] > tol * f[w[1]] {
            out.push(Vec::new());
        }
        out.last_mut().unwrap().push(w[1]);
    }
    out
}

/// `wow_corr(a, b)`: correlation of two frame-peak series (relative, detrended).
pub fn wow_corr(a: Option<&[f64]>, b: Option<&[f64]>) -> f64 {
    let (Some(a), Some(b)) = (a, b) else {
        return f64::NAN;
    };
    let m: Vec<usize> = (0..a.len().min(b.len()))
        .filter(|&i| a[i].is_finite() && b[i].is_finite())
        .collect();
    if a.len() != b.len() || m.len() < 30 {
        return f64::NAN;
    }
    let prep = |v: &[f64]| -> Option<Vec<f64>> {
        let s: Vec<f64> = m.iter().map(|&i| v[i]).collect();
        let mu = mean(&s);
        detrend_linear(&s.iter().map(|x| x / mu).collect::<Vec<_>>()).ok()
    };
    let (Some(x), Some(y)) = (prep(a), prep(b)) else {
        return f64::NAN;
    };
    if std(&x) > 0.0 && std(&y) > 0.0 {
        corrcoef(&x, &y).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

/// `classify(f, lossy, sr, search, play)` → (class, nominal, label, codec_risk).
pub fn classify(
    f: f64,
    lossy: bool,
    sr: f64,
    search: f64,
    play: f64,
) -> (String, Option<f64>, String, bool) {
    if f < LOW_MAX {
        let mut best = (f64::INFINITY, 0.0, 0usize);
        for b in [50.0, 60.0] {
            for k in 1..=20usize {
                let d = (f / (k as f64 * b * play) - 1.0).abs();
                if d < best.0 {
                    best = (d, b, k);
                }
            }
        }
        let lim = if best.2 <= 6 { search } else { HUM_EXACT };
        if best.0 <= lim {
            return (
                "hum".into(),
                Some(best.1 * best.2 as f64),
                format!("mains hum {} Hz h{}", ts_io::fmt_general(best.1, 6), best.2),
                false,
            );
        }
        return ("unknown".into(), None, "unknown stable line".into(), false);
    }
    let mut bi = 0;
    let mut bd = f64::INFINITY;
    for (i, (_, n, _)) in NOMINAL.iter().enumerate() {
        let d = (f / (n * play) - 1.0).abs();
        if d < bd {
            bd = d;
            bi = i;
        }
    }
    let (cls, nom, lab) = NOMINAL[bi];
    let risk = lossy && (f >= 15000.0 || (f / (sr / 4.0) - 1.0).abs() < 2e-3);
    if bd <= search {
        return (cls.into(), Some(nom), lab.into(), risk);
    }
    if risk {
        return (
            "codec".into(),
            None,
            format!(
                "likely codec artifact (lossy, {})",
                if f >= 15000.0 { ">= 15 kHz" } else { "~sr/4" }
            ),
            true,
        );
    }
    ("unknown".into(), None, "unknown stable line".into(), false)
}

/// `cliff(fk, D)`: lowest spectrum cliff (codec / anti-alias lowpass) or None.
pub fn cliff(fk: &[f64], d: &[f64]) -> Res<Option<f64>> {
    let s: Vec<f64> = d.iter().map(|v| 10.0 * (v + 1e-30).log10()).collect();
    let sel: Vec<f64> = fk
        .iter()
        .zip(&s)
        .filter(|(f, _)| **f > 1000.0 && **f < 2000.0)
        .map(|(_, v)| *v)
        .collect();
    let rf = median(&sel);
    let j = (1.03f64.ln() / STEP) as usize;
    let w = (1.1f64.ln() / STEP) as usize;
    let m = median_filter(&s, w | 1, Mode::Nearest)?;
    let n = s.len();
    let (k0, k1) = (
        w / 2,
        (n as isize - j as isize - (w / 2) as isize - 1).max(0) as usize,
    );
    for k in k0..k1.max(k0) {
        let (before, after) = (m[k - w / 2], m[k + j + w / 2]);
        if before - after >= 20.0 && after < rf - 60.0 && fk[k] > 2000.0 {
            return Ok(Some(fk[k] * 1.015));
        }
    }
    Ok(None)
}

/// `level_db(fk, D, lo, hi)`: median background in lo..hi rel. 1-2 kHz [dB] (NaN if empty).
pub fn level_db(fk: &[f64], d: &[f64], lo: f64, hi: f64) -> f64 {
    let a: Vec<f64> = fk
        .iter()
        .zip(d)
        .filter(|(f, _)| **f >= lo && **f <= hi)
        .map(|(_, v)| *v)
        .collect();
    if a.is_empty() {
        return f64::NAN;
    }
    let r: Vec<f64> = fk
        .iter()
        .zip(d)
        .filter(|(f, _)| **f > 1000.0 && **f < 2000.0)
        .map(|(_, v)| *v)
        .collect();
    10.0 * ((median(&a) + 1e-30) / (median(&r) + 1e-30)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Line classification by frequency (pilot, TV line, AM beat, ...).
    #[test]
    fn test_classes() {
        let c = |f: f64, lossy: bool| {
            let r = classify(f, lossy, 44100.0, 0.03, 1.0);
            (r.0, r.1)
        };
        assert_eq!(c(8934.0, false), ("AM beat".into(), Some(9000.0)));
        assert_eq!(c(9850.0, false), ("AM beat".into(), Some(10000.0)));
        assert_eq!(c(18640.0, false), ("FM pilot".into(), Some(19000.0)));
        assert_eq!(c(15625.0, false), ("TV line".into(), Some(15625.0)));
        assert_eq!(c(15734.0, false), ("TV line".into(), Some(15734.0)));
        assert_eq!(c(100.02, false), ("hum".into(), Some(100.0)));
        assert_eq!(c(49.0, false), ("hum".into(), Some(50.0)));
        assert_eq!(c(60.0, false), ("hum".into(), Some(60.0)));
        assert_eq!(c(851.0, false).0, "unknown");
        assert_eq!(c(12345.0, false).0, "unknown");
        assert_eq!(c(12345.0, true).0, "unknown");
        assert_eq!(c(20700.0, true).0, "codec");
        assert_eq!(c(11025.0, true).0, "codec");
        assert_eq!(c(20700.0, false).0, "unknown");
        assert!(classify(15620.0, true, 44100.0, 0.03, 1.0).3);
        let r = classify(9000.0 * 1.5, false, 48000.0, 0.03, 1.5);
        assert_eq!((r.0.as_str(), r.1), ("AM beat", Some(9000.0)));
    }

    #[test]
    fn blocks_and_group() {
        assert_eq!(
            blocks(24.0, 3, 8.0),
            vec![(0.0, 8.0), (8.0, 8.0), (16.0, 8.0)]
        );
        assert_eq!(blocks(100.0, 2, 20.0)[0], (25.0 - 10.0, 20.0));
        let g = group(&[1000.0, 3000.0, 1002.0, 3001.0], 4e-3);
        assert_eq!(g, vec![vec![0, 2], vec![1, 3]]);
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(5, 5), 3);
    }
}
