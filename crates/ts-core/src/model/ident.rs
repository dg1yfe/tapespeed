//! Component identification: masked spectrogram, per-frame ridges, greedy linking, laws,
//! duplicate merge and harmonic grouping.

use rayon::prelude::*;
use realfft::RealFftPlanner;
use ts_dsp::ValueError;
use ts_dsp::ndimage::{Mode, median_filter};
use ts_dsp::peaks::find_peaks;
use ts_dsp::spectral::rfftfreq;
use ts_dsp::stats::{argmin, median, np_sum};
use ts_dsp::window::{hanning, linspace};
use ts_io::numfmt::fmt_fixed;

use super::law::{Comp, Kind, fit_law};

/// Result of [`masked_spectrogram`]; `p` is frame-major (`tc.len() × f.len()`) (empty in
/// [`IdentInfo`] unless [`IdentOpts::keep_p`]).
#[derive(Debug, Clone)]
pub struct Spectrogram {
    pub tc: Vec<f64>,
    pub f: Vec<f64>,
    pub p: Vec<f64>,
    pub vf: Vec<f64>,
}

const FRAME_CHUNK: usize = 4;

/// `masked_spectrogram(t, y, w, win, hop, pad)`: Hann-windowed spectrogram of `y` on the
/// valid samples (`w > 0`), weighted mean removed per frame, normalised by the window-weighted
/// valid fraction. Frames in parallel (each computed independently).
pub fn masked_spectrogram(
    t: &[f64],
    y: &[f64],
    w: &[f64],
    win: f64,
    hop: f64,
    pad: usize,
) -> Result<Spectrogram, ValueError> {
    if t.len() < 2 {
        return Err(ValueError("masked_spectrogram: need >= 2 samples".into()));
    }
    let fs = 1.0 / (t[1] - t[0]);
    let n = (win * fs) as usize;
    let h = (hop * fs) as usize;
    if n == 0 || h == 0 || y.len() < n {
        return Err(ValueError(format!(
            "masked_spectrogram: window {n} / hop {h} vs {} samples",
            y.len()
        )));
    }
    let starts: Vec<usize> = (0..=y.len() - n).step_by(h).collect();
    let hw = hanning(n);
    let hw2: Vec<f64> = hw.iter().map(|v| v * v).collect();
    let shw2 = np_sum(&hw2);
    let nfft = pad * n;
    let f = rfftfreq(nfft, 1.0 / fs);
    let nb = f.len();
    let yw: Vec<f64> = y
        .iter()
        .zip(w)
        .map(|(&v, &wv)| if wv > 0.0 { v } else { 0.0 })
        .collect();
    let plan = RealFftPlanner::<f64>::new().plan_fft_forward(nfft);
    let nfr = starts.len();
    let mut p = vec![0.0; nfr * nb];
    let mut vf = vec![0.0; nfr];
    p.par_chunks_mut(nb * FRAME_CHUNK)
        .zip(vf.par_chunks_mut(FRAME_CHUNK))
        .enumerate()
        .for_each_init(
            || {
                (
                    vec![0.0; nfft],
                    plan.make_output_vec(),
                    plan.make_scratch_vec(),
                    vec![0.0; n],
                    vec![0.0; n],
                )
            },
            |(buf, spec, scratch, ww, tmp), (c, (prow, vrow))| {
                for (q, (row, vq)) in prow.chunks_mut(nb).zip(vrow.iter_mut()).enumerate() {
                    let s = starts[c * FRAME_CHUNK + q];
                    for i in 0..n {
                        ww[i] = if w[s + i] > 0.0 { hw[i] } else { 0.0 };
                    }
                    for i in 0..n {
                        tmp[i] = ww[i] * ww[i];
                    }
                    let sww2 = np_sum(tmp);
                    *vq = sww2 / shw2;
                    let seg = &yw[s..s + n];
                    for i in 0..n {
                        tmp[i] = seg[i] * ww[i];
                    }
                    let mu = np_sum(tmp) / np_sum(ww).max(1e-9);
                    buf.iter_mut().for_each(|v| *v = 0.0);
                    for i in 0..n {
                        buf[i] = (seg[i] - mu) * ww[i];
                    }
                    plan.process_with_scratch(buf, spec, scratch)
                        .expect("fft length");
                    let den = fs * sww2.max(1e-9);
                    for (o, x) in row.iter_mut().zip(spec.iter()) {
                        let a = x.norm();
                        *o = 2.0 * (a * a) / den;
                    }
                }
            },
        );
    let tc = starts.iter().map(|&s| t[s + n / 2]).collect();
    Ok(Spectrogram { tc, f, p, vf })
}

/// Per-frame peaks `(f, prominence dB, power)` from [`ridges`].
pub type FramePeaks = (Vec<f64>, Vec<f64>, Vec<f64>);

/// `ridges(tc, f, P, vf, fmin, fmax, prom_db, min_vf, ref_bw)`: per-frame spectral peaks
/// prominent above a running median in frequency, parabolic peak interpolation.
pub fn ridges(
    sg: &Spectrogram,
    fmin: f64,
    fmax: f64,
    prom_db: f64,
    min_vf: f64,
    ref_bw: f64,
) -> Vec<FramePeaks> {
    let f = &sg.f;
    let nb = f.len();
    let idx: Vec<usize> = (0..nb).filter(|&i| f[i] >= fmin && f[i] <= fmax).collect();
    let fi: Vec<f64> = idx.iter().map(|&i| f[i]).collect();
    let df = f[1] - f[0];
    let k = 3usize.max((ref_bw / df) as usize) | 1;
    let dist = 1usize.max((0.03 / df) as usize) as f64;
    (0..sg.tc.len())
        .into_par_iter()
        .map(|i| {
            if sg.vf[i] < min_vf || idx.is_empty() {
                return (vec![], vec![], vec![]);
            }
            let row = &sg.p[i * nb..(i + 1) * nb];
            let pm: Vec<f64> = idx.iter().map(|&j| row[j]).collect();
            let l: Vec<f64> = pm.iter().map(|v| 10.0 * (v + 1e-30).log10()).collect();
            let base = median_filter(&l, k, Mode::Nearest).expect("size >= 3");
            let d: Vec<f64> = l.iter().zip(&base).map(|(a, b)| a - b).collect();
            let pk = find_peaks(&d, Some(prom_db), Some(dist)).expect("distance >= 1");
            let last = l.len() - 1;
            let mut fo = Vec::with_capacity(pk.len());
            let mut pr = Vec::with_capacity(pk.len());
            let mut pw = Vec::with_capacity(pk.len());
            for &q in &pk {
                let a_ = l[q.saturating_sub(1)];
                let b_ = l[q];
                let c_ = l[(q + 1).min(last)];
                let den = a_ - 2.0 * b_ + c_;
                let off = if den < 0.0 {
                    0.5 * (a_ - c_) / den
                } else {
                    0.0
                };
                fo.push(fi[q] + off.clamp(-0.5, 0.5) * df);
                pr.push(l[q] - base[q]);
                pw.push(pm[q]);
            }
            (fo, pr, pw)
        })
        .collect()
}

/// A linked ridge track (output of [`link`]): frame indices, frequency, prominence, power.
#[derive(Debug, Clone, Default)]
pub struct Track {
    pub i: Vec<usize>,
    pub f: Vec<f64>,
    pub pr: Vec<f64>,
    pub p: Vec<f64>,
}

/// `link(tc, peaks, tol_abs, tol_rel, max_gap)`: greedy ridge linking (prediction by the
/// last slope), tracks in creation order.
pub fn link(peaks: &[FramePeaks], tol_abs: f64, tol_rel: f64, max_gap: usize) -> Vec<Track> {
    let mut tracks: Vec<Track> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for (i, (fp, pr, pw)) in peaks.iter().enumerate() {
        let mut used = vec![false; fp.len()];
        // sorted(active, key=-mean(pr)): stable
        let mut order: Vec<(f64, usize)> = active
            .iter()
            .map(|&a| (-(np_sum(&tracks[a].pr) / tracks[a].pr.len() as f64), a))
            .collect();
        order.sort_by(|x, y| x.0.partial_cmp(&y.0).expect("finite prominence"));
        let mut d = vec![0.0; fp.len()];
        for &(_, a) in &order {
            let tr = &mut tracks[a];
            let nl = tr.f.len();
            let last_i = tr.i[nl - 1];
            let gap = (i - last_i) as f64;
            let slope = if nl >= 3 {
                (tr.f[nl - 1] - tr.f[nl - 3]) / (last_i - tr.i[nl - 3]) as f64
            } else {
                0.0
            };
            let pred = tr.f[nl - 1] + slope * gap;
            let tol = (tol_abs + tol_rel * pred) * gap.min(2.0);
            if !fp.is_empty() {
                for (j, dv) in d.iter_mut().enumerate() {
                    *dv = (fp[j] - pred).abs() + if used[j] { 1e9 } else { 0.0 };
                }
                let j = argmin(&d).expect("non-empty");
                if d[j] < tol {
                    used[j] = true;
                    tr.i.push(i);
                    tr.f.push(fp[j]);
                    tr.pr.push(pr[j]);
                    tr.p.push(pw[j]);
                }
            }
        }
        for j in 0..fp.len() {
            if !used[j] {
                active.push(tracks.len());
                tracks.push(Track {
                    i: vec![i],
                    f: vec![fp[j]],
                    pr: vec![pr[j]],
                    p: vec![pw[j]],
                });
            }
        }
        active.retain(|&a| i - tracks[a].i[tracks[a].i.len() - 1] < max_gap);
    }
    tracks
}

/// Options of [`identify`] (with their defaults).
#[derive(Debug, Clone)]
pub struct IdentOpts {
    pub fmin: f64,
    pub fmax: f64,
    pub win: f64,
    pub hop: f64,
    pub prom_db: f64,
    pub min_frames: Option<usize>,
    pub max_comps: usize,
    pub harm_tol: f64,
    /// Keep the spectrogram power `P` in [`IdentInfo`] (nothing in
    /// the pipeline reads it, and it is ~32 MB at 43 min → off by default, freed after the
    /// ridges).
    pub keep_p: bool,
}

impl Default for IdentOpts {
    fn default() -> Self {
        IdentOpts {
            fmin: 0.2,
            fmax: 22.0,
            win: 60.0,
            hop: 15.0,
            prom_db: 9.0,
            min_frames: None,
            max_comps: 14,
            harm_tol: 0.01,
            keep_p: false,
        }
    }
}

impl IdentOpts {
    /// `model.ident_kw(dur)`: window 60 s for ≥ 10 min, ≥ 15 s; hop = win / 4.
    pub fn for_duration(dur: f64) -> Self {
        let win = 60.0f64.min(15.0f64.max(dur / 10.0));
        IdentOpts {
            win,
            hop: win / 4.0,
            ..IdentOpts::default()
        }
    }
}

/// Candidate track after the law fit (`cand` dict of `identify`).
#[derive(Debug, Clone)]
pub struct Cand {
    pub comp: Comp,
    pub tt: Vec<f64>,
    pub ff: Vec<f64>,
    /// median power
    pub p: f64,
    pub n: usize,
    pub rms: f64,
}

/// Harmonic family: `root` and members `(h, index)` into [`IdentInfo::tracks`], in insertion
/// order (part of the report format).
#[derive(Debug, Clone)]
pub struct Family {
    pub root: usize,
    pub harm: Vec<(u32, usize)>,
}

/// `info` of `identify`.
#[derive(Debug, Clone)]
pub struct IdentInfo {
    pub sg: Spectrogram,
    /// merged tracks (`uniq`)
    pub tracks: Vec<Cand>,
    pub families: Vec<Family>,
}

fn median_abs_diff(a: &[f64], b: &[f64], hb: f64) -> f64 {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - hb * y).abs()).collect();
    median(&d)
}

fn mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// `identify(t, y, w, ...)`: periodic component families in `y = r - r0` (valid where
/// `w > 0`). Returns the comps sorted by `f_mean` and the identification info.
pub fn identify(
    t: &[f64],
    y: &[f64],
    w: &[f64],
    o: &IdentOpts,
) -> Result<(Vec<Comp>, IdentInfo), ValueError> {
    let mut sg = masked_spectrogram(t, y, w, o.win, o.hop, 4)?;
    let peaks = ridges(&sg, o.fmin, o.fmax, o.prom_db, 0.5, 0.6);
    if !o.keep_p {
        sg.p = Vec::new();
    }
    let tracks = link(&peaks, 0.025, 0.006, 4);
    let nvalid = sg.vf.iter().filter(|&&v| v >= 0.5).count();
    let min_frames = o
        .min_frames
        .unwrap_or_else(|| 4usize.max((0.3 * nvalid as f64) as usize));
    let span = (t[0], t[t.len() - 1]);
    let mut cand: Vec<Cand> = Vec::new();
    for tr in &tracks {
        if tr.i.len() < min_frames {
            continue;
        }
        let tt: Vec<f64> = tr.i.iter().map(|&i| sg.tc[i]).collect();
        let ff = tr.f.clone();
        let wp: Vec<f64> = tr.p.iter().map(|v| v.sqrt()).collect();
        let law = fit_law(&tt, &ff, Some(&wp), Some(span), 161);
        let mut c = Comp::new("", law.a, law.b, law.beta);
        c.kind = law.kind;
        c.f_mean = c.mean_freq(span);
        c.amp_db = median(&tr.pr);
        cand.push(Cand {
            comp: c,
            n: tt.len(),
            tt,
            ff,
            p: median(&tr.p),
            rms: law.rms,
        });
    }
    // merge duplicates (same law within tolerance), strongest first (stable sort)
    cand.sort_by(|a, b| {
        let (ka, kb) = (-a.p * a.n as f64, -b.p * b.n as f64);
        ka.partial_cmp(&kb).expect("finite")
    });
    let tg = linspace(span.0, span.1, 100);
    let mut uniq: Vec<Cand> = Vec::new();
    let mut ufreq: Vec<Vec<f64>> = Vec::new();
    for d in cand {
        let fd = d.comp.freq(&tg);
        let lim = 0.03 + 0.004 * mean(&fd);
        if ufreq.iter().any(|fu| median_abs_diff(&fd, fu, 1.0) < lim) {
            continue;
        }
        ufreq.push(fd);
        uniq.push(d);
    }
    // harmonic grouping: d is harmonic h of a lower family u
    let mut by_f: Vec<usize> = (0..uniq.len()).collect();
    by_f.sort_by(|&a, &b| {
        uniq[a]
            .comp
            .f_mean
            .partial_cmp(&uniq[b].comp.f_mean)
            .expect("finite")
    });
    let mut fams: Vec<Family> = Vec::new();
    for &d in &by_f {
        let fd = &ufreq[d];
        let mut placed = false;
        for u in fams.iter_mut() {
            let fu = &ufreq[u.root];
            let ratio: Vec<f64> = fd.iter().zip(fu).map(|(a, b)| a / b).collect();
            let hr = median(&ratio).round_ties_even();
            if (2.0..=3.0).contains(&hr)
                && median_abs_diff(fd, fu, hr) < 0.02 + o.harm_tol * mean(fd)
            {
                let h = hr as u32;
                match u.harm.iter_mut().find(|(hh, _)| *hh == h) {
                    Some(e) => e.1 = d,
                    None => u.harm.push((h, d)),
                }
                placed = true;
                break;
            }
        }
        if !placed {
            fams.push(Family {
                root: d,
                harm: vec![(1, d)],
            });
        }
    }
    // refine family laws from all members (f_h / h), strongest families first
    let mut order: Vec<usize> = (0..fams.len()).collect();
    order.sort_by(|&a, &b| {
        let (ua, ub) = (&uniq[fams[a].root], &uniq[fams[b].root]);
        let (ka, kb) = (-ua.p * ua.n as f64, -ub.p * ub.n as f64);
        ka.partial_cmp(&kb).expect("finite")
    });
    let mut comps = Vec::new();
    for &fi in order.iter().take(o.max_comps) {
        let fam = &fams[fi];
        let (mut tt, mut ff, mut ww) = (vec![], vec![], vec![]);
        for &(h, m) in &fam.harm {
            let mm = &uniq[m];
            tt.extend_from_slice(&mm.tt);
            ff.extend(mm.ff.iter().map(|v| v / h as f64));
            ww.extend(std::iter::repeat_n((h * h) as f64, mm.tt.len()));
        }
        let law = fit_law(&tt, &ff, Some(&ww), Some(span), 161);
        let mut c = Comp::new("", law.a, law.b, law.beta);
        c.kind = law.kind;
        c.amp_db = uniq[fam.root].comp.amp_db;
        c.f_mean = c.mean_freq(span);
        let fmx = c.freq_at(span.0).max(c.freq_at(span.1));
        c.harmonics = (1..=3u32)
            .filter(|&h| h as f64 * fmx < o.fmax + 1.0)
            .collect();
        c.name = format!(
            "{}{}",
            if law.kind == Kind::Chirp {
                "chirp"
            } else {
                "f"
            },
            fmt_fixed(c.f_mean, 3)
        );
        c.pts = Some((tt, ff, ww));
        comps.push(c);
    }
    comps.sort_by(|a, b| a.f_mean.partial_cmp(&b.f_mean).expect("finite"));
    Ok((
        comps,
        IdentInfo {
            sg,
            tracks: uniq,
            families: fams,
        },
    ))
}
