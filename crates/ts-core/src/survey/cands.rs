//! Survey candidates: `summarise`, `tv_region`, `score`, `tape_corr`, `verdict`,
//! `usable`, `coherence_check`, `pair_hz`, `pair_check`.

use rayon::prelude::*;
use ts_dsp::stats::{median, nanmedian, percentiles, ptp};
use ts_io::{fmt_fixed, fmt_general, round_dec};

use super::detect::*;
use super::{Coh, PairCheck, Res, SurveyError, Tracker, Trial, TrialPair};
use crate::absolute::pair::{PairParams, pair_measure};
use crate::evidence::{BANDS, COH_MIN, band_stats};
use crate::hetdec::Channel;

/// Per-block entry of a candidate (`blocks[k]`: rounded f / cn0 / prom).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockEntry {
    pub f: f64,
    pub cn0: f64,
    pub prom: f64,
}

/// Per-channel statistics (`channels[ch]`).
#[derive(Debug, Clone, PartialEq)]
pub struct ChanStat {
    pub name: String,
    pub n: usize,
    pub lock: usize,
    /// rounded to 0.1 dB; None without detections
    pub cn0: Option<f64>,
}

/// One candidate line (JSON key order defined in report.rs).
#[derive(Debug, Clone)]
pub struct Cand {
    /// rounded to 0.01 Hz
    pub f: f64,
    pub cls: String,
    pub label: String,
    pub nominal: Option<f64>,
    pub codec_risk: bool,
    pub ch: String,
    pub low: bool,
    pub n: usize,
    pub nb: usize,
    pub presence: f64,
    pub lock_blocks: usize,
    pub prom: f64,
    pub cn0: f64,
    pub pers: f64,
    pub wow_pct: f64,
    pub spread_hz: [f64; 2],
    pub spread_pct: f64,
    /// sorted by block index
    pub blocks: Vec<(usize, BlockEntry)>,
    pub channels: Vec<ChanStat>,
    /// Per channel (survey channel order) the best detection per block, in first-seen order
    pub per: Vec<Vec<(usize, usize)>>,
    pub ratio: Option<f64>,
    pub offset_pct: Option<f64>,
    pub sigma25_pct: f64,
    /// `wow_corr` (set by tape_corr for high-band lines; inner None = null)
    pub wow_corr: Option<Option<f64>>,
    pub tape: Option<Option<bool>>,
    pub primary: bool,
    pub verdict: String,
    pub trial_valid: Option<f64>,
    pub coh: Option<Coh>,
    pub id: usize,
}

impl Cand {
    pub fn block(&self, b: usize) -> Option<&BlockEntry> {
        self.blocks.iter().find(|(k, _)| *k == b).map(|(_, e)| e)
    }

    pub fn chan(&self, name: &str) -> Option<&ChanStat> {
        self.channels.iter().find(|c| c.name == name)
    }

    fn best_ch_idx(&self) -> usize {
        self.channels
            .iter()
            .position(|c| c.name == self.ch)
            .unwrap()
    }

    /// `per[ch][b]` detection index for the candidate's best channel.
    fn per_best(&self, b: usize) -> Option<usize> {
        self.per[self.best_ch_idx()]
            .iter()
            .find(|(k, _)| *k == b)
            .map(|(_, i)| *i)
    }
}

/// `max(seq, key)`: first maximal element (`gt(a, b)` = a strictly greater).
pub(crate) fn first_max<T: Copy>(
    it: impl IntoIterator<Item = T>,
    gt: impl Fn(&T, &T) -> bool,
) -> Option<T> {
    let mut best: Option<T> = None;
    for v in it {
        match &best {
            Some(b) if !gt(&v, b) => {}
            _ => best = Some(v),
        }
    }
    best
}

/// `summarise(dets, idx, nb, chans, inf, search, play)`.
#[allow(clippy::too_many_arguments)]
pub fn summarise(
    dets: &[Det],
    idx: &[usize],
    nb: usize,
    chans: &[String],
    lossy: bool,
    sr: f64,
    search: f64,
    play: f64,
) -> Res<Cand> {
    let mut per: Vec<Vec<(usize, usize)>> = vec![Vec::new(); chans.len()];
    for &i in idx {
        let d = &dets[i];
        let m = &mut per[d.ch];
        match m.iter_mut().find(|(b, _)| *b == d.blk) {
            None => m.push((d.blk, i)),
            Some(e) => {
                if d.prom > dets[e.1].prom {
                    e.1 = i;
                }
            }
        }
    }
    let st: Vec<(usize, usize, f64)> = per
        .iter()
        .map(|v| {
            let cn: Vec<f64> = v.iter().map(|(_, i)| dets[*i].cn0).collect();
            let lock = cn.iter().filter(|&&c| c >= LOCK_DB).count();
            (
                v.len(),
                lock,
                if cn.is_empty() {
                    f64::NEG_INFINITY
                } else {
                    median(&cn)
                },
            )
        })
        .collect();
    let best = first_max(0..chans.len(), |&a, &b| {
        let (x, y) = (st[a], st[b]);
        (x.1, x.0)
            .cmp(&(y.1, y.0))
            .then(x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal))
            == std::cmp::Ordering::Greater
    })
    .unwrap();
    let mut bl: Vec<(usize, usize)> = per[best].clone();
    bl.sort_by_key(|(k, _)| *k);
    let fb: Vec<f64> = bl.iter().map(|(_, i)| dets[*i].f).collect();
    let f = median(&fb);
    let (cls, nom, lab, risk) = classify(f, lossy, sr, search, play);
    let med = |g: &dyn Fn(&Det) -> f64| -> f64 {
        let v: Vec<f64> = bl.iter().map(|(_, i)| g(&dets[*i])).collect();
        if v.iter().any(|x| x.is_finite()) {
            nanmedian(&v)
        } else {
            f64::NAN
        }
    };
    let q = percentiles(&fb, &[5.0, 95.0])?;
    let ratio = nom.map(|n| f / (n * play));
    let cn0 = st[best].2;
    Ok(Cand {
        f: round_dec(f, 2),
        cls,
        label: lab,
        nominal: nom,
        codec_risk: risk,
        ch: chans[best].clone(),
        low: dets[bl[0].1].low,
        n: bl.len(),
        nb,
        presence: bl.len() as f64 / nb as f64,
        lock_blocks: st[best].1,
        prom: med(&|d| d.prom),
        cn0,
        pers: med(&|d| d.pers),
        wow_pct: med(&|d| d.wow),
        spread_hz: [round_dec(q[0], 2), round_dec(q[1], 2)],
        spread_pct: ptp(&q).unwrap() / f * 100.0,
        blocks: bl
            .iter()
            .map(|(k, i)| {
                let d = &dets[*i];
                (
                    *k,
                    BlockEntry {
                        f: round_dec(d.f, 2),
                        cn0: round_dec(d.cn0, 1),
                        prom: round_dec(d.prom, 1),
                    },
                )
            })
            .collect(),
        channels: chans
            .iter()
            .zip(&st)
            .map(|(name, s)| ChanStat {
                name: name.clone(),
                n: s.0,
                lock: s.1,
                cn0: s.2.is_finite().then(|| round_dec(s.2, 1)),
            })
            .collect(),
        per,
        ratio,
        offset_pct: ratio.map(|r| (r - 1.0) * 100.0),
        sigma25_pct: sigma_ratio(cn0, f, PB_FULL) * 100.0,
        wow_corr: None,
        tape: None,
        primary: false,
        verdict: String::new(),
        trial_valid: None,
        coh: None,
        id: 0,
    })
}

/// `tv_region(cands, search, play)`.
pub fn tv_region(cands: &mut [Cand], search: f64, play: f64) {
    let beats: Vec<f64> = cands
        .iter()
        .filter(|c| c.cls == "AM beat" && c.pers >= 0.8)
        .filter_map(|c| c.nominal)
        .collect();
    let pref = if beats.contains(&9000.0) {
        Some(15625.0)
    } else if beats.contains(&10000.0) {
        Some(15734.0)
    } else {
        None
    };
    for c in cands.iter_mut() {
        if c.cls == "TV line"
            && [15625.0, 15734.0]
                .iter()
                .all(|n| (c.f / (n * play) - 1.0).abs() <= search)
        {
            let nom = pref.or(c.nominal).unwrap();
            c.nominal = Some(nom);
            let r = c.f / (nom * play);
            c.ratio = Some(r);
            c.label = format!(
                "TV line whistle ({}{}",
                if nom == 15625.0 { "PAL" } else { "NTSC" },
                if pref.is_some() {
                    ", region from the AM beat)"
                } else {
                    ", nearest; PAL/NTSC ambiguous)"
                }
            );
            c.offset_pct = Some((r - 1.0) * 100.0);
        }
    }
}

/// `score(c)`: lockable dB summed over blocks.
pub fn score(c: &Cand) -> f64 {
    let mut s = 0.0;
    for (_, e) in &c.blocks {
        let v = e.cn0 - LOCK_DB;
        s += if 0.0 > v { 0.0 } else { v };
    }
    s
}

/// `tape_corr(cands)`: frame-wow correlation with the primary; returns the primary's index.
pub fn tape_corr(cands: &mut [Cand], dets: &[Det]) -> Option<usize> {
    let hi: Vec<usize> = (0..cands.len()).filter(|&i| !cands[i].low).collect();
    type Pred = fn(&Cand) -> bool;
    let pools: [Pred; 3] = [
        |c| c.pers >= 0.8 && !c.codec_risk && c.cn0 >= LOCK_DB,
        |c| c.pers >= 0.8 && !c.codec_risk,
        |c| !c.codec_risk,
    ];
    let mut pool: Vec<usize> = Vec::new();
    for f in &pools {
        pool = hi.iter().copied().filter(|&i| f(&cands[i])).collect();
        if !pool.is_empty() {
            break;
        }
    }
    if pool.is_empty() {
        pool = hi.clone();
    }
    let p = first_max(pool, |&a, &b| score(&cands[a]) > score(&cands[b]))?;
    for &ci in &hi {
        let mut rs = Vec::new();
        if ci != p {
            let c = &cands[ci];
            let pc = &cands[p];
            for &(b, di) in &c.per[c.best_ch_idx()] {
                if let Some(pi) = pc.per_best(b) {
                    let r = wow_corr(dets[di].fr.as_deref(), dets[pi].fr.as_deref());
                    if r.is_finite() {
                        rs.push(r);
                    }
                }
            }
        }
        let c = &mut cands[ci];
        let wc = (!rs.is_empty()).then(|| median(&rs));
        c.wow_corr = Some(wc);
        c.tape = Some(wc.map(|w| w >= TAPE_CORR));
        if c.codec_risk && wc.is_some_and(|w| w < FREE_CORR) && c.cls != "codec" {
            c.label = format!(
                "likely codec artifact (lossy, no shared wow; near {})",
                c.label
            );
            c.cls = "codec".into();
            c.nominal = None;
            c.ratio = None;
            c.offset_pct = None;
        }
    }
    let c = &mut cands[p];
    c.wow_corr = Some(Some(1.0));
    c.tape = Some(None);
    c.primary = true;
    Some(p)
}

/// `verdict(c)`.
pub fn verdict(c: &Cand) -> String {
    if c.cls == "codec" {
        return "not usable (codec artifact)".into();
    }
    if c.cls == "hum" {
        return if (c.ratio.unwrap() - 1.0).abs() < HUM_EXACT {
            "not usable: transfer/playback hum (exact nominal)".into()
        } else {
            format!(
                "hum (record or playback: not distinguishable), sigma25 {} %",
                fmt_general(c.sigma25_pct, 2)
            )
        };
    }
    let mut v = String::from(if c.cn0 < LOCK_DB {
        "no lock"
    } else if c.cn0 < MARGIN_DB {
        "marginal"
    } else {
        "trackable"
    });
    if c.lock_blocks < c.nb {
        v += &format!(", ref in {}/{} blocks", c.lock_blocks, c.nb);
    }
    if c.codec_risk && !matches!(c.tape, Some(Some(true))) {
        v += ", lossy: may be codec";
    }
    v
}

/// Whether candidate `c` is usable as a reference line.
pub fn usable(c: &Cand) -> bool {
    c.cls != "codec"
        && c.cls != "hum"
        && !c.low
        && c.cn0 >= LOCK_DB
        && c.trial_valid.unwrap_or(1.0) >= TRIAL_MIN
}

/// Channel selection from its name.
pub fn chan_of(ch: &str) -> Channel {
    match ch {
        "L" => Channel::L,
        "R" => Channel::R,
        "S" => Channel::S,
        _ => Channel::Mid,
    }
}

pub fn chan_name(ch: &str) -> &str {
    if ch == "mono" { "mid" } else { ch }
}

/// Trial-track excerpt start: `clip(centre(b) - L/2, 0, max(0, dur - L))`.
fn trial_t0(bl: &[(f64, f64)], b: usize, l: f64, dur: f64) -> f64 {
    let x = bl[b].0 + bl[b].1 / 2.0 - l / 2.0;
    x.max(0.0).min(0.0f64.max(dur - l))
}

/// Context of the trial tracks: file, duration, blocks, audio loader, tracker.
pub struct TrialCtx<'a> {
    pub path: &'a str,
    pub duration: f64,
    pub blocks: &'a [(f64, f64)],
    pub tracker: &'a dyn Tracker,
}

impl TrialCtx<'_> {
    /// `audio.load(path, t0, min(L, dur))` + `analyze(x, sr, f, 0.01, "auto", chan)`.
    pub(crate) fn run(&self, t0: f64, f: f64, ch: &str) -> Res<Trial> {
        let l = 60.0f64.min(self.duration);
        let a = ts_io::load(self.path, Some(t0), Some(l)).map_err(SurveyError::Io)?;
        let curve = self
            .tracker
            .track(&a.data, a.channels, a.sr, f, chan_of(ch), t0, l)
            .map_err(SurveyError::Value)?;
        Ok(Trial {
            t0,
            dur: a.frames() as f64 / a.sr as f64,
            curve,
            audio: a,
        })
    }
}

fn mean_bool(v: &[bool]) -> f64 {
    v.iter().filter(|b| **b).count() as f64 / v.len() as f64
}

/// `coherence_check(path, inf, p, others)`: trial tracks (computed in parallel, bookkeeping in
/// a fixed call order) and band coherence of the primary with each other line.
pub fn coherence_check(
    ctx: &TrialCtx,
    cands: &mut [Cand],
    p: usize,
    others: &[usize],
) -> Res<Vec<Coh>> {
    let l = 60.0;
    let strongest = |c: &Cand| -> usize {
        first_max(c.blocks.iter().map(|(k, e)| (*k, e.cn0)), |a, b| a.1 > b.1)
            .unwrap()
            .0
    };
    // call sequence of track(c, b)
    enum Step {
        Track(usize, usize),
        Pair(usize, usize),
    }
    let mut steps = vec![Step::Track(p, strongest(&cands[p]))];
    for &o in others {
        let com: Vec<usize> = cands[p]
            .blocks
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| cands[o].block(*k).is_some())
            .collect();
        if com.is_empty() {
            steps.push(Step::Track(o, strongest(&cands[o])));
            continue;
        }
        let mn = |k: usize| {
            cands[p]
                .block(k)
                .unwrap()
                .cn0
                .min(cands[o].block(k).unwrap().cn0)
        };
        let b = first_max(com, |&a, &bb| mn(a) > mn(bb)).unwrap();
        steps.push(Step::Pair(o, b));
    }
    let mut keys: Vec<(usize, usize)> = Vec::new(); // (cand, block) in first-call order
    for s in &steps {
        let ks = match *s {
            Step::Track(c, b) => vec![(c, b)],
            Step::Pair(o, b) => vec![(p, b), (o, b)],
        };
        for k in ks {
            let dup = keys
                .iter()
                .any(|&(c, b)| b == k.1 && cands[c].f == cands[k.0].f);
            if !dup {
                keys.push(k);
            }
        }
    }
    let trials: Vec<Trial> = keys
        .par_iter()
        .map(|&(c, b)| {
            let cd = &cands[c];
            ctx.run(
                trial_t0(ctx.blocks, b, l, ctx.duration),
                cd.block(b).unwrap().f,
                chan_name(&cd.ch),
            )
        })
        .collect::<Res<_>>()?;
    let cf: Vec<f64> = cands.iter().map(|c| c.f).collect();
    let find = |c: usize, b: usize| -> usize {
        keys.iter()
            .position(|&(cc, bb)| bb == b && cf[cc] == cf[c])
            .unwrap()
    };
    let mut seen = vec![false; keys.len()];
    let mut touch = |cands: &mut [Cand], c: usize, b: usize| -> usize {
        let i = find(c, b);
        if !seen[i] {
            seen[i] = true;
            let v = mean_bool(&trials[i].curve.valid);
            let cd = &mut cands[c];
            cd.trial_valid = Some(cd.trial_valid.unwrap_or(0.0).max(v));
        }
        i
    };
    let mut out = Vec::new();
    for s in steps {
        match s {
            Step::Track(c, b) => {
                touch(cands, c, b);
            }
            Step::Pair(o, b) => {
                let i1 = touch(cands, p, b);
                let i2 = touch(cands, o, b);
                let (t1, c1, c2) = (&trials[i1], &trials[i1].curve, &trials[i2].curve);
                let n = c1.t.len().min(c2.t.len());
                let v: Vec<bool> = (0..n).map(|k| c1.valid[k] && c2.valid[k]).collect();
                let q = if v.iter().any(|x| *x) {
                    let r: Vec<f64> = (0..n)
                        .filter(|&k| v[k])
                        .map(|k| c1.ratio[k] / c2.ratio[k])
                        .collect();
                    median(&r)
                } else {
                    1.0
                };
                let r1: Vec<f64> = c1.ratio[..n].iter().map(|x| x / q).collect();
                let st = band_stats(&c1.t[..n], &r1, &c2.ratio[..n], &v, &BANDS, 2048, 0.5)
                    .map_err(|e| SurveyError::Value(e.to_string()))?;
                let coh: Vec<(String, f64)> = st
                    .coh
                    .iter()
                    .map(|(k, x)| (k.clone(), round_dec(*x, 4)))
                    .collect();
                let tape = !coh.is_empty()
                    && coh.iter().find(|(k, _)| k == "0.5-6").map_or(0.0, |x| x.1) >= COH_MIN;
                let rec = Coh {
                    a: cands[p].f,
                    b: cands[o].f,
                    t0: round_dec(t1.t0, 1),
                    dur: round_dec(t1.dur, 1),
                    common: mean_bool(&v),
                    n_seg: st.n_seg,
                    coh,
                    tape,
                };
                cands[o].coh = Some(rec.clone());
                out.push(rec);
            }
        }
    }
    Ok(out)
}

/// `pair_hz(p, search, play)`.
pub fn pair_hz(p: &Cand, search: f64, play: f64) -> Option<f64> {
    if p.cls == "AM beat" {
        return p.nominal;
    }
    if p.cls == "unknown" && !p.low {
        let mut best = (f64::INFINITY, 0.0);
        for (c, n, _) in NOMINAL {
            if c == "AM beat" {
                let d = (p.f / (n * play) - 1.0).abs();
                if d < best.0 {
                    best = (d, n);
                }
            }
        }
        return (best.0 <= 2.0 * search).then_some(best.1);
    }
    None
}

/// `np.array_split(ks, parts)` → parts (first `len % parts` get one more).
fn array_split(ks: &[usize], parts: usize) -> Vec<Vec<usize>> {
    let (q, r) = (ks.len() / parts, ks.len() % parts);
    let mut out = Vec::new();
    let mut s = 0;
    for i in 0..parts {
        let l = q + usize::from(i < r);
        out.push(ks[s..s + l].to_vec());
        s += l;
    }
    out
}

/// `pair_check(path, inf, p, hz, xcheck, L, n, dmax, play)`: trial `--abs pair` on up to 3
/// excerpts (computed in parallel).
pub fn pair_check(ctx: &TrialCtx, p: &Cand, hz: f64, xcheck: &[f64], play: f64) -> Res<PairCheck> {
    let (l, n, dmax) = (60.0, 3usize, 300.0);
    let ks: Vec<usize> = p.blocks.iter().map(|(k, _)| *k).collect();
    let mut pick = Vec::new();
    for part in array_split(&ks, n.min(ks.len())) {
        if let Some(b) = first_max(part, |&a, &b| {
            p.block(a).unwrap().cn0 > p.block(b).unwrap().cn0
        }) {
            pick.push(b);
        }
    }
    let chn = chan_name(&p.ch).to_string();
    let res: Vec<(TrialPair, Option<crate::absolute::pair::PairResult>)> = pick
        .par_iter()
        .map(|&b| -> Res<_> {
            let t0 = trial_t0(ctx.blocks, b, l, ctx.duration);
            let fb = p.block(b).unwrap().f;
            let tr = ctx.run(t0, fb, &chn)?;
            let valid = mean_bool(&tr.curve.valid);
            let mut t = TrialPair {
                t0: round_dec(t0, 1),
                valid,
                ..Default::default()
            };
            if valid < TRIAL_MIN {
                t.error = Some(format!(
                    "primary not tracked ({} % valid)",
                    fmt_fixed(100.0 * valid, 0)
                ));
                return Ok((t, None));
            }
            let o = PairParams {
                hz,
                dmax,
                channel: chan_of(&chn),
                blk_s: Some(20.0),
                xcheck: xcheck.to_vec(),
                play,
                ..Default::default()
            };
            let pm = pair_measure(
                &tr.audio.data,
                tr.audio.channels,
                tr.audio.sr as f64,
                &tr.curve,
                fb,
                &o,
            )
            .map_err(|e| SurveyError::Value(e.to_string()))?;
            t.ambiguity = Some(pm.ambiguity.clone());
            t.n_cand = pm.candidates.len();
            t.n_ok = pm.candidates.iter().filter(|e| e.why.is_empty()).count();
            let mut keep = None;
            if let Some(e) = &pm.error {
                t.error = Some(e.clone());
            } else if let Some(s) = &pm.solution {
                let e = &pm.candidates[s.best];
                t.sol = Some(super::TrialSol {
                    s: s.s,
                    f_true: s.f_true,
                    f_lo: s.f_lo,
                    f_hi: s.f_hi,
                    delta: s.delta,
                    side: s.side.as_str().to_string(),
                    prom: e.prom,
                    pres: e.pres,
                    k: e.k_hat,
                    xcheck: pm
                        .xcheck
                        .iter()
                        .filter_map(|x| x.ppm.map(|v| (fmt_general(x.nom, 6), v)))
                        .collect(),
                });
                keep = Some(pm.clone());
            }
            t.cands = Some(
                pm.candidates
                    .iter()
                    .map(|e| super::PairCand {
                        f: round_dec(e.f, 2),
                        prom: round_dec(e.prom, 1),
                        d: round_dec(e.delta, 2),
                        ok: e.why.is_empty(),
                        why: e.why.join("; "),
                    })
                    .collect(),
            );
            Ok((t, keep))
        })
        .collect::<Res<_>>()?;
    let mut trials = Vec::new();
    let mut pms = Vec::new();
    for (t, pm) in res {
        trials.push(t);
        if let Some(pm) = pm {
            pms.push(pm);
        }
    }
    let good: Vec<&TrialPair> = trials.iter().filter(|t| t.sol.is_some()).collect();
    let mut out = PairCheck {
        hz,
        trials: trials.clone(),
        ..Default::default()
    };
    let sol = |t: &TrialPair| t.sol.clone().unwrap();
    if good.is_empty() {
        out.why = Some(
            trials
                .iter()
                .map(|t| {
                    format!(
                        "{} s: {}",
                        fmt_fixed(t.t0, 0),
                        t.error.clone().unwrap_or_else(|| "?".into())
                    )
                })
                .collect::<Vec<_>>()
                .join("; "),
        );
        return Ok(out);
    }
    let dd: Vec<f64> = good.iter().map(|t| sol(t).delta).collect();
    let groups: Vec<Vec<&TrialPair>> = dd
        .iter()
        .map(|&d| {
            good.iter()
                .copied()
                .filter(|t| (sol(t).delta - d).abs() <= 1.0)
                .collect()
        })
        .collect();
    let gi = first_max(0..groups.len(), |&a, &b| groups[a].len() > groups[b].len()).unwrap();
    let grp = &groups[gi];
    let need = 2.min(trials.len());
    out.n_agree = Some(grp.len());
    out.n_trials = Some(trials.len());
    if grp.len() >= need && grp.len() as f64 > good.len() as f64 / 2.0 {
        let cnt = |s: &str| grp.iter().filter(|t| sol(t).side == s).count();
        let side = if cnt("hi") > cnt("lo") { "hi" } else { "lo" };
        let ft: Vec<f64> = grp
            .iter()
            .map(|t| {
                if side == "lo" {
                    sol(t).f_lo
                } else {
                    sol(t).f_hi
                }
            })
            .collect();
        let medf = |g: &dyn Fn(&super::TrialSol) -> f64| {
            median(&grp.iter().map(|t| g(&sol(t))).collect::<Vec<_>>())
        };
        let delta = medf(&|s| s.delta);
        let f_true = median(&ft);
        out.found = true;
        out.sol = Some(super::PairFound {
            delta,
            f_true,
            f_true_spread_ppm: if ft.len() > 1 {
                ptp(&ft).unwrap() / median(&ft) * 1e6
            } else {
                0.0
            },
            f_lo: medf(&|s| s.f_lo),
            f_hi: medf(&|s| s.f_hi),
            side: side.into(),
            prom: medf(&|s| s.prom),
            xcheck_ppm: {
                let xs: Vec<f64> = grp
                    .iter()
                    .flat_map(|t| sol(t).xcheck.into_iter().map(|x| x.1))
                    .collect();
                (!xs.is_empty()).then(|| median(&xs))
            },
        });
        out.pm = Some(
            pms.into_iter()
                .find(|pm| {
                    pm.solution
                        .as_ref()
                        .is_some_and(|s| (s.delta - delta).abs() <= 1.0 && s.side.as_str() == side)
                })
                .ok_or_else(|| {
                    SurveyError::Value("StopIteration (pair_check: no matching trial)".into())
                })?,
        );
    } else {
        out.why = Some(format!(
            "partner chosen in {}/{} trials with the same d (d = {} Hz)",
            grp.len(),
            trials.len(),
            dd.iter()
                .map(|d| fmt_fixed(*d, 1))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_like_numpy() {
        assert_eq!(
            array_split(&[1, 2, 3, 4, 5, 6, 7], 3),
            vec![vec![1, 2, 3], vec![4, 5], vec![6, 7]]
        );
        assert_eq!(array_split(&[1, 2], 2), vec![vec![1], vec![2]]);
        assert_eq!(first_max([1, 3, 3, 2], |a, b| a > b), Some(3));
    }
}
