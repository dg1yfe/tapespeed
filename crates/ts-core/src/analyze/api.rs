//! Public pass-1 entry: bw "auto" candidates, stage 6 (grid, gap fill, selection) and the
//! curve meta in its fixed key order.
//!
//! [`analyze_tracks`] is stage 6 on given per-bw [`track`] outputs (the stage tests inject
//! stored reference tracks); [`analyze`] runs stages 1-5 (`stage1`, `coarse`, `track`, the bw
//! candidates in parallel) and then [`analyze_tracks`].

use indexmap::IndexMap;
use rayon::prelude::*;
use ts_dsp::ValueError;
use ts_io::{fmt_fixed, fmt_general, np_round, round_dec};

use super::coarse::{CoarseParams, coarse, fcd};
use super::gaps::{DT, Grid, TrackGridIn, to_grid};
use super::select::{SelIn, Selection, select};
use super::stage1::{Stage1, stage1};
use super::track::{TrackOut, track};
use crate::curve::{Curve, MetaValue};
use crate::hetdec::Channel;

pub use super::track::BW_AUTO;

/// `bw` argument of `analyze`: `"auto"` or a PLL natural frequency [Hz].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bw {
    Auto,
    Fixed(f64),
}

impl Bw {
    /// `BW_AUTO if bw == "auto" else (float(bw),)`.
    pub fn list(&self) -> Vec<f64> {
        match self {
            Bw::Auto => BW_AUTO.to_vec(),
            Bw::Fixed(b) => vec![*b],
        }
    }
}

impl std::str::FromStr for Bw {
    type Err = ValueError;
    fn from_str(s: &str) -> Result<Self, ValueError> {
        if s == "auto" {
            return Ok(Bw::Auto);
        }
        s.trim()
            .parse::<f64>()
            .map(Bw::Fixed)
            .map_err(|_| ValueError(format!("could not convert string to float: {s:?}")))
    }
}

/// Keyword arguments of [`analyze`] (defaults via `Default`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyzeOpts {
    /// max |speed offset| (fraction)
    pub search: f64,
    pub bw: Bw,
    pub channel: Channel,
    /// max flutter frequency in the curve [Hz]
    pub fmax: f64,
}

impl Default for AnalyzeOpts {
    fn default() -> Self {
        Self {
            search: 0.05,
            bw: Bw::Auto,
            channel: Channel::Mid,
            fmax: 25.0,
        }
    }
}

/// Channel name as written to the curve meta.
pub fn channel_str(c: Channel) -> &'static str {
    match c {
        Channel::Mid => "mid",
        Channel::S => "S",
        Channel::L => "L",
        Channel::R => "R",
    }
}

/// Output grid `np.arange(int(np.ceil(n / sr / DT)) + 1) * DT`.
pub fn grid_t(n: usize, sr: f64) -> Vec<f64> {
    let m = (n as f64 / sr / DT).ceil() as usize + 1;
    (0..m).map(|i| i as f64 * DT).collect()
}

/// One bw candidate's [`track`] output as stage 6 needs it.
#[derive(Debug, Clone, Copy)]
pub struct Stage6In<'a> {
    pub track: TrackGridIn<'a>,
    /// `d["cn0"]` (fs_p grid)
    pub cn0: &'a [f64],
    pub bw: f64,
    pub pb: f64,
}

/// Debug output of [`analyze_full`] (without the track outputs themselves).
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzeDebug {
    /// index of the chosen candidate
    pub k: usize,
    /// per candidate (bw order): [`to_grid`] outputs
    pub grids: Vec<Grid>,
    /// [`select`] valid fractions (`None` for a single bw)
    pub valid_frac: Option<Vec<f64>>,
    /// [`select`] predicted errors (`None` if not computed)
    pub pred_err: Option<Vec<f64>>,
}

/// Stage 6 of [`analyze`]: [`to_grid`] per candidate (in parallel, independent), [`select`],
/// the Curve and its meta (`ref_hz, sr, channel, bw, passband_hz, search`, then `meta`
/// merged in: existing keys keep their position, new keys are appended).
pub fn analyze_tracks(
    n: usize,
    sr: u32,
    ref_hz: f64,
    opts: &AnalyzeOpts,
    cands: &[Stage6In],
    meta: Option<&IndexMap<String, MetaValue>>,
) -> Result<(Curve, AnalyzeDebug), ValueError> {
    if cands.is_empty() {
        return Err(ValueError("analyze: no bw candidates".into()));
    }
    let t = grid_t(n, sr as f64);
    let grids: Vec<Grid> = cands
        .par_iter()
        .map(|c| to_grid(&c.track, &t, ref_hz))
        .collect::<Result<_, _>>()?;
    let keep: Vec<Kept> = cands
        .iter()
        .map(|c| Kept {
            cn0: c.cn0.to_vec(),
            valid: c.track.valid.to_vec(),
            bw: c.bw,
            pb: c.pb,
        })
        .collect();
    select_curve(t, sr, ref_hz, opts, grids, &keep, meta)
}

/// What stage 6 keeps of a candidate once its grid is built ([`select`] inputs).
struct Kept {
    cn0: Vec<f64>,
    valid: Vec<bool>,
    bw: f64,
    pb: f64,
}

/// Stage 6 after the grids: [`select`], the Curve and its meta.
fn select_curve(
    t: Vec<f64>,
    sr: u32,
    ref_hz: f64,
    opts: &AnalyzeOpts,
    grids: Vec<Grid>,
    cands: &[Kept],
    meta: Option<&IndexMap<String, MetaValue>>,
) -> Result<(Curve, AnalyzeDebug), ValueError> {
    let (k, vf, err) = if cands.len() > 1 {
        let si: Vec<SelIn> = cands
            .iter()
            .zip(&grids)
            .map(|(c, g)| SelIn {
                vg: &g.vg,
                ratio: &g.ratio,
                pb: c.pb,
                cn0: &c.cn0,
                valid: &c.valid,
            })
            .collect();
        let Selection {
            k,
            valid_frac,
            pred_err,
        } = select(&si, ref_hz, 0.03)?;
        (k, Some(valid_frac), pred_err)
    } else {
        (0, None, None)
    };
    let d = &cands[k];
    let g = &grids[k];
    let mut m: IndexMap<String, MetaValue> = IndexMap::new();
    m.insert("ref_hz".into(), MetaValue::Float(ref_hz));
    m.insert("sr".into(), MetaValue::Int(sr as i64));
    m.insert("channel".into(), channel_str(opts.channel).into());
    m.insert(
        "bw".into(),
        match opts.bw {
            Bw::Auto => MetaValue::Str(format!("auto:{}", fmt_general(d.bw, 6))),
            Bw::Fixed(_) => MetaValue::Float(d.bw),
        },
    );
    m.insert("passband_hz".into(), MetaValue::Float(round_dec(d.pb, 2)));
    m.insert("search".into(), MetaValue::Float(opts.search));
    if let Some(extra) = meta {
        for (key, v) in extra {
            m.insert(key.clone(), v.clone());
        }
    }
    let curve = Curve {
        ratio: g.ratio.clone(),
        confidence: g.cg.iter().map(|&c| np_round(c, 4)).collect(),
        valid: g.vg.clone(),
        t,
        meta: m,
    };
    Ok((
        curve,
        AnalyzeDebug {
            k,
            grids,
            valid_frac: vf,
            pred_err: err,
        },
    ))
}

impl<'a> From<&'a TrackOut> for Stage6In<'a> {
    fn from(d: &'a TrackOut) -> Self {
        Stage6In {
            track: TrackGridIn {
                tp: &d.tp,
                fc: &d.fc,
                f: &d.f,
                conf: &d.conf,
                valid: &d.valid,
                soft: &d.soft,
                fs_p: d.fs_p,
            },
            cn0: &d.cn0,
            bw: d.bw,
            pb: d.pb,
        }
    }
}

/// `analyze(..., return_debug=True)` extras: stage-1 rate and the per-bw track dicts.
#[derive(Debug, Clone)]
pub struct AnalyzeFull {
    pub curve: Curve,
    pub debug: AnalyzeDebug,
    pub fs_d: f64,
    /// per bw candidate (bw order)
    pub tracks: Vec<TrackOut>,
}

/// `analyze(x, sr, ref_hz, search, bw, channel, fmax, meta)` for interleaved f32 frames
/// (`ch` channels): stage-1 baseband, coarse Viterbi track, [`track`] per bw candidate (in
/// parallel; each candidate is computed independently, so the result does not depend on
/// the thread count), then stage 6 ([`analyze_tracks`]).
pub fn analyze(
    x: &[f32],
    ch: usize,
    sr: u32,
    ref_hz: f64,
    opts: &AnalyzeOpts,
    meta: Option<&IndexMap<String, MetaValue>>,
) -> Result<Curve, ValueError> {
    analyze_sched(x, ch, sr, ref_hz, opts, meta, &Unlimited)
}

/// Memory estimates of the bw candidates (live heap, measured in docs/MEMORY.md §2), per fs_p
/// sample: scratch of a candidate running alone, extra for each further concurrent one (their
/// peaks do not coincide: 163-249 B measured), what stays after its grid; and stage-1
/// baseband + coarse track per fs_d sample.
pub const CAND_SCRATCH_B: u64 = 420;
pub const CAND_EXTRA_B: u64 = 250;
pub const CAND_KEEP_B: u64 = 20;
pub const STAGE1_B: u64 = 24;
/// The stage-1 baseband alone per fs_d sample (c128): a later pass's `z`, streamed during the
/// decode, is resident while the earlier passes run.
pub const Z_B: u64 = 16;

/// `(N_d, [N_k])`: stage-1 and per-candidate fs_p sample counts for `n` input frames.
pub fn sizes(
    n: usize,
    sr: f64,
    ref_hz: f64,
    opts: &AnalyzeOpts,
) -> Result<(u64, Vec<u64>), ValueError> {
    let (d1, fs_d, _, _) = super::stage1::design(sr, ref_hz, opts.search, &opts.bw.list())?;
    let nd = n.div_ceil(d1);
    let nk = opts
        .bw
        .list()
        .iter()
        .map(|&b| {
            super::track::track_design(fs_d, ref_hz, b, opts.fmax).map(|r| nd.div_ceil(r.2) as u64)
        })
        .collect::<Result<_, _>>()?;
    Ok((nd as u64, nk))
}

/// Memory budget for bw candidates beyond the first (the caller holds the first one's share).
/// Called only from the analyze driver thread, never from a rayon worker.
pub trait CandBudget: Sync {
    /// Take `bytes` if available now (never blocks).
    fn try_take(&self, bytes: u64) -> bool;
    fn give(&self, bytes: u64);
}

/// No limit: all candidates run concurrently.
pub struct Unlimited;
impl CandBudget for Unlimited {
    fn try_take(&self, _: u64) -> bool {
        true
    }
    fn give(&self, _: u64) {}
}

/// [`analyze`] under a memory budget. Each candidate is gridded right after its [`track`] and
/// its track arrays are dropped (only grid, cn0 and valid stay). Candidates start largest
/// first; the first always runs (its memory is the caller's), further ones while
/// `budget.try_take(CAND_EXTRA_B · N_k)` succeeds. Each runs on its own thread (its inner
/// loops use the rayon pool), so this driver never blocks a rayon worker. The result does not
/// depend on the schedule. On a rayon worker (nested use) all candidates run via rayon.
pub fn analyze_sched(
    x: &[f32],
    ch: usize,
    sr: u32,
    ref_hz: f64,
    opts: &AnalyzeOpts,
    meta: Option<&IndexMap<String, MetaValue>>,
    budget: &dyn CandBudget,
) -> Result<Curve, ValueError> {
    analyze_pre(x, ch, sr, ref_hz, opts, meta, budget, None)
}

/// [`analyze_sched`] with an optional precomputed stage 1 (e.g. [`Stage1Stream`] fed by the
/// decoder); `pre` must be stage 1 of exactly this `x`, `ref_hz` and `opts`.
///
/// [`Stage1Stream`]: super::stage1::Stage1Stream
#[allow(clippy::too_many_arguments)]
pub fn analyze_pre(
    x: &[f32],
    ch: usize,
    sr: u32,
    ref_hz: f64,
    opts: &AnalyzeOpts,
    meta: Option<&IndexMap<String, MetaValue>>,
    budget: &dyn CandBudget,
    pre: Option<Stage1>,
) -> Result<Curve, ValueError> {
    if ch == 0 || !x.len().is_multiple_of(ch) {
        return Err(ValueError("analyze: bad channel layout".into()));
    }
    let n = x.len() / ch;
    let bws = opts.bw.list();
    let srf = sr as f64;
    let s1 = match pre {
        Some(s) => s,
        None => stage1(x, ch, srf, ref_hz, opts.search, &bws, opts.channel)?,
    };
    ts_io::prof::mark("an.stage1");
    let co = coarse(
        &s1.z,
        s1.fs_d,
        opts.search * ref_hz,
        ref_hz,
        CoarseParams::default(),
    )?;
    let fd = fcd(s1.z.len(), s1.fs_d, &co.tc, &co.f)?;
    drop(co);
    ts_io::prof::mark("an.coarse");
    let t = grid_t(n, srf);
    let one = |b: f64| -> Result<(Grid, Kept), ValueError> {
        let d = track(&s1.z, s1.fs_d, &fd, ref_hz, b, opts.fmax)?;
        let g = to_grid(&Stage6In::from(&d).track, &t, ref_hz)?;
        Ok((
            g,
            Kept {
                cn0: d.cn0,
                valid: d.valid,
                bw: d.bw,
                pb: d.pb,
            },
        ))
    };
    let mut out: Vec<Option<Result<(Grid, Kept), ValueError>>> = bws.iter().map(|_| None).collect();
    if rayon::current_thread_index().is_some() {
        let r: Vec<_> = bws.par_iter().map(|&b| one(b)).collect();
        out = r.into_iter().map(Some).collect();
    } else {
        let (_, nk) = sizes(n, srf, ref_hz, opts)?;
        let mut order: Vec<usize> = (0..bws.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(nk[k])); // stable: ties keep bw order
        std::thread::scope(|sc| {
            let (tx, rx) = std::sync::mpsc::channel();
            let (mut next, mut running, mut own_free) = (0, 0, true);
            // `--threads N` (global rayon pool size) also caps the concurrent candidates
            let cap = rayon::current_num_threads().max(1);
            loop {
                while next < order.len() && running < cap {
                    let k = order[next];
                    let need = CAND_EXTRA_B * nk[k];
                    let own = own_free;
                    if !own && !budget.try_take(need) {
                        break;
                    }
                    own_free = false;
                    next += 1;
                    running += 1;
                    let (tx, one, b) = (tx.clone(), &one, bws[k]);
                    sc.spawn(move || {
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one(b)));
                        let _ = tx.send((k, own, need, r));
                    });
                }
                if running == 0 {
                    break;
                }
                let (k, own, need, r) = rx.recv().expect("candidate thread");
                running -= 1;
                if own {
                    own_free = true;
                } else {
                    budget.give(need);
                }
                out[k] = Some(r.unwrap_or_else(|p| std::panic::resume_unwind(p)));
            }
        });
    }
    drop((s1, fd));
    ts_io::prof::mark("an.tracks (bw cand.)");
    let (grids, keep): (Vec<Grid>, Vec<Kept>) = out
        .into_iter()
        .map(|r| r.expect("every candidate ran"))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    let c = select_curve(t, sr, ref_hz, opts, grids, &keep, meta).map(|(c, _)| c);
    ts_io::prof::mark("an.select");
    c
}

/// [`analyze`] with the debug outputs.
pub fn analyze_full(
    x: &[f32],
    ch: usize,
    sr: u32,
    ref_hz: f64,
    opts: &AnalyzeOpts,
    meta: Option<&IndexMap<String, MetaValue>>,
) -> Result<AnalyzeFull, ValueError> {
    if ch == 0 || !x.len().is_multiple_of(ch) {
        return Err(ValueError("analyze: bad channel layout".into()));
    }
    let n = x.len() / ch;
    let bws = opts.bw.list();
    let srf = sr as f64;
    let s1 = stage1(x, ch, srf, ref_hz, opts.search, &bws, opts.channel)?;
    let co = coarse(
        &s1.z,
        s1.fs_d,
        opts.search * ref_hz,
        ref_hz,
        CoarseParams::default(),
    )?;
    let fd = fcd(s1.z.len(), s1.fs_d, &co.tc, &co.f)?;
    let tracks: Vec<TrackOut> = bws
        .par_iter()
        .map(|&b| track(&s1.z, s1.fs_d, &fd, ref_hz, b, opts.fmax))
        .collect::<Result<_, _>>()?;
    let s6: Vec<Stage6In> = tracks.iter().map(Stage6In::from).collect();
    let (curve, debug) = analyze_tracks(n, sr, ref_hz, opts, &s6, meta)?;
    Ok(AnalyzeFull {
        curve,
        debug,
        fs_d: s1.fs_d,
        tracks,
    })
}

/// The summary line printed after writing the curve:
/// `bw {bw}, passband {passband_hz} Hz, {n} points, valid {pct:.1f} %, mean ratio {m:.6f}`.
pub fn cli_summary(c: &Curve) -> String {
    let get = |k: &str| c.meta.get(k).map_or_else(String::new, |v| v.to_text());
    let nv = c.valid.iter().filter(|&&v| v).count();
    let vmean = nv as f64 / c.valid.len() as f64;
    let rv: Vec<f64> = c
        .ratio
        .iter()
        .zip(&c.valid)
        .filter(|(_, v)| **v)
        .map(|(r, _)| *r)
        .collect();
    let m = if rv.is_empty() {
        f64::NAN
    } else {
        ts_dsp::stats::mean(&rv)
    };
    format!(
        "bw {}, passband {} Hz, {} points, valid {} %, mean ratio {}",
        get("bw"),
        get("passband_hz"),
        c.t.len(),
        fmt_fixed(vmean * 100.0, 1),
        fmt_fixed(m, 6)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Budget with nothing free: one candidate at a time.
    struct Empty(std::sync::atomic::AtomicUsize);
    impl CandBudget for Empty {
        fn try_take(&self, _: u64) -> bool {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            false
        }
        fn give(&self, _: u64) {
            panic!("nothing was taken");
        }
    }

    #[test]
    fn scheduled_equals_full() {
        // 30 s stereo, 9 kHz line with 0.5 Hz / 0.2 % wow, noise
        let (sr, n) = (16000u32, 16000 * 30);
        let mut ph = 0.0f64;
        let x: Vec<f32> = (0..n)
            .flat_map(|i| {
                let t = i as f64 / sr as f64;
                ph += std::f64::consts::TAU
                    * 9000.0
                    * (1.0 + 0.002 * (std::f64::consts::TAU * 0.5 * t).sin())
                    / sr as f64;
                let v = (0.1 * ph.sin() + 0.01 * ((i as u64 * 7919 % 1000) as f64 / 500.0 - 1.0))
                    as f32;
                [v, v]
            })
            .collect();
        let o = AnalyzeOpts::default();
        let full = analyze_full(&x, 2, sr, 9000.0, &o, None).unwrap().curve;
        let e = Empty(Default::default());
        let one = analyze_sched(&x, 2, sr, 9000.0, &o, None, &e).unwrap();
        assert!(
            e.0.load(std::sync::atomic::Ordering::Relaxed) >= 4,
            "candidates beyond the first asked the budget"
        );
        let all = analyze(&x, 2, sr, 9000.0, &o, None).unwrap();
        for c in [&one, &all] {
            assert_eq!(c.t, full.t);
            assert!(
                c.ratio
                    .iter()
                    .zip(&full.ratio)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(
                (&c.valid, &c.confidence, &c.meta),
                (&full.valid, &full.confidence, &full.meta)
            );
        }
        // stage 1 streamed in odd chunks (as fed by the decoder)
        let mut st = super::super::stage1::Stage1Stream::new(
            2,
            sr as f64,
            9000.0,
            o.search,
            &o.bw.list(),
            o.channel,
        )
        .unwrap();
        for c in x.chunks(2 * 7777) {
            st.push(c);
        }
        let pre = analyze_pre(&x, 2, sr, 9000.0, &o, None, &Unlimited, Some(st.finish())).unwrap();
        assert!(
            pre.ratio
                .iter()
                .zip(&full.ratio)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!((&pre.valid, &pre.meta), (&full.valid, &full.meta));
        // on a rayon worker the plain parallel path runs
        let nested = rayon::scope(|_| analyze_sched(&x, 2, sr, 9000.0, &o, None, &e)).unwrap();
        assert_eq!(nested.ratio, full.ratio);
        // estimates: N_d and N_k as the run uses them
        let (nd, nk) = sizes(n as usize, sr as f64, 9000.0, &o).unwrap();
        assert_eq!(nk.len(), 5);
        assert!(nk.iter().all(|&k| k <= nd && k > 0));
    }

    #[test]
    fn bw_parse_and_list() {
        assert_eq!("auto".parse::<Bw>().unwrap(), Bw::Auto);
        assert_eq!(" 12 ".parse::<Bw>().unwrap(), Bw::Fixed(12.0));
        assert!("x".parse::<Bw>().is_err());
        assert_eq!(Bw::Auto.list(), BW_AUTO.to_vec());
        assert_eq!(Bw::Fixed(25.0).list(), vec![25.0]);
    }

    #[test]
    fn grid_len() {
        // test_no_reference_all_invalid: 8 s at 48 kHz -> ceil(8 / 0.005) + 1 points
        assert_eq!(grid_t(8 * 48000, 48000.0).len(), 1601);
    }

    #[test]
    fn summary_line() {
        let mut meta = IndexMap::new();
        meta.insert("bw".to_string(), MetaValue::Str("auto:12".into()));
        meta.insert("passband_hz".to_string(), MetaValue::Float(6.6));
        let c = Curve {
            t: vec![0.0, 0.005, 0.01, 0.015],
            ratio: vec![1.0, 1.002, 1.004, 2.0],
            confidence: vec![1.0; 4],
            valid: vec![true, true, true, false],
            meta,
        };
        assert_eq!(
            cli_summary(&c),
            "bw auto:12, passband 6.6 Hz, 4 points, valid 75.0 %, mean ratio 1.002000"
        );
        let c2 = Curve {
            valid: vec![false; 4],
            ..c
        };
        assert!(cli_summary(&c2).ends_with("valid 0.0 %, mean ratio nan"));
    }
}
