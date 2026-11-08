//! survey (`--survey`): reference-line candidates from sampled
//! blocks, trial tracks / coherence / pair check, recommendation, run plan, report text and
//! JSON. No audio output.
//!
//! - [`detect`]: log-grid line detection per block and channel, frame tracking, grouping,
//!   classes, spectrum cliff.
//! - [`cands`]: candidate summary, TV region, tape correlation, verdicts, trial tracks
//!   (`coherence_check`, `pair_check`).
//! - [`plan`]: `recommend`, `plan` (shared `params::resolve` / `absolute::apply_l1`).
//! - [`report`]: `report_text`, `to_json` (cleaned JSON), plot data.
//!
//! Trial tracks call `analyze()`; the survey takes it as a [`Tracker`] (the pipeline passes
//! the real analyze, tests replay stored reference curves). Blocks are decoded and analysed
//! in parallel, trial tracks run in parallel; every result is merged in a fixed order, so the
//! output does not depend on the thread count.

pub mod cands;
pub mod detect;
pub mod plan;
pub mod report;

use rayon::prelude::*;
use ts_dsp::stats::median;
use ts_io::{Audio, fmt_fixed, fmt_general, round_dec};

use crate::curve::Curve;
use crate::hetdec::Channel;
use cands::*;
pub use cands::{BlockEntry, Cand, ChanStat};
use detect::*;

/// Survey error: decoding (ffmpeg) or an invalid value.
#[derive(Debug, thiserror::Error)]
pub enum SurveyError {
    #[error("{0}")]
    Io(ts_io::Error),
    #[error("{0}")]
    Value(String),
}

impl From<ts_dsp::ValueError> for SurveyError {
    fn from(e: ts_dsp::ValueError) -> Self {
        SurveyError::Value(e.to_string())
    }
}

pub type Res<T> = Result<T, SurveyError>;

/// `analyze(x, sr, f, 0.01, "auto", channel)` on a trial excerpt (`x` interleaved, `ch`
/// channels) that starts at `t0` s and was requested with length `dur` s.
pub trait Tracker: Sync {
    #[allow(clippy::too_many_arguments)]
    fn track(
        &self,
        x: &[f32],
        ch: usize,
        sr: u32,
        f: f64,
        channel: Channel,
        t0: f64,
        dur: f64,
    ) -> Result<Curve, String>;
}

/// The real trial track: `analyze(x, sr, f, 0.01, "auto", channel)`.
pub struct AnalyzeTracker;

impl Tracker for AnalyzeTracker {
    fn track(
        &self,
        x: &[f32],
        ch: usize,
        sr: u32,
        f: f64,
        channel: Channel,
        _t0: f64,
        _dur: f64,
    ) -> Result<Curve, String> {
        let o = crate::analyze::AnalyzeOpts {
            search: 0.01,
            bw: crate::analyze::Bw::Auto,
            channel,
            ..Default::default()
        };
        crate::analyze::analyze(x, ch, sr, f, &o, None).map_err(|e| e.to_string())
    }
}

/// One trial track: excerpt start / length, its audio and the analyze curve.
pub struct Trial {
    pub t0: f64,
    pub dur: f64,
    pub curve: Curve,
    pub audio: Audio,
}

/// Coherence entry (`coherence[i]`, also `candidate["coh"]`).
#[derive(Debug, Clone, PartialEq)]
pub struct Coh {
    pub a: f64,
    pub b: f64,
    pub t0: f64,
    pub dur: f64,
    pub common: f64,
    pub n_seg: usize,
    /// band -> coherence (rounded to 4 decimals)
    pub coh: Vec<(String, f64)>,
    pub tape: bool,
}

impl Coh {
    pub fn get(&self, k: &str) -> Option<f64> {
        self.coh.iter().find(|(kk, _)| kk == k).map(|(_, v)| *v)
    }
}

/// Solution of one pair trial.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrialSol {
    pub s: f64,
    pub f_true: f64,
    pub f_lo: f64,
    pub f_hi: f64,
    pub delta: f64,
    pub side: String,
    pub prom: f64,
    pub pres: f64,
    pub k: f64,
    /// (nominal in general format, ppm)
    pub xcheck: Vec<(String, f64)>,
}

/// Candidate summary of one pair trial (`cands[i]`).
#[derive(Debug, Clone, PartialEq)]
pub struct PairCand {
    pub f: f64,
    pub prom: f64,
    pub d: f64,
    pub ok: bool,
    pub why: String,
}

/// One pair trial (`pair["trials"][i]`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrialPair {
    pub t0: f64,
    pub valid: f64,
    pub ambiguity: Option<String>,
    pub n_cand: usize,
    pub n_ok: usize,
    pub error: Option<String>,
    pub sol: Option<TrialSol>,
    pub cands: Option<Vec<PairCand>>,
}

/// Agreed pair result.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PairFound {
    pub delta: f64,
    pub f_true: f64,
    pub f_true_spread_ppm: f64,
    pub f_lo: f64,
    pub f_hi: f64,
    pub side: String,
    pub prom: f64,
    pub xcheck_ppm: Option<f64>,
}

/// `pair_check` result (`S["pair"]`).
#[derive(Debug, Clone, Default)]
pub struct PairCheck {
    pub hz: f64,
    pub trials: Vec<TrialPair>,
    pub found: bool,
    pub n_agree: Option<usize>,
    pub n_trials: Option<usize>,
    pub sol: Option<PairFound>,
    pub why: Option<String>,
    /// The trial's `pair_measure` result matching the agreed d / side
    pub pm: Option<crate::absolute::pair::PairResult>,
}

/// File info (`audio.info` + `lossy`).
#[derive(Debug, Clone, PartialEq)]
pub struct FileInfo {
    pub codec: String,
    pub format: String,
    pub sr: u32,
    pub channels: usize,
    pub duration: f64,
    pub bit_rate: i64,
    pub lossy: bool,
}

/// `S["bandwidth"]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bandwidth {
    pub nyquist: f64,
    pub cliff_hz: Option<f64>,
    /// band -> level [dB] rounded to 0.1 (NaN = none)
    pub levels_db: Vec<(String, f64)>,
}

/// `S["_spec"]` (mid/mono grid): plot data.
#[derive(Debug, Clone, Default)]
pub struct SpecData {
    pub f: Vec<f64>,
    pub d: Vec<f64>,
    pub prom: Vec<Vec<f64>>,
    pub d_blocks: Vec<Vec<f64>>,
}

/// The survey dict `S`.
#[derive(Debug, Clone)]
pub struct Survey {
    pub file: String,
    pub info: FileInfo,
    /// unrounded (t0, len)
    pub blocks: Vec<(f64, f64)>,
    pub channels: Vec<String>,
    pub search: f64,
    pub play: f64,
    pub coherence: Vec<Coh>,
    pub candidates: Vec<Cand>,
    /// candidate id
    pub primary: Option<usize>,
    pub pair: Option<PairCheck>,
    pub bandwidth: Bandwidth,
    pub spec: SpecData,
    pub runtime_s: f64,
}

impl Survey {
    pub fn cand(&self, id: usize) -> Option<&Cand> {
        self.candidates.iter().find(|c| c.id == id)
    }

    /// Blocks as written to JSON / text (rounded to 0.01 s).
    pub fn blocks_rounded(&self) -> Vec<(f64, f64)> {
        self.blocks
            .iter()
            .map(|&(a, b)| (round_dec(a, 2), round_dec(b, 2)))
            .collect()
    }
}

/// Survey parameters (`survey(path, n, L, search, play, coh)`).
#[derive(Debug, Clone, PartialEq)]
pub struct SurveyParams {
    pub n: usize,
    pub l: f64,
    pub search: f64,
    pub play: f64,
    pub coh: bool,
}

impl Default for SurveyParams {
    fn default() -> Self {
        Self {
            n: 12,
            l: 20.0,
            search: 0.03,
            play: 1.0,
            coh: true,
        }
    }
}

/// The smallest `--search` (0.01 steps) that covers line `c` around `nominal x play`: its
/// offset + 0.5 % (wow / drift headroom) and its p5-p95 spread + 0.25 %.
pub fn search_needed(c: &Cand, nominal: f64, play: f64) -> f64 {
    let n = nominal * play;
    let dev = [c.spread_hz[0], c.spread_hz[1]]
        .iter()
        .map(|f| (f / n - 1.0).abs())
        .fold(0.0, f64::max);
    let need = ((c.f / n - 1.0).abs() + 0.005).max(dev + 0.0025);
    (need * 100.0 - 1e-6).ceil() / 100.0
}

const LEVEL_BANDS: [(f64, f64); 7] = [
    (3000.0, 4000.0),
    (5000.0, 6000.0),
    (7000.0, 8000.0),
    (8250.0, 9250.0),
    (10000.0, 12000.0),
    (14000.0, 15000.0),
    (16000.0, 18000.0),
];

/// `collect(path, inf, n, L)`: decode + detect every block (blocks in parallel).
type Collected = (Vec<(f64, f64)>, Vec<Det>, Vec<Vec<BlockSpec>>);

fn collect(path: &str, dur: f64, n: usize, l: f64) -> Res<Collected> {
    let bl = blocks(dur, n, l);
    let per: Vec<(Vec<Det>, Vec<BlockSpec>)> = bl
        .par_iter()
        .enumerate()
        .map(|(i, &(t0, d))| -> Res<_> {
            let a = ts_io::load(path, Some(t0), Some(d)).map_err(SurveyError::Io)?;
            let (mut dets, sp) = detect_block(&a.data, a.channels, a.sr, LOW_MAX)?;
            for e in &mut dets {
                e.blk = i;
            }
            Ok((dets, sp))
        })
        .collect::<Res<_>>()?;
    let mut dets = Vec::new();
    let mut specs = Vec::new();
    for (d, s) in per {
        dets.extend(d);
        specs.push(s);
    }
    Ok((bl, dets, specs))
}

/// `survey(path, n, L, search, play, coh)`.
pub fn survey(path: &str, p: &SurveyParams, tracker: &dyn Tracker) -> Res<Survey> {
    let t_start = std::time::Instant::now();
    let inf = ts_io::info(path).map_err(SurveyError::Io)?;
    let info = FileInfo {
        lossy: LOSSY.contains(&inf.codec.as_str()),
        codec: inf.codec,
        format: inf.format,
        sr: inf.sr,
        channels: inf.channels,
        duration: inf.duration,
        bit_rate: inf.bit_rate,
    };
    let (search, play) = (p.search, p.play);
    let (bl, dets, specs) = collect(path, info.duration, p.n, p.l)?;
    let chans: Vec<String> = if info.channels == 1 {
        vec!["mono".into()]
    } else {
        ["L", "R", "mid", "S"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    let fs: Vec<f64> = dets.iter().map(|d| d.f).collect();
    let mut keep = Vec::new();
    for g in group(&fs, GROUP_TOL) {
        let c = summarise(
            &dets,
            &g,
            bl.len(),
            &chans,
            info.lossy,
            info.sr as f64,
            search,
            play,
        )?;
        let ok = if c.low {
            c.cls == "hum" && c.presence >= 0.5 && c.pers >= 0.8
        } else {
            let nmin = if c.cls != "unknown" {
                2
            } else {
                2usize.max((bl.len() as f64 / 4.0).ceil() as usize)
            };
            c.pers >= 0.5 && (c.n >= nmin || bl.len() <= 2 || (c.prom >= 10.0 && c.pers >= 0.8))
        };
        if ok {
            keep.push(c);
        }
    }
    tv_region(&mut keep, search, play);
    let chi = chans
        .iter()
        .position(|c| c == "mono" || c == "mid")
        .unwrap();
    let fk = specs[0][chi].f.clone();
    let nk = fk.len();
    let dm: Vec<f64> = (0..nk)
        .map(|k| median(&specs.iter().map(|sp| sp[chi].d[k]).collect::<Vec<_>>()))
        .collect();
    let fc = cliff(&fk, &dm)?;
    for c in keep.iter_mut() {
        if info.lossy && fc.is_some_and(|fc| c.f >= 0.98 * fc) {
            c.codec_risk = true;
            c.cls = "codec".into();
            c.nominal = None;
            c.ratio = None;
            c.offset_pct = None;
            c.label = format!(
                "likely codec artifact (lossy, above the {} kHz cliff)",
                fmt_fixed(fc.unwrap() / 1000.0, 1)
            );
        } else if c.cls == "unknown" && !c.low {
            let mut best = (f64::INFINITY, f64::INFINITY);
            for (_, n, _) in NOMINAL {
                let d = (c.f / (n * play) - 1.0).abs();
                if d < best.0 || (d == best.0 && n < best.1) {
                    best = (d, n);
                }
            }
            if best.0 <= 2.0 * search {
                let n = best.1;
                let off = report::sign_f((c.f / (n * play) - 1.0) * 100.0, 2);
                let am = NOMINAL.iter().find(|(k, v, _)| *k == "AM beat" && *v == n);
                if let Some((cls, _, lab)) = am {
                    // an AM beat just outside the window (tape > 3 % off speed)
                    let r = c.f / (n * play);
                    c.cls = cls.to_string();
                    c.nominal = Some(n);
                    c.ratio = Some(r);
                    c.offset_pct = Some((r - 1.0) * 100.0);
                    c.label = format!(
                        "{} at {off} %: outside --search {} %, needs --search {})",
                        lab.trim_end_matches(')'),
                        fmt_general(search * 100.0, 6),
                        fmt_general(search_needed(c, n, play), 6)
                    );
                } else {
                    c.label += &format!(
                        " ({off} % from {} Hz: outside --search {} %?)",
                        fmt_general(n, 6),
                        fmt_general(search * 100.0, 6)
                    );
                }
            }
        }
    }
    tv_region(&mut keep, search, play);
    let pi = tape_corr(&mut keep, &dets);
    let pw = pi.map(|i| keep[i].wow_pct);
    let keep: Vec<Cand> = keep
        .into_iter()
        .filter(|c| !(c.cls == "unknown" && pw.is_some_and(|w| c.wow_pct > OWN_FM * w)))
        .collect();
    let mut keep = keep;
    // a pilot on the tape shares the tape's wow with the primary; a "pilot" whose frame
    // wow is uncorrelated with it (wcorr < 0.2) is some other line near 19 kHz
    let pc = keep.iter().find(|c| c.primary).map(|c| c.cls.clone());
    for c in keep.iter_mut() {
        if let (Some(pcls), Some(Some(wc))) = (&pc, c.wow_corr)
            && c.cls == "FM pilot"
            && !c.primary
            && wc < FREE_CORR
        {
            c.label = format!(
                "unknown line near the FM pilot: no FM signature (frame wow uncorrelated with the {pcls} primary: wcorr {} < {})",
                fmt_fixed(wc, 2),
                fmt_general(FREE_CORR, 6)
            );
            c.cls = "unknown".into();
            c.nominal = None;
            c.ratio = None;
            c.offset_pct = None;
        }
    }
    for c in keep.iter_mut() {
        c.verdict = verdict(c);
    }
    let rank = |a: &Cand, b: &Cand| -> std::cmp::Ordering {
        let ka = (usable(a), a.cls != "codec", !a.low);
        let kb = (usable(b), b.cls != "codec", !b.low);
        kb.cmp(&ka)
            .then(
                score(b)
                    .partial_cmp(&score(a))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                b.cn0
                    .partial_cmp(&a.cn0)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    };
    keep.sort_by(rank);
    let top: Vec<usize> = (0..keep.len()).filter(|&i| usable(&keep[i])).collect();
    let ctx = TrialCtx {
        path,
        duration: info.duration,
        blocks: &bl,
        tracker,
    };
    let mut coherence = Vec::new();
    if p.coh && !top.is_empty() {
        let mut oth: Vec<usize> = top.iter().skip(1).take(2).copied().collect();
        oth.extend(
            top.iter()
                .skip(3)
                .copied()
                .filter(|&i| keep[i].cls == "TV line")
                .take(1),
        );
        coherence = coherence_check(&ctx, &mut keep, top[0], &oth)?;
    }
    for c in keep.iter_mut() {
        if let Some(tv) = c.trial_valid
            && tv < TRIAL_MIN
        {
            c.verdict = format!("no lock (trial track {} % valid)", fmt_fixed(100.0 * tv, 0));
        }
    }
    keep.sort_by(rank);
    for (i, c) in keep.iter_mut().enumerate() {
        c.id = i + 1;
    }
    let top: Vec<usize> = (0..keep.len()).filter(|&i| usable(&keep[i])).collect();
    let primary = top.first().map(|&i| keep[i].id);
    let mut pair = None;
    if let Some(&t0) = top.first()
        && p.coh
        && let Some(hz) = pair_hz(&keep[t0], search, play)
    {
        let tv: Vec<f64> = top
            .iter()
            .filter(|&&i| keep[i].cls == "TV line" && i != t0)
            .filter_map(|&i| keep[i].nominal)
            .take(1)
            .collect();
        pair = Some(pair_check(&ctx, &keep[t0], hz, &tv, play)?);
    }
    let sr = info.sr as f64;
    let bandwidth = Bandwidth {
        nyquist: sr / 2.0,
        cliff_hz: fc,
        levels_db: LEVEL_BANDS
            .iter()
            .filter(|(_, hi)| *hi < sr / 2.0)
            .map(|&(lo, hi)| {
                (
                    format!("{}-{}", fmt_general(lo, 6), fmt_general(hi, 6)),
                    round_dec(level_db(&fk, &dm, lo, hi), 1),
                )
            })
            .collect(),
    };
    let spec = SpecData {
        prom: specs.iter().map(|sp| sp[chi].prom.clone()).collect(),
        d_blocks: specs.iter().map(|sp| sp[chi].d.clone()).collect(),
        f: fk,
        d: dm,
    };
    Ok(Survey {
        file: path.to_string(),
        info,
        blocks: bl,
        channels: chans,
        search,
        play,
        coherence,
        candidates: keep,
        primary,
        pair,
        bandwidth,
        spec,
        runtime_s: t_start.elapsed().as_secs_f64(),
    })
}

/// `--survey` result: survey, recommendation, plan, report text and JSON.
pub struct SurveyRun {
    pub survey: Survey,
    pub recommend: plan::Recommend,
    pub plan: plan::Plan,
    /// `report_text(S)`
    pub text: String,
    /// `json.dumps(_clean(S), indent=1)`
    pub json: String,
    /// `<stem>_survey.json`, `<stem>_survey.png`
    pub json_path: String,
    pub png_path: String,
}

impl SurveyRun {
    /// The last stdout line of `survey.main`.
    pub fn final_line(&self) -> String {
        let rt = self.survey.runtime_s;
        format!(
            "\nsurvey {}, {}; {} s ({} min per hour of audio)",
            self.json_path,
            self.png_path,
            fmt_fixed(rt, 1),
            fmt_fixed(rt / self.survey.info.duration * 3600.0 / 60.0, 2)
        )
    }
}

/// `<stem>_survey.json`, `<stem>_survey.png` next to `-o` (else the input stem in the cwd).
pub fn survey_paths(input: &str, output: Option<&str>) -> Res<(String, String)> {
    let base = match output {
        Some(o) if !o.is_empty() => crate::params::norm_path(o),
        _ => crate::params::path_stem(input),
    };
    let stem = crate::params::path_stem(&base);
    let jp = crate::params::with_file_name(&base, &format!("{stem}_survey.json"))
        .map_err(|e| SurveyError::Value(e.to_string()))?;
    let pp = crate::params::with_file_name(&base, &format!("{stem}_survey.png"))
        .map_err(|e| SurveyError::Value(e.to_string()))?;
    Ok((jp, pp))
}

/// The `--survey` run without the file writes / plot: survey, recommend, plan, text, JSON.
/// `fix_runtime`: replace the measured runtime (e.g. 0.0 for comparisons against stored reports).
pub fn run(
    a: &plan::SurveyArgs,
    tracker: &dyn Tracker,
    fix_runtime: Option<f64>,
) -> Res<SurveyRun> {
    let (jp, pp) = survey_paths(&a.input, a.output.as_deref())?;
    let play = crate::params::play_factor(&a.run_args())
        .map_err(|e| SurveyError::Value(e.to_string()))?
        .2;
    let sp = SurveyParams {
        n: a.survey_blocks,
        l: a.survey_block_s,
        search: a.search,
        play,
        coh: true,
    };
    let mut s = survey(&a.input, &sp, tracker)?;
    if let Some(r) = fix_runtime {
        s.runtime_s = r;
    }
    let rec = plan::recommend(&s);
    let pl = if a.r#ref.is_none() {
        // without --ref, PLAN plans the cmd: line (typed non-default options win)
        let mut a2 = plan::with_recommended(a, &rec);
        let mut pl = plan::plan(&mut a2, &s, &rec)?;
        pl.basis = Some("recommended options = the cmd: line, typed options win".into());
        pl
    } else {
        let mut a2 = a.clone();
        let mut pl = plan::plan(&mut a2, &s, &rec)?;
        pl.basis = Some("typed options; the cmd: line is the recommendation".into());

        pl
    };
    let text = report::report_text(&s, &rec, &pl, 12);
    let json = report::to_json(&s, &rec, &pl).dumps(Some(1));
    Ok(SurveyRun {
        survey: s,
        recommend: rec,
        plan: pl,
        text,
        json,
        json_path: jp,
        png_path: pp,
    })
}
