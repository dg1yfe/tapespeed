//! One-shot pipeline: analyze (pass 1) → [line pass] →
//! [pair / switch repair] → model step → long gaps → evidence / bandwidth policy → L1 → curve,
//! plots → correct (streaming) → notches → lowpass → encoder.
//!
//! Stdout lines, curve meta and the report JSON have a fixed text, key order and value
//! types. Timings are real unless `TAPESPEED_FIXED_CLOCK` is set (then 0.0), so stdout and
//! reports can be compared byte for byte with stored reference results.
//!
//! Memory: the output is never held. `--auto-notch` runs the corrector up to three times
//! (pass 1: detection on the mono mean, pass 2 only if lines were found: keying statistics,
//! final pass: gated notches into the encoder); each pass keeps O(chunk) state.
use std::path::Path;
use std::time::Instant;

use indexmap::IndexMap;
use ts_core::absolute::{self, Abs, PairParams, PairResult, ghost_rho, pair_measure, pair_table};
use ts_core::analyze::api::{CandBudget, Unlimited, analyze_pre};
use ts_core::analyze::stage1::{Stage1, Stage1Stream};
use ts_core::analyze::{AnalyzeOpts, Bw as ABw};
use ts_core::correct::{self, Lowpass, Params, Post, RunError};
use ts_core::curve::{Curve, MetaValue, write_curve};
use ts_core::evidence::{
    BandStats, Channel, CorrLimit, ExtraCurve, FadingParams, apply_bandwidth, band_stats_default,
    bandwidth_policy, fading_metric,
};
use ts_core::interf::{self, DetectParams, Detector, Gate, KeyingStream, is_keyed};
use ts_core::model::gaps::{fill_long_gaps_default, long_gaps_default};
use ts_core::model::step::{ModelMode, ModelReport, StepFit, StepOpts, model_step, report_text};
use ts_core::params::{self, Bw, Resolved, RunArgs};
use ts_core::physics::fmt_speed;
use ts_dsp::stats::median;
use ts_io::json::Value;
use ts_io::{fmt_fixed, fmt_general, round_dec};

use std::io::Write;

use crate::cli::{self, Cli};
use crate::plan::{self, Budget, Mode, Plan};

/// Failure of a run: a plain exit message (`msg`) or an error with a class name (`Class: msg`).
/// Both exit with status 1; the message goes to stderr.
#[derive(Debug, Clone)]
pub enum Fail {
    Exit(String),
    Error(String, String),
    /// Usage error found after parsing: `tapespeed: error: msg`, exit status 2.
    Usage(String),
}

impl Fail {
    fn value(m: impl std::fmt::Display) -> Fail {
        Fail::Error("ValueError".into(), m.to_string())
    }
    fn runtime(m: impl std::fmt::Display) -> Fail {
        Fail::Error("RuntimeError".into(), m.to_string())
    }
    /// stderr text (`Exit`: the message; else `Class: message`).
    pub fn message(&self) -> String {
        match self {
            Fail::Exit(m) => m.clone(),
            Fail::Error(c, m) => format!("{c}: {m}"),
            Fail::Usage(m) => format!("tapespeed: error: {m}"),
        }
    }
    /// Process exit status (2 for usage errors, else 1).
    pub fn code(&self) -> i32 {
        match self {
            Fail::Usage(_) => 2,
            _ => 1,
        }
    }
}

impl From<params::ParamError> for Fail {
    fn from(e: params::ParamError) -> Self {
        match e {
            params::ParamError::Exit(m) => Fail::Exit(m),
            params::ParamError::Value(m) => Fail::value(m),
            params::ParamError::ZeroDivision(m) => Fail::Error("ZeroDivisionError".into(), m),
        }
    }
}

impl From<ts_dsp::ValueError> for Fail {
    fn from(e: ts_dsp::ValueError) -> Self {
        Fail::value(e.0)
    }
}

impl From<absolute::AbsError> for Fail {
    fn from(e: absolute::AbsError) -> Self {
        match e {
            absolute::AbsError::Runtime(m) => Fail::runtime(m),
            e => Fail::value(e),
        }
    }
}

/// The clock of the report and the timing line (`TAPESPEED_FIXED_CLOCK` → always 0.0).
pub struct Clock {
    t0: Instant,
    fixed: bool,
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock {
    pub fn new() -> Self {
        Clock {
            t0: Instant::now(),
            fixed: std::env::var_os("TAPESPEED_FIXED_CLOCK").is_some(),
        }
    }
    fn time(&self) -> f64 {
        if self.fixed {
            0.0
        } else {
            self.t0.elapsed().as_secs_f64()
        }
    }
}

/// `Cli` → the run parameters (`--ref`, `-o` present).
pub fn run_args(a: &Cli) -> RunArgs {
    RunArgs {
        r#ref: a.r#ref.unwrap_or(0.0),
        output: a.output.clone().unwrap_or_default(),
        abs: a.abs.clone(),
        tape_speed: a.tape_speed.clone(),
        play_speed: a.play_speed.clone(),
        bw: a.bw.clone(),
        check_line: a.check_line,
        weak_rf: a.weak_rf,
        corr_limit: a.corr_limit.clone(),
        curve: a.curve.clone(),
        report: a.report.clone(),
        plot: a.plot.clone(),
        no_plot: a.no_plot,
        lowpass: a.lowpass,
        lowpass_width: a.lowpass_width,
        keep_ref: a.keep_ref,
        notch_at: a.notch_at.clone(),
        notch_width: a.notch_width,
    }
}

/// `MetaValue` as a JSON value.
pub fn meta_json(v: &MetaValue) -> Value {
    match v {
        MetaValue::Str(s) => Value::Str(s.clone()),
        MetaValue::Int(i) => Value::Int(*i),
        MetaValue::Float(x) => Value::Float(*x),
        MetaValue::Bool(b) => Value::Bool(*b),
    }
}

fn opt_f(v: Option<f64>) -> Value {
    v.map(Value::Float).unwrap_or(Value::Null)
}

/// An optional float that counts as set only when non-zero.
fn truthy(v: Option<f64>) -> Option<f64> {
    v.filter(|&x| x != 0.0)
}

fn scaled(mut c: Curve, f: f64) -> Curve {
    for r in c.ratio.iter_mut() {
        *r *= f;
    }
    c
}

fn frac_true(v: &[bool]) -> f64 {
    v.iter().filter(|&&b| b).count() as f64 / v.len() as f64
}

/// Outputs of [`run`] besides the final curve.
pub struct RunOut {
    pub curve: Curve,
    pub rep: Value,
    pub c1: Curve,
    pub cm: Curve,
    pub fit: Option<StepFit>,
    pub mrep: ModelReport,
}

pub(crate) fn analyze_opts(a: &Cli, p: &Resolved) -> Result<AnalyzeOpts, Fail> {
    Ok(AnalyzeOpts {
        search: a.search,
        bw: match p.bw {
            Bw::Auto => ABw::Auto,
            Bw::Hz(b) => ABw::Fixed(b),
        },
        channel: a.channel.parse::<Channel>()?,
        ..AnalyzeOpts::default()
    })
}

fn pair_params(a: &Cli, hz: f64, xc: &[f64], play: f64) -> Result<PairParams, Fail> {
    Ok(PairParams {
        hz,
        dmax: a.pair_dmax,
        channel: a.channel.parse::<Channel>()?,
        xcheck: xc.to_vec(),
        play,
        ..PairParams::default()
    })
}

/// `run(x, sr, a)`: the pipeline on interleaved audio `x` (`ch` channels). `log` gets each
/// printed line (with embedded newlines where several lines form one message).
pub fn run(
    x: &[f32],
    ch: usize,
    sr: u32,
    a: &Cli,
    clock: &Clock,
    log: &mut dyn FnMut(&str),
) -> Result<RunOut, Fail> {
    run_with(x, ch, sr, a, clock, log, &Unlimited)
}

/// [`run`] with the bw candidates of each analyze pass under `budget` (same result).
pub fn run_with(
    x: &[f32],
    ch: usize,
    sr: u32,
    a: &Cli,
    clock: &Clock,
    log: &mut dyn FnMut(&str),
    budget: &dyn CandBudget,
) -> Result<RunOut, Fail> {
    run_pre(x, ch, sr, a, clock, log, budget, Pre::default())
}

/// The analyze passes that only need the raw input, in pipeline order: pass 1 at the search
/// centre, the `line:` / check-line pass at `line · play`. Their stage 1 can run while the
/// input is still being decoded ([`load_streamed`]); everything later needs pass 1's curve.
pub(crate) fn stage1_passes(a: &Cli) -> Result<(AnalyzeOpts, Vec<f64>), Fail> {
    let p = params::resolve(&run_args(a))?;
    let o = analyze_opts(a, &p)?;
    let mut refs = vec![p.centre];
    if let Some(l) = truthy(p.line_hz) {
        refs.push(l * p.play);
    }
    Ok((o, refs))
}

/// Stage-1 basebands computed while decoding, for the input of `(frames, channels, sr)`.
#[derive(Default)]
pub struct Pre {
    shape: (usize, usize, u32),
    s1: Vec<(f64, Stage1)>,
}

impl Pre {
    /// Stage 1 at `ref_hz` for the input `x` (`ch`, `sr`), if it was streamed.
    fn take(&mut self, ref_hz: f64, x: &[f32], ch: usize, sr: u32) -> Option<Stage1> {
        if self.shape != (x.len() / ch.max(1), ch, sr) {
            return None;
        }
        let i = self
            .s1
            .iter()
            .position(|(r, _)| r.to_bits() == ref_hz.to_bits())?;
        Some(self.s1.remove(i).1)
    }
}

/// Decoded chunks in flight between the decoder and the stage-1 thread (64 Ki frames each).
const S1_QUEUE: usize = 32;
/// Stage-1 work items in flight while decoding (leaves cores to ffmpeg and the reader).
const S1_PAR: usize = 3;

/// Decode the input (RAM or spill, as `mode`) and, overlapped with the decoder, stage 1 of the
/// analyze passes that only need the raw input ([`stage1_passes`]). The decoder hands each
/// chunk to a stage-1 thread (bounded queue) that feeds one [`Stage1Stream`] per pass; the
/// basebands are bit-identical to `stage1` on the whole input. A pass whose design fails is
/// left out (the analyze pass then reports the error in its usual place).
pub fn load_streamed(
    input: &str,
    mode: Mode,
    tmp: &Path,
    passes: Option<(AnalyzeOpts, Vec<f64>)>,
) -> Result<(ts_io::Samples, Pre), ts_io::Error> {
    let Some((ao, refs)) = passes else {
        let s = match mode {
            Mode::Ram => ts_io::spill::load_ram(input),
            Mode::Spill => ts_io::spill::spill(input, tmp),
        }?;
        return Ok((s, Pre::default()));
    };
    enum Msg {
        Start(u32, usize, usize),
        Data(Vec<f32>),
    }
    std::thread::scope(|sc| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Msg>(S1_QUEUE);
        let worker = sc.spawn(move || {
            let (mut shape, mut st) = ((0, 0, 0), Vec::<(f64, Stage1Stream)>::new());
            // keeping up: S1_PAR work items at a time, the other cores stay with the decoder;
            // behind (a backlog of S1_QUEUE / 4 chunks, e.g. on a loaded machine): the whole pool
            let mut batch = Vec::with_capacity(S1_QUEUE + 1);
            while let Ok(m) = rx.recv() {
                batch.push(m);
                batch.extend(rx.try_iter().take(S1_QUEUE));
                let par = (batch.len() <= S1_QUEUE / 4).then_some(S1_PAR);
                for m in batch.drain(..) {
                    match m {
                        Msg::Start(sr, ch, hint) => {
                            shape = (0, ch, sr);
                            let bws = ao.bw.list();
                            st = refs
                                .iter()
                                .filter_map(|&r| {
                                    let s = Stage1Stream::new(
                                        ch, sr as f64, r, ao.search, &bws, ao.channel,
                                    );
                                    let mut s = s.ok()?;
                                    s.reserve_frames(hint);
                                    Some((r, s))
                                })
                                .collect();
                        }
                        Msg::Data(c) => {
                            shape.0 += c.len() / shape.1;
                            for (_, s) in st.iter_mut() {
                                s.set_par(par);
                                s.push(&c);
                            }
                        }
                    }
                }
            }
            // after the decode: the whole pool
            let s1 = st
                .into_iter()
                .map(|(r, mut s)| {
                    s.set_par(None);
                    (r, s.finish())
                })
                .collect();
            Pre { shape, s1 }
        });
        let tx2 = tx.clone();
        // a send fails only if the worker died; its panic surfaces at join
        let init = move |sr, ch, hint| {
            let _ = tx2.send(Msg::Start(sr, ch, hint));
        };
        let tap = move |c: &[f32]| {
            let _ = tx.send(Msg::Data(c.to_vec()));
        };
        let r = match mode {
            Mode::Ram => ts_io::spill::load_ram_tap(input, init, tap),
            Mode::Spill => ts_io::spill::spill_tap(input, tmp, init, tap),
        };
        ts_io::prof::mark("load (decode, stage 1 overlapped)");
        let pre = worker
            .join()
            .unwrap_or_else(|p| std::panic::resume_unwind(p));
        ts_io::prof::mark("stage-1 tail (after decode)");
        Ok((r?, pre))
    })
}

/// [`run_with`] with the stage-1 basebands `pre` streamed during the decode (see
/// [`load_streamed`]); a pass without one computes its stage 1 itself. Same result.
#[allow(clippy::too_many_arguments)]
pub fn run_pre(
    x: &[f32],
    ch: usize,
    sr: u32,
    a: &Cli,
    clock: &Clock,
    log: &mut dyn FnMut(&str),
    budget: &dyn CandBudget,
    mut pre: Pre,
) -> Result<RunOut, Fail> {
    let mut analyze = |x: &[f32], ch, sr, r, o: &AnalyzeOpts, m| {
        let s1 = pre.take(r, x, ch, sr);
        analyze_pre(x, ch, sr, r, o, m, budget, s1)
    };
    let t0 = clock.time();
    let ra = run_args(a);
    let p = params::resolve(&ra)?;
    let r#ref = a.r#ref.unwrap_or(0.0);
    let (play, line_hz) = (p.play, p.line_hz);
    let srf = sr as f64;
    let chan = a.channel.parse::<Channel>()?;
    let ao = analyze_opts(a, &p)?;
    let mut rep = Value::obj();

    let mut meta0 = IndexMap::new();
    meta0.insert("source".to_string(), MetaValue::from(a.input.as_str()));
    let mut c1 = scaled(analyze(x, ch, sr, p.centre, &ao, Some(&meta0))?, play);
    c1.meta.insert("ref_hz".into(), r#ref.into());
    c1.meta
        .insert("search_centre_hz".into(), round_dec(p.centre, 3).into());
    if !c1.ratio.iter().all(|v| v.is_finite()) {
        return Err(Fail::Error(
            "AssertionError".into(),
            "non-finite ratio from analyze".into(),
        ));
    }
    ts_io::prof::mark("analyze#1 rest");
    let bw_meta = c1
        .meta
        .get("bw")
        .cloned()
        .unwrap_or(MetaValue::Str(String::new()));
    let v1 = frac_true(&c1.valid);
    rep.push(
        "analyze",
        Value::obj()
            .with("bw", meta_json(&bw_meta))
            .with("valid", v1)
            .with("s", clock.time() - t0),
    );
    log(&format!(
        "analyze {} Hz (centre {}): bw {}, valid {} %",
        fmt_general(r#ref, 6),
        fmt_general(p.centre, 6),
        bw_meta.to_text(),
        fmt_fixed(v1 * 100.0, 1)
    ));

    let mut c2: Option<Curve> = None;
    if let Some(lh) = truthy(line_hz) {
        let c = scaled(analyze(x, ch, sr, lh * play, &ao, None)?, play);
        if !c.ratio.iter().all(|v| v.is_finite()) {
            return Err(Fail::Error("AssertionError".into(), String::new()));
        }
        log(&format!(
            "analyze {} Hz: bw {}, valid {} %",
            fmt_general(lh, 6),
            c.meta.get("bw").map(|v| v.to_text()).unwrap_or_default(),
            fmt_fixed(frac_true(&c.valid) * 100.0, 1)
        ));
        c2 = Some(c);
        ts_io::prof::mark("analyze#2(line) rest");
    }

    // partner beat on the pass-1-normalised spectrum (relative variation only)
    let mut pm: Option<PairResult> = None;
    if let Abs::Pair(hz) = p.abs {
        let t1 = clock.time();
        let xc: Vec<f64> = match line_hz {
            Some(l) if absolute::XCHECK.contains(&l) => vec![l],
            _ => vec![],
        };
        let pp = pair_params(a, hz, &xc, play)?;
        ts_io::prof::mark("pre-pair");
        let mut m = pair_measure(x, ch, srf, &c1, r#ref, &pp)?;
        ts_io::prof::mark("pair.measure");
        let mut lines = vec![format!(
            "pair ({} Hz spacing, |d| <= {} Hz): {}",
            fmt_general(hz, 6),
            fmt_general(a.pair_dmax, 6),
            m.ambiguity
        )];
        lines.extend(pair_table(&m));
        log(&lines.join("\n"));
        // beat switching: pass 1 follows whichever beat is stronger; rho from the pair, else
        // from the ghost signature
        let solved = m.solution.is_some();
        let rho = m.rho().or_else(|| ghost_rho(&m, 2.5));
        if let Some(rho) = rho.filter(|&r| r != 0.0)
            && !a.no_switch_repair
        {
            let (c1r, sw) = absolute::switch_repair(
                x,
                ch,
                srf,
                &c1,
                r#ref,
                rho,
                chan,
                absolute::repair::SW_BLK,
                absolute::repair::SW_SEG,
            )?;
            ts_io::prof::mark("pair.switch_repair");
            rep.push("switch", sw.to_value());
            c1.meta.insert("sw_rho".into(), round_dec(rho, 7).into());
            c1.meta.insert("sw_spans".into(), sw.sw_spans.into());
            c1.meta.insert("sw_s".into(), round_dec(sw.sw_s, 1).into());
            c1.meta
                .insert("sw_major".into(), sw.sw_major.as_str().into());
            log(&format!(
                "beat switching (rho {}{}): {} span(s), {} s on the other beat, repaired to the {} beat",
                fmt_fixed(rho, 6),
                if solved {
                    ""
                } else {
                    ", from the ghost line pair"
                },
                sw.sw_spans,
                fmt_fixed(sw.sw_s, 0),
                sw.sw_major
            ));
            if sw.sw_spans > 0 {
                c1.ratio = c1r.ratio;
                m = pair_measure(x, ch, srf, &c1, r#ref, &pp)?;
                ts_io::prof::mark("pair.remeasure");
                let mut lines = vec![format!("pair on the repaired curve: {}", m.ambiguity)];
                lines.extend(pair_table(&m));
                log(&lines.join("\n"));
            }
        }
        rep.push("pair", m.to_value());
        rep.push("pair_s", clock.time() - t1);
        if let Some(e) = &m.error {
            let mut msg = vec![e.clone()];
            msg.extend(pair_table(&m));
            return Err(Fail::Exit(msg.join("\n")));
        }
        pm = Some(m);
    }

    let v_phys = p.v_phys;
    let so = StepOpts {
        mode: a.model.parse::<ModelMode>()?,
        min_ev: a.model_min_ev,
        b: a.model_band,
        ta: a.model_ta,
        channel: chan,
        v_phys,
        v_given: v_phys.is_some(),
        ..StepOpts::default()
    };
    ts_io::prof::mark("pre-model");
    let (mut cm, mrep, fit) = model_step(x, ch, sr, r#ref, &c1, &so)?;
    rep.push("model", mrep.to_value());
    log(&report_text(&mrep));
    let path = mrep.path.clone();
    let gaps = long_gaps_default(&cm.t, &cm.valid)?;
    if !gaps.is_empty() {
        // no reference for >= LONG_GAP_S: slow fallback (edge-linear / L1), explicit warning
        let (r_, v_, l1g) = fill_long_gaps_default(&cm.t, &cm.ratio, &cm.valid, &gaps)?;
        let dt = cm.t[1] - cm.t[0];
        let gs = gaps
            .iter()
            .map(|&(ga, gb)| format!("{}-{}", fmt_fixed(cm.t[ga], 1), fmt_fixed(cm.t[gb - 1], 1)))
            .collect::<Vec<_>>()
            .join(" ");
        let tot = gaps.iter().map(|&(ga, gb)| gb - ga).sum::<usize>() as f64 * dt;
        let tl = cm.t[cm.t.len() - 1];
        cm.confidence = cm
            .confidence
            .iter()
            .zip(&v_)
            .map(|(&q, &v)| if v { q } else { 0.0 })
            .collect();
        cm.ratio = r_;
        cm.valid = v_;
        cm.meta.insert("ref_gaps".into(), gs.as_str().into());
        cm.meta.insert("ref_gap_s".into(), round_dec(tot, 1).into());
        cm.meta
            .insert("ref_gap_fill".into(), "edge-linear / L1 ramp".into());
        cm.meta
            .insert("ref_gap_L1".into(), round_dec(l1g, 7).into());
        cm.meta.insert(
            "ref_coverage_pct".into(),
            round_dec(100.0 * (1.0 - tot / tl), 1).into(),
        );
        rep.push(
            "ref_gaps",
            Value::obj()
                .with("spans", gs.as_str())
                .with("total_s", tot)
                .with("L1", l1g),
        );
        log(&format!(
            "WARNING: no usable reference for {} s ({} % of the file: {gs}); there only the slow level is \
             corrected (linear between gap edges, else ramp to L1 {}), no wow/flutter correction",
            fmt_fixed(tot, 0),
            fmt_fixed(100.0 * tot / tl, 0),
            fmt_fixed(l1g, 5)
        ));
    }

    ts_io::prof::mark("model.rest+gaps");
    // evidence of tape origin and correction bandwidth
    let comps_present = fit.as_ref().is_some_and(|f| !f.hier.comps.is_empty());
    let mpred: Option<Vec<f64>> = if comps_present {
        fit.as_ref().map(|f| f.final_model().predict(&cm.t))
    } else {
        None
    };
    let ones = vec![true; cm.t.len()];
    let mut extra: Vec<ExtraCurve> = vec![("final", &cm.ratio, &cm.valid)];
    if let Some(mp) = &mpred {
        extra.push(("model", mp, &ones));
    }
    let fad = fading_metric(
        x,
        ch,
        srf,
        r#ref,
        &c1,
        chan,
        &FadingParams::default(),
        &extra,
    );
    let (fad_v, fad_rho, fad_fm) = match &fad {
        Ok(f) => (f.to_value(), Some(f.rho), Some(f.fm_ratio)),
        // evidence is advisory; never block the correction
        Err(e) => (
            Value::obj().with(
                "error",
                format!("ValueError({})", absolute::quote_str(&e.0)),
            ),
            None,
            None,
        ),
    };
    let mut coh: Option<BandStats> = None;
    if let Some(c2) = &c2 {
        let n = c1.t.len().min(c2.t.len());
        let com: Vec<bool> = (0..n).map(|i| c1.valid[i] && c2.valid[i]).collect();
        let q = if com.iter().any(|&b| b) {
            let v: Vec<f64> = (0..n)
                .filter(|&i| com[i])
                .map(|i| c1.ratio[i] / c2.ratio[i])
                .collect();
            median(&v)
        } else {
            1.0
        };
        let r1q: Vec<f64> = c1.ratio[..n].iter().map(|v| v / q).collect();
        let st = band_stats_default(&c1.t[..n], &r1q, &c2.ratio[..n], &com)?;
        let rmq: Vec<f64> = cm.ratio[..n].iter().map(|v| v / q).collect();
        let vf: Vec<bool> = (0..n).map(|i| cm.valid[i] && c2.valid[i]).collect();
        let cf = band_stats_default(&c1.t[..n], &rmq, &c2.ratio[..n], &vf)?;
        rep.push("coherence", st.to_value());
        rep.push("coherence_final", cf.to_value());
        coh = Some(st);
    }
    let climit = p.corr_limit.parse::<CorrLimit>()?;
    let (fc, why) = bandwidth_policy(&path, fad_rho, coh.as_ref(), a.max_corr_hz, climit);
    rep.push("fading", fad_v);
    rep.push("corr_max_hz", opt_f(fc));
    rep.push("corr_policy", why.as_str());
    let (ratio, keep_hf) = apply_bandwidth(&cm.t, &cm.ratio, fc, mpred.as_deref())?;
    if keep_hf {
        rep.push("corr_keep_model_hf", true);
    }
    let mut m = cm.meta.clone();
    m.insert(
        "fade_rho".into(),
        round_dec(fad_rho.unwrap_or(f64::NAN), 3).into(),
    );
    m.insert(
        "fade_fm_ratio".into(),
        round_dec(fad_fm.unwrap_or(f64::NAN), 2).into(),
    );
    m.insert(
        "corr_max_hz".into(),
        fc.map(MetaValue::Float).unwrap_or("none".into()),
    );
    m.insert("corr_policy".into(), why.as_str().into());
    m.insert("corr_keep_model_hf".into(), (keep_hf as i64).into());
    let mut coh_bands = None;
    if let Some(st) = coh.as_ref().filter(|s| !s.coh.is_empty()) {
        m.insert("coh_line_hz".into(), opt_meta(line_hz));
        let s = st
            .coh
            .iter()
            .map(|(k, v)| format!("{k}:{}", fmt_fixed(*v, 3)))
            .collect::<Vec<_>>()
            .join(" ");
        m.insert("coh_bands".into(), s.as_str().into());
        coh_bands = Some(s);
    }
    log(&format!(
        "evidence: fading rho {} (deep-dip/flat residual FM {}x){} -> correction band {} ({why})",
        fmt_fixed(fad_rho.unwrap_or(f64::NAN), 2),
        fmt_fixed(fad_fm.unwrap_or(f64::NAN), 1),
        match &coh_bands {
            Some(s) => format!(
                ", coherence with {} Hz: {s}",
                fmt_general(line_hz.unwrap_or(f64::NAN), 6)
            ),
            None => String::new(),
        },
        match fc {
            None => "full".to_string(),
            Some(f) => format!("<= {} Hz", fmt_general(f, 6)),
        }
    ));

    ts_io::prof::mark("evidence");
    // absolute speed
    let cl = Curve {
        t: cm.t.clone(),
        ratio,
        confidence: cm.confidence.clone(),
        valid: cm.valid.clone(),
        meta: m.clone(),
    };
    let (s, info) = absolute::apply_l1(&cl, r#ref, &p.abs, play, c2.as_ref(), 60.0, pm.as_ref())?;
    for (k, v) in &info {
        let v = match v {
            MetaValue::Float(f) => MetaValue::Float(round_dec(*f, 7)),
            o => o.clone(),
        };
        m.insert(k.clone(), v);
    }
    let consistent = mrep
        .physics
        .as_ref()
        .map(|ph| ph.consistent.clone())
        .unwrap_or_default();
    m.insert(
        "tape_speed_given".into(),
        p.v_rec
            .filter(|&v| v != 0.0)
            .map(fmt_speed)
            .unwrap_or("none".into())
            .into(),
    );
    m.insert(
        "play_speed".into(),
        p.v_play
            .filter(|&v| v != 0.0)
            .map(fmt_speed)
            .unwrap_or("none".into())
            .into(),
    );
    let inferred = consistent
        .iter()
        .map(|&v| fmt_speed(v))
        .collect::<Vec<_>>()
        .join(",");
    m.insert(
        "tape_speed_inferred".into(),
        if inferred.is_empty() {
            "none".into()
        } else {
            inferred
        }
        .into(),
    );
    let mut infov = Value::obj();
    for (k, v) in &info {
        infov.push(k, meta_json(v));
    }
    rep.push("abs", infov);
    let f = |k: &str| match info.get(k) {
        Some(MetaValue::Float(x)) => *x,
        Some(MetaValue::Int(i)) => *i as f64,
        _ => f64::NAN,
    };
    let sv = |k: &str| info.get(k).map(|v| v.to_text()).unwrap_or_default();
    let mut l1 = format!(
        "L1 ({}): mean speed {} ({} %), implied true line {} Hz",
        sv("abs_mode"),
        fmt_fixed(f("abs_l1"), 6),
        fmt_signed(f("abs_l1_pct"), 3),
        fmt_fixed(f("abs_ref_true_hz"), 1)
    );
    if matches!(p.abs, Abs::Line(_)) {
        l1 += &format!(
            "; line {} Hz: q {} (primary {} % off nominal), q spread {} %, drift {} %",
            fmt_general(line_hz.unwrap_or(f64::NAN), 6),
            fmt_fixed(f("line_q"), 6),
            fmt_signed(f("line_disagree_pct"), 3),
            fmt_fixed(f("line_q_std_pct"), 3),
            fmt_fixed(f("line_q_drift_pct"), 3)
        );
    }
    if matches!(p.abs, Abs::Pair(_)) {
        l1 += &format!(
            "; pair {} / {} Hz true, d {} Hz ({}; station {} Hz from the raster if the primary is the lower \
             neighbour's beat, else {})",
            fmt_fixed(f("pair_lo_hz"), 2),
            fmt_fixed(f("pair_hi_hz"), 2),
            fmt_fixed(f("pair_delta_hz"), 2),
            sv("pair_side"),
            fmt_signed(f("pair_offset_hz"), 2),
            fmt_signed(-f("pair_offset_hz"), 2)
        );
        if info.contains_key("pair_blk_delta_ptp") {
            l1 += &format!(
                ", block spread f_true {} ppm rms / d {} Hz ptp",
                fmt_fixed(f("pair_blk_ftrue_std_ppm"), 1),
                fmt_fixed(f("pair_blk_delta_ptp"), 2)
            );
        }
        if info.contains_key("pair_xcheck_ppm") {
            l1 += &format!(", cross-check {} ppm", sv("pair_xcheck_ppm"));
        }
    }
    l1 += &format!(
        "; tape speed given {}, inferred {}",
        m.get("tape_speed_given")
            .map(|v| v.to_text())
            .unwrap_or_default(),
        m.get("tape_speed_inferred")
            .map(|v| v.to_text())
            .unwrap_or_default()
    );
    log(&l1);
    if !s.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err(Fail::Error(
            "AssertionError".into(),
            "non-finite speed curve".into(),
        ));
    }
    ts_io::prof::mark("l1");
    rep.push("runtime_s", clock.time() - t0);
    let curve = Curve {
        t: cm.t.clone(),
        ratio: s,
        confidence: cm.confidence.clone(),
        valid: cm.valid.clone(),
        meta: m,
    };
    Ok(RunOut {
        curve,
        rep,
        c1,
        cm,
        fit,
        mrep,
    })
}

fn opt_meta(v: Option<f64>) -> MetaValue {
    match v {
        Some(x) => MetaValue::Float(x),
        None => MetaValue::Str("None".into()),
    }
}

/// Fixed-point with explicit sign (`+1.25`, `-0.50`).
pub fn fmt_signed(x: f64, n: usize) -> String {
    let s = fmt_fixed(x, n);
    if s.starts_with('-') {
        s
    } else {
        format!("+{s}")
    }
}

/// One planned extra notch (`todo` entry of `extra_notches`).
struct Todo {
    hz: f64,
    spans: Option<Vec<(f64, f64)>>,
    auto: Option<(f64, f64, f64)>, // total_s, snr_db, contrast
}

/// Resampler + post chain into a sink, with gated notches when present.
#[allow(clippy::too_many_arguments)]
fn correct_into<E: Send + std::fmt::Display>(
    x: &[f32],
    ch: usize,
    sr: u32,
    c: &Curve,
    p: &Params,
    post: &Post,
    gated: &[interf::GatedSpec],
    sink: impl FnMut(&[f32]) -> Result<(), E> + Send,
) -> Result<usize, RunError<E>> {
    if gated.is_empty() {
        correct::run(x, ch, sr, c, p, post, sink)
    } else {
        interf::run_gated(
            x,
            ch,
            sr,
            &c.t,
            &c.ratio,
            p,
            post,
            gated,
            &Gate::default(),
            sink,
        )
    }
}

fn sink_err<E: std::fmt::Display>(e: RunError<E>) -> Fail {
    match e {
        RunError::Value(v) => Fail::value(v.0),
        RunError::Sink(s) => Fail::Error("OSError".into(), s.to_string()),
    }
}

/// Create the missing parent directories of output paths.
pub fn make_parents<'a>(paths: impl IntoIterator<Item = &'a String>) -> Result<(), Fail> {
    for p in paths {
        if let Some(d) = std::path::Path::new(p).parent()
            && !d.as_os_str().is_empty()
        {
            std::fs::create_dir_all(d)
                .map_err(|e| Fail::Error("OSError".into(), format!("{}: {e}", d.display())))?;
        }
    }
    Ok(())
}

/// One file of `main(argv)` (non-survey path): the printed lines go to `out`. The input is
/// loaded as `job.plan` says (RAM or spill); the job's reservation is held by the caller.
pub fn run_file(job: &Job, ctx: &Ctx, out: &mut dyn FnMut(&str)) -> Result<(), Fail> {
    macro_rules! println {
        ($($t:tt)*) => { out(&format!($($t)*)) };
    }
    let a = &job.a;
    let clock = Clock::new();
    let ra = run_args(a);
    let fp = params::out_paths(&ra)?;
    for n in &job.notes {
        println!("{n}");
    }
    make_parents(
        [&fp.audio, &fp.curve, &fp.report]
            .into_iter()
            .chain(fp.plot.iter())
            .chain(fp.model_plot.iter()),
    )?;
    ts_io::prof::start();
    let t0 = clock.time();
    let oserr = |e: ts_io::Error| Fail::Error("OSError".into(), e.to_string());
    let mode = job.plan.as_ref().map_or(Mode::Ram, |p| p.mode);
    // errors in the pass arguments surface in run_pre, so they keep their order relative to I/O errors
    let passes = stage1_passes(a).ok();
    let (audio, pre) = load_streamed(&a.input, mode, &ctx.tmp, passes).map_err(oserr)?;
    let (ch, sr) = (audio.channels, audio.sr);
    if let Some(n) = ts_io::output_note(&fp.audio, ch) {
        println!("{n}");
    }
    let x: &[f32] = &audio;
    let RunOut {
        curve: mut c,
        mut rep,
        c1,
        cm,
        fit,
        mrep: _,
    } = run_pre(x, ch, sr, a, &clock, out, &ctx.budget, pre)?;
    let r#ref = a.r#ref.unwrap_or(0.0);
    let true_hz = match c.meta.get("abs_ref_true_hz") {
        Some(MetaValue::Float(v)) => *v,
        Some(MetaValue::Int(i)) => *i as f64,
        _ => r#ref,
    };
    // the line sits here after correction
    let (notch_hz, notch_why) = params::ref_notch(&ra, true_hz);
    let nz = truthy(notch_hz);
    c.meta.insert(
        "notch_hz".into(),
        nz.map(|v| MetaValue::Float(round_dec(v, 2)))
            .unwrap_or("none".into()),
    );
    c.meta.insert(
        "notch_q".into(),
        if nz.is_some() {
            MetaValue::Float(a.notch_q)
        } else {
            "none".into()
        },
    );
    rep.push(
        "notch",
        Value::obj()
            .with("hz", opt_f(notch_hz))
            .with("q", a.notch_q),
    );
    let lp_on = truthy(a.lowpass);
    c.meta.insert(
        "lowpass_hz".into(),
        lp_on.map(MetaValue::Float).unwrap_or("none".into()),
    );
    write_curve(&fp.curve, &c).map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
    if let Some(d) = std::env::var_os("TAPESPEED_DUMP_RATIO") {
        // debug: the full-precision final ratio (raw f64 LE) for comparisons below CSV precision
        let b: Vec<u8> = c.ratio.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(d, b).map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
    }
    if let Some(pp) = &fp.plot {
        let mut fm: Vec<f64> = fit
            .as_ref()
            .map(|f| f.hier.comps.iter().map(|cp| cp.f_mean).collect())
            .unwrap_or_default();
        fm.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
        let title = Path::new(&a.input)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let nm =
            ts_plot::plot_speed(&c1, &c, &cm, &fm, Path::new(pp), &title).map_err(Fail::runtime)?;
        rep.push(
            "plot",
            Value::obj()
                .with("wow_pct", nm.wow_pct)
                .with("flutter_pct", nm.flutter_pct)
                .with("l1_pct", nm.l1_pct),
        );
        println!("plot {pp}");
        if let Some(mp) = &fp.model_plot {
            let mf = fit.as_ref().map(|f| model_fit(f, &c1.t));
            ts_plot::plot_model(&c1, &cm, mf.as_ref(), Path::new(mp)).map_err(Fail::runtime)?;
        }
    }
    ts_io::prof::mark("curve+plot");
    let rmin = c.ratio.iter().cloned().fold(f64::INFINITY, f64::min);
    let rmax = c.ratio.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "curve {}: path {}, EV {}, valid {} %, ratio {}..{}",
        fp.curve,
        c.meta
            .get("model_path")
            .map(|v| v.to_text())
            .unwrap_or_default(),
        c.meta
            .get("model_ev")
            .map(|v| v.to_text())
            .unwrap_or_default(),
        fmt_fixed(frac_true(&c.valid) * 100.0, 1),
        fmt_fixed(rmin, 5),
        fmt_fixed(rmax, 5)
    );
    let t1 = clock.time();
    let rp = Params {
        taps: usize::try_from(a.taps).map_err(|_| Fail::value("taps must be >= 0"))?,
        ..Params::default()
    };
    let ref_notches: Vec<(f64, f64)> = nz.map(|h| vec![(h, a.notch_q)]).unwrap_or_default();
    println!(
        "reference tone {}",
        match nz {
            None => notch_why.clone(),
            Some(h) => format!(
                "notched at {} Hz (Q {})",
                fmt_fixed(h, 1),
                fmt_general(a.notch_q, 6)
            ),
        }
    );

    // extra notches: manual (whole file) and detected (gated); detection on the corrected,
    // reference-notched output (pass 1), keying statistics on the same signal (pass 2)
    let mut todo: Vec<Todo> = params::manual_notches(&ra)?
        .into_iter()
        .map(|hz| Todo {
            hz,
            spans: None,
            auto: None,
        })
        .collect();
    let first = Post {
        notches: ref_notches.clone(),
        lowpass: None,
    };
    if let Some(mode) = &a.auto_notch {
        let ta = clock.time();
        let dp = DetectParams {
            fmin: a.auto_notch_min,
            ..DetectParams::default()
        };
        let mut det = Detector::new(sr, &dp)?;
        correct::run(x, ch, sr, &c, &rp, &first, |b: &[f32]| {
            det.push_frames(b, ch)
        })
        .map_err(sink_err)?;
        let dets = det.finish()?;
        ts_io::prof::mark("autonotch.detect");
        let keys = if dets.is_empty() {
            vec![]
        } else {
            let lines: Vec<(f64, Vec<(f64, f64)>)> =
                dets.iter().map(|d| (d.hz, d.spans.clone())).collect();
            let mut ks = KeyingStream::new(sr, &lines);
            correct::run(x, ch, sr, &c, &rp, &first, |b: &[f32]| {
                ks.push_frames(b, ch)
            })
            .map_err(sink_err)?;
            ks.finish()?
        };
        ts_io::prof::mark("autonotch.keying");
        for (d, &(con, mid)) in dets.iter().zip(&keys) {
            if mode == "keyed" && !is_keyed(con, mid) {
                let fate = if d.hz >= params::lowpass_stop(&ra) {
                    "removed by lowpass"
                } else {
                    "kept"
                };
                println!(
                    "line {} Hz: steady (contrast {} dB, mid {}), {fate}",
                    fmt_fixed(d.hz, 1),
                    fmt_fixed(con, 1),
                    fmt_fixed(mid, 2)
                );
                continue;
            }
            todo.push(Todo {
                hz: d.hz,
                spans: Some(d.spans.clone()),
                auto: Some((d.total_s, d.snr_db, con)),
            });
        }
        rep.push("auto_notch_s", clock.time() - ta);
    }
    let mut post = first;
    let mut gated: Vec<interf::GatedSpec> = vec![];
    let mut done = vec![];
    for d in &todo {
        let src = if d.auto.is_some() { "auto" } else { "manual" };
        if let Some(why) = params::notch_skip(&ra, d.hz, notch_hz) {
            println!(
                "notch {} Hz ({src}): skipped, covered by {why}",
                fmt_fixed(d.hz, 1)
            );
            continue;
        }
        let q = d.hz / a.notch_width;
        let where_ = match &d.spans {
            None => {
                post.notches.push((d.hz, q));
                "whole file".to_string()
            }
            Some(sp) => {
                gated.push((d.hz, q, sp.clone()));
                format!(
                    "{} s in {} span(s)",
                    fmt_fixed(sp.iter().map(|(s0, s1)| s1 - s0).sum::<f64>(), 0),
                    sp.len()
                )
            }
        };
        let tail = match d.auto {
            Some((tot, snr, con)) => format!(
                ", present {} s, SNR {} dB, keying contrast {} dB",
                fmt_fixed(tot, 0),
                fmt_fixed(snr, 0),
                fmt_fixed(con, 1)
            ),
            None => String::new(),
        };
        println!(
            "notch {} Hz ({src}, width {} Hz): {where_}{tail}",
            fmt_fixed(d.hz, 1),
            fmt_general(a.notch_width, 6)
        );
        done.push((d.hz, d.spans.clone().unwrap_or_default(), src));
    }
    if let Some(lp) = lp_on {
        post.lowpass = Some(Lowpass {
            fc: lp,
            width: a.lowpass_width,
            ..Lowpass::new(lp)
        });
    }
    ts_io::prof::mark("pre-correct");
    let mut w = ts_io::Writer::create(&fp.audio, sr, ch, None)
        .map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
    let res = correct_into(x, ch, sr, &c, &rp, &post, &gated, |b: &[f32]| w.write(b));
    let n_out = match res {
        Ok(n) => {
            ts_io::prof::mark("correct+notch+lp (pipe)");
            w.finish()
                .map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
            n
        }
        // a broken pipe means ffmpeg died: prefer its stderr
        Err(e) => {
            return Err(match w.finish() {
                Err(fe) => Fail::Error("OSError".into(), fe.to_string()),
                Ok(()) => sink_err(e),
            });
        }
    };
    if let Some(lp) = lp_on {
        println!(
            "lowpass {} Hz (transition ±{} Hz, 80 dB)",
            fmt_general(lp, 6),
            fmt_general(a.lowpass_width / 2.0, 6)
        );
    }
    let en = done
        .iter()
        .map(|(hz, sp, _)| {
            if sp.is_empty() {
                format!("{}@all", fmt_fixed(*hz, 1))
            } else {
                format!(
                    "{}@{}",
                    fmt_fixed(*hz, 1),
                    sp.iter()
                        .map(|(s0, s1)| format!("{}-{}", fmt_fixed(*s0, 0), fmt_fixed(*s1, 0)))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    c.meta.insert(
        "extra_notches".into(),
        if en.is_empty() { "none".into() } else { en }.into(),
    );
    write_curve(&fp.curve, &c).map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
    rep.push(
        "extra_notches",
        Value::Arr(
            done.iter()
                .map(|(hz, sp, src)| {
                    Value::obj()
                        .with("hz", *hz)
                        .with(
                            "spans",
                            Value::Arr(
                                sp.iter()
                                    .map(|&(s0, s1)| {
                                        Value::Arr(vec![Value::Float(s0), Value::Float(s1)])
                                    })
                                    .collect(),
                            ),
                        )
                        .with("src", *src)
                })
                .collect(),
        ),
    );
    ts_io::prof::mark("encode finish, curve");
    let dur = audio.frames() as f64 / sr as f64;
    rep.push("correct_s", clock.time() - t1);
    let total = clock.time() - t0;
    rep.push("total_s", total);
    rep.push("duration_s", dur);
    std::fs::write(&fp.report, rep.dumps(Some(1)))
        .map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
    println!(
        "audio {}: {} s -> {} s @ {sr} Hz; {} s total ({} min per hour of audio)",
        fp.audio,
        fmt_fixed(dur, 3),
        fmt_fixed(n_out as f64 / sr as f64, 3),
        fmt_fixed(total, 0),
        fmt_fixed(total / dur * 3600.0 / 60.0, 1)
    );
    ts_io::prof::mark("report");
    ts_io::prof::end();
    Ok(())
}

/// One planned file (`plan` fails if the input cannot be probed).
pub struct Job {
    pub a: Cli,
    pub plan: Result<Plan, Fail>,
    /// Lines printed before the run (warnings about adjusted options).
    pub notes: Vec<String>,
}

/// What all jobs of a run share.
pub struct Ctx {
    pub budget: Budget,
    pub why: String,
    pub tmp: std::path::PathBuf,
    pub threads: usize,
    pub jobs: usize,
}

/// Machine probe, budget, and a plan per file (`a` already expanded per input).
pub fn plan_files(files: &[Cli], top: &Cli) -> Result<(Ctx, Vec<Job>), Fail> {
    let m = ts_io::machine::probe();
    let (cap, why) = plan::budget_of(&m, top.max_ram.as_deref()).map_err(Fail::Exit)?;
    let tmp = ts_io::spill::tmp_dir(top.tmp_dir.as_deref());
    let threads = rayon::current_num_threads();
    let free = ts_io::spill::free_bytes(&tmp);
    let plan_one = |a: &Cli| -> Result<(Plan, usize), Fail> {
        let inf =
            ts_io::info(&a.input).map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
        let secs = if inf.duration.is_finite() {
            inf.duration
        } else {
            // no container duration: bit rate, else the PCM size bound
            let sz = std::fs::metadata(&a.input).map(|m| m.len()).unwrap_or(0) as f64;
            let bps = if inf.bit_rate > 0 {
                inf.bit_rate as f64 / 8.0
            } else {
                4.0 * (inf.sr as usize * inf.channels) as f64
            };
            sz / bps
        };
        let frames = ((secs + 1.0) * inf.sr as f64) as u64;
        let est = plan::estimate(frames, inf.channels, inf.sr, a)?;
        let spill_ok = tmp.is_dir() && free.is_some_and(|f| f > est.x + est.x / 20);
        Ok((plan::decide(&est, cap, spill_ok), inf.channels))
    };
    let jobs: Vec<Job> = files
        .iter()
        .map(|a| {
            let r = plan_one(a);
            let mut a = a.clone();
            let mut notes = vec![];
            // --channel L/R/S on a mono file: the mono signal is analysed
            if matches!(r, Ok((_, 1))) && a.channel != "mid" {
                notes.push(format!(
                    "warning: --channel {} on a mono file: analysing the mono signal (as --channel mid)",
                    a.channel
                ));
                a.channel = "mid".into();
            }
            Job {
                a,
                plan: r.map(|(p, _)| p),
                notes,
            }
        })
        .collect();
    let n = plan::max_jobs(threads, jobs.len());
    Ok((
        Ctx {
            budget: Budget::new(cap),
            why,
            tmp,
            threads,
            jobs: n,
        },
        jobs,
    ))
}

impl Ctx {
    /// The `memory:` line of one job.
    pub fn line(&self, p: &Plan) -> String {
        plan::line(
            p,
            self.budget.cap,
            &self.why,
            self.jobs,
            self.threads,
            &self.tmp.to_string_lossy(),
        )
    }
}

/// `main(argv)` after parsing, non-survey path: one file (lines stream to stdout)
/// or a batch (files in parallel within the shared budget, each file's lines printed as one
/// block when it finishes; failures go to stderr and make the exit status 1).
pub fn main_run(a: &Cli) -> Result<(), Fail> {
    let files = cli::expand(a).map_err(Fail::Usage)?;
    let outs: Vec<_> = files
        .iter()
        .filter_map(|b| {
            let fp = params::out_paths(&run_args(b)).ok()?;
            let mut v = vec![
                ("audio", fp.audio),
                ("curve", fp.curve),
                ("report", fp.report),
            ];
            v.extend(fp.plot.map(|p| ("plot", p)));
            v.extend(fp.model_plot.map(|p| ("model plot", p)));
            Some((b.input.clone(), v))
        })
        .collect();
    cli::protect_inputs(a, &outs).map_err(Fail::Usage)?;
    let (ctx, mut jobs) = plan_files(&files, a)?;
    if let [j] = &jobs[..] {
        let p = j.plan.as_ref().map_err(Fail::clone)?;
        println!("{}", ctx.line(p));
        let _lease = ctx.budget.acquire(p.hold);
        return run_file(j, &ctx, &mut |s: &str| println!("{s}"));
    }
    // largest reservation first (long files start early)
    jobs.sort_by_key(|j| std::cmp::Reverse(j.plan.as_ref().map_or(0, |p| p.hold)));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::atomic::AtomicUsize::new(0);
    let n = jobs.len();
    std::thread::scope(|sc| {
        for _ in 0..ctx.jobs {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(j) = jobs.get(i) else { break };
                    let p = match &j.plan {
                        Ok(p) => p,
                        Err(e) => {
                            failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("{}: {}", j.a.input, e.message());
                            continue;
                        }
                    };
                    let mut buf = format!("== {}\n{}\n", j.a.input, ctx.line(p));
                    let r = {
                        let _lease = ctx.budget.acquire(p.hold);
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_file(j, &ctx, &mut |s: &str| {
                                buf.push_str(s);
                                buf.push('\n');
                            })
                        }))
                    };
                    let mut o = std::io::stdout().lock();
                    let _ = o.write_all(buf.as_bytes());
                    let _ = o.flush();
                    drop(o);
                    let msg = match r {
                        Ok(Ok(())) => continue,
                        Ok(Err(e)) => e.message(),
                        Err(p) => p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "panic".into()),
                    };
                    failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("{}: {msg}", j.a.input);
                }
            });
        }
    });
    let nf = failed.into_inner();
    println!(
        "batch: {n} files, {} ok, {nf} failed; jobs {}, budget {} ({}), threads {}",
        n - nf,
        ctx.jobs,
        plan::gib(ctx.budget.cap),
        ctx.why,
        ctx.threads
    );
    if nf > 0 {
        return Err(Fail::Exit(format!("{nf} of {n} files failed")));
    }
    Ok(())
}

/// The fitted model on `t` for `plot_model` (final model, falling back to the third-stage model, and its components).
fn model_fit(f: &StepFit, t: &[f64]) -> ts_plot::ModelFit {
    let m = f.final_model();
    ts_plot::ModelFit {
        predict: m.predict(t),
        r0: m.r0,
        comps: f
            .hier
            .comps
            .iter()
            .map(|cp| ts_plot::CompTrack {
                chirp: cp.kind.as_str() == "chirp",
                harmonics: cp.harmonics.iter().map(|&h| h as f64).collect(),
                freq: cp.freq(t),
            })
            .collect(),
    }
}
