//! `--survey` against stored reference results (tests/data/k: two8, am9cli, am9plan) and
//! behaviour tests of candidates, the no-audio survey and the plan.
//!
//! Each case runs `survey::run` on the case's argv and compares the JSON line by line
//! (indented JSON = one value per line: keys / order / strings equal, numbers within
//! tolerance) and the report text. Trial tracks: `Replay` injects the stored analyze curves
//! (isolates the survey logic), `AnalyzeTracker` is the real analyze.
#[path = "../../testdata.rs"]
mod testdata;
use std::path::{Path, PathBuf};

use ts_core::curve::Curve;
use ts_core::hetdec::Channel;
use ts_core::survey::plan::SurveyArgs;
use ts_core::survey::{self, AnalyzeTracker, SurveyRun, Tracker};

fn small(case: &str) -> PathBuf {
    testdata::dir("k").join(case)
}

fn npy_f64(p: &Path) -> Vec<f64> {
    let f = std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let n = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    if n.dtype().descr().contains("b1") {
        n.into_vec::<bool>()
            .unwrap()
            .into_iter()
            .map(|b| b as u8 as f64)
            .collect()
    } else {
        n.into_vec::<f64>().unwrap()
    }
}

/// The stored analyze curves of the survey run, matched by (t0, f, channel).
struct Replay {
    trials: Vec<(f64, f64, String, Curve)>,
}

impl Replay {
    fn load(dir: &Path) -> Self {
        let txt = std::fs::read_to_string(dir.join("trials.txt")).unwrap_or_default();
        let trials = txt
            .lines()
            .map(|l| {
                let w: Vec<&str> = l.split(' ').collect();
                let t0 = if w[1] == "None" {
                    0.0
                } else {
                    w[1].parse().unwrap()
                };
                let c = |k: &str| npy_f64(&dir.join(format!("trial{}_{k}.npy", w[0])));
                let curve = Curve {
                    t: c("t"),
                    ratio: c("ratio"),
                    confidence: c("confidence"),
                    valid: c("valid").into_iter().map(|v| v != 0.0).collect(),
                    meta: Default::default(),
                };
                (t0, w[3].parse().unwrap(), w[4].to_string(), curve)
            })
            .collect();
        Replay { trials }
    }
}

impl Tracker for Replay {
    fn track(
        &self,
        _x: &[f32],
        _ch: usize,
        _sr: u32,
        f: f64,
        channel: Channel,
        t0: f64,
        _dur: f64,
    ) -> Result<Curve, String> {
        let name = match channel {
            Channel::Mid => "mid",
            Channel::L => "L",
            Channel::R => "R",
            Channel::S => "S",
        };
        self.trials
            .iter()
            .find(|(a, b, c, _)| (a - t0).abs() < 1e-6 && (b - f).abs() < 0.006 && c == name)
            .map(|t| t.3.clone())
            .ok_or_else(|| format!("no recorded trial for t0 {t0} f {f} {name}"))
    }
}

/// Minimal option parser for the options the cases use (same semantics as the CLI).
fn parse_args(argv: &[String], input: &str) -> SurveyArgs {
    let mut a = SurveyArgs {
        input: input.to_string(),
        ..Default::default()
    };
    let mut i = 1;
    let num = |s: &str| s.parse::<f64>().ok();
    while i < argv.len() {
        let k = argv[i].as_str();
        let v = argv.get(i + 1).cloned();
        let mut take = || {
            i += 1;
            v.clone().unwrap()
        };
        match k {
            "--survey" => {}
            "--survey-blocks" => a.survey_blocks = take().parse().unwrap(),
            "--survey-block-s" => a.survey_block_s = take().parse().unwrap(),
            "-o" | "--output" => a.output = Some(take()),
            "--ref" => a.r#ref = Some(take().parse().unwrap()),
            "--abs" => a.abs = take(),
            "--weak-rf" => a.weak_rf = true,
            "--keep-ref" => a.keep_ref = true,
            "--no-plot" => a.no_plot = true,
            "--notch-at" => a.notch_at = Some(take()),
            "--search" => a.search = take().parse().unwrap(),
            "--tape-speed" => a.tape_speed = Some(take()),
            "--play-speed" => a.play_speed = Some(take()),
            "--model-min-ev" => a.model_min_ev = take().parse().unwrap(),
            "--check-line" => a.check_line = Some(take().parse().unwrap()),
            "--auto-notch-min" => a.auto_notch_min = take().parse().unwrap(),
            "--lowpass" => {
                a.lowpass = Some(match v.as_deref().and_then(num) {
                    Some(x) => {
                        i += 1;
                        x
                    }
                    None => 8000.0,
                })
            }
            "--auto-notch" => {
                a.auto_notch = Some(match v.as_deref() {
                    Some(s @ ("keyed" | "all")) => {
                        i += 1;
                        s.to_string()
                    }
                    _ => "keyed".into(),
                })
            }
            _ => panic!("unhandled option {k}"),
        }
        i += 1;
    }
    a
}

/// Case: argv, stored reference outputs, and the input file.
struct Case {
    name: String,
    dir: PathBuf,
    argv: Vec<String>,
    input: PathBuf,
}

fn case_small(name: &str) -> Case {
    let dir = small(name);
    let argv: Vec<String> = std::fs::read_to_string(dir.join("argv.txt"))
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    Case {
        name: name.into(),
        input: dir.join("input.wav"),
        dir,
        argv,
    }
}

/// Run the case: the input is linked as argv[0] inside a temp dir; that dir prefix is removed
/// from the output afterwards (the reference was produced with cwd = that dir).
fn run_case(c: &Case, tracker: &dyn Tracker) -> (SurveyRun, String, String, f64) {
    let (path, prefix, _tmp);
    if c.argv[0].starts_with('/') {
        path = c.argv[0].clone();
        prefix = String::new();
        _tmp = None;
    } else {
        let tmp = std::env::temp_dir().join(format!(
            "survey_k_{}_{}_{:?}",
            c.name,
            std::process::id(),
            std::thread::current().id()
        ));
        let link = tmp.join(&c.argv[0]);
        std::fs::create_dir_all(link.parent().unwrap()).unwrap();
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(c.input.canonicalize().unwrap(), &link).unwrap();
        prefix = format!("{}/", tmp.display());
        path = link.display().to_string();
        _tmp = Some(tmp);
    }
    let a = parse_args(&c.argv, &path);
    let t = std::time::Instant::now();
    let r = survey::run(&a, tracker, Some(0.0)).unwrap_or_else(|e| panic!("{}: {e}", c.name));
    let el = t.elapsed().as_secs_f64();
    let strip = |s: &str| {
        if prefix.is_empty() {
            s.to_string()
        } else {
            s.replace(&prefix, "")
        }
    };
    let json = strip(&r.json);
    let stdout = strip(&format!("{}\n{}\n", r.text, r.final_line()));
    if let Some(t) = _tmp {
        let _ = std::fs::remove_dir_all(t);
    }
    (r, json, stdout, el)
}

#[derive(Default, Debug)]
struct Diff {
    lines: usize,
    num_bad: Vec<String>,
    str_bad: Vec<String>,
    max_f: f64,
    max_cn0: f64,
    max_rel: f64,
    max_trial: f64,
}

fn key_of(line: &str) -> Option<&str> {
    let t = line.trim_start();
    t.strip_prefix('"')
        .and_then(|r| r.split_once("\": ").map(|(k, _)| k))
}

fn scalar(line: &str) -> &str {
    let t = line.trim().trim_end_matches(',');
    match t.split_once("\": ") {
        Some((_, v)) if t.starts_with('"') => v,
        _ => t,
    }
}

/// Line-by-line JSON comparison; `trial_tol` loosens values that come from trial tracks.
fn cmp_json(want: &str, rs: &str, trial_tol: bool) -> Diff {
    let (pl, rl): (Vec<&str>, Vec<&str>) = (want.lines().collect(), rs.lines().collect());
    let mut d = Diff {
        lines: pl.len(),
        ..Default::default()
    };
    if pl.len() != rl.len() {
        d.str_bad
            .push(format!("line count {} vs {}", pl.len(), rl.len()));
        return d;
    }
    let mut last_key = String::new();
    let mut section = String::new();
    for (i, (a, b)) in pl.iter().zip(&rl).enumerate() {
        if let Some(k) = key_of(a) {
            last_key = k.to_string();
            if a.starts_with(" \"") {
                section = k.to_string();
            }
            if key_of(b) != Some(k) {
                d.str_bad.push(format!("{i}: key {a:?} vs {b:?}"));
                continue;
            }
        }
        if a == b {
            continue;
        }
        let (va, vb) = (scalar(a), scalar(b));
        match (va.parse::<f64>(), vb.parse::<f64>()) {
            (Ok(x), Ok(y)) => {
                let k = last_key.as_str();
                let e = (x - y).abs();
                // values rounded to <= 2 decimals (block f / cn0 / prom, ...): one unit
                // in the last place is a rounding-tie flip, not a difference
                let dec = va.split_once('.').map_or(0, |(_, f)| f.len());
                let unit = if dec <= 2 {
                    10f64.powi(-(dec as i32)) * (1.0 + 1e-9)
                } else {
                    0.0
                };
                let tol = if matches!(
                    k,
                    "f" | "a" | "b" | "hz" | "f_lo" | "f_hi" | "f_true" | "spread_hz"
                ) {
                    d.max_f = d.max_f.max(e);
                    0.01
                } else if k == "cn0" {
                    d.max_cn0 = d.max_cn0.max(e);
                    0.05
                } else {
                    d.max_rel = d.max_rel.max(e / y.abs().max(1.0));
                    let trial = trial_tol
                        && (matches!(section.as_str(), "coherence" | "pair")
                            || matches!(
                                k,
                                "trial_valid"
                                    | "common"
                                    | "0-0.5"
                                    | "0.5-6"
                                    | "6-25"
                                    | "abs_l1"
                                    | "true_hz"
                                    | "out_len_s"
                            ));
                    if trial {
                        d.max_trial = d.max_trial.max(e / y.abs().max(1.0));
                    }
                    if trial {
                        1e-2 * y.abs().max(1.0)
                    } else {
                        1e-6 * y.abs().max(1.0) + 1.5e-6
                    }
                };
                if e > tol.max(unit) {
                    d.num_bad.push(format!("{i} [{section}.{k}]: {a} vs {b}"));
                }
            }
            _ => d
                .str_bad
                .push(format!("{i} [{section}.{last_key}]: {a}\n      vs {b}")),
        }
    }
    d
}

fn cmp_text(want: &str, rs: &str) -> Vec<String> {
    let (pl, rl): (Vec<&str>, Vec<&str>) = (want.lines().collect(), rs.lines().collect());
    let mut bad: Vec<String> = pl
        .iter()
        .zip(&rl)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| format!("{i}: {a}\n    vs {b}"))
        .collect();
    if pl.len() != rl.len() {
        bad.push(format!("line count {} vs {}", pl.len(), rl.len()));
    }
    bad
}

fn primary_and_rec(json: &str) -> (String, String) {
    let v: serde_json::Value = serde_json::from_str(json).unwrap();
    let pid = v["primary"].clone();
    let pf = v["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == pid)
        .map(|c| c["f"].to_string())
        .unwrap_or("None".into());
    let r = &v["recommend"];
    let rec = ["ref", "channel", "abs", "check_line"]
        .iter()
        .map(|k| r.get(*k).map_or("-".into(), |x| x[0].to_string()))
        .collect::<Vec<_>>()
        .join(" ");
    (format!("{pid} @ {pf}"), rec)
}

/// Compare a case; returns (json diff, text diffs).
fn check(
    c: &Case,
    tracker: &dyn Tracker,
    trial_tol: bool,
    tag: &str,
) -> (Diff, Vec<String>, f64, SurveyRun) {
    let (r, json, stdout, el) = run_case(c, tracker);
    if std::env::var_os("TS_BLESS").is_some() && !trial_tol {
        // rewrite the stored reference results from this run (replay cases only)
        std::fs::write(c.dir.join("survey.json"), &json).unwrap();
        std::fs::write(c.dir.join("stdout.txt"), &stdout).unwrap();
    }
    let want = std::fs::read_to_string(c.dir.join("survey.json")).unwrap();
    let want_txt = std::fs::read_to_string(c.dir.join("stdout.txt")).unwrap();
    let d = cmp_json(&want, &json, trial_tol);
    let t = cmp_text(&want_txt, &stdout);
    println!(
        "{tag} {:14} {:5.2} s  json {} lines: {} num / {} str diffs (max |df| {:.1e} Hz, |dcn0| {:.1e} dB, rel {:.1e}, trial rel {:.1e}); text {} diff lines",
        c.name,
        el,
        d.lines,
        d.num_bad.len(),
        d.str_bad.len(),
        d.max_f,
        d.max_cn0,
        d.max_rel,
        d.max_trial,
        t.len()
    );
    for s in d.num_bad.iter().chain(&d.str_bad).take(12) {
        println!("    {s}");
    }
    for s in t.iter().take(6) {
        println!("    text {s}");
    }
    (d, t, el, r)
}

#[test]
fn small_cases_replay_exact() {
    for name in ["two8", "am9cli", "am9plan"] {
        let c = case_small(name);
        let (d, t, _, _) = check(&c, &Replay::load(&c.dir), false, "replay");
        assert!(
            d.num_bad.is_empty() && d.str_bad.is_empty() && t.is_empty(),
            "{name}: survey differs from the stored reference"
        );
    }
}

#[test]
fn small_cases_real_analyze() {
    for name in ["two8", "am9cli", "am9plan"] {
        let c = case_small(name);
        let (d, _, _, r) = check(&c, &AnalyzeTracker, true, "analyze");
        assert!(d.str_bad.is_empty() && d.num_bad.is_empty(), "{name}");
        let want = std::fs::read_to_string(c.dir.join("survey.json")).unwrap();
        assert_eq!(primary_and_rec(&r.json), primary_and_rec(&want));
    }
}

/// Two lines: AM beat as primary, the TV line as on-tape absolute reference (24 s two_lines,
/// 3 x 8 s blocks)
#[test]
fn test_candidates_two_lines() {
    let c = case_small("two8");
    let (r, ..) = run_case(&c, &Replay::load(&c.dir));
    let s = &r.survey;
    let p = s.cand(s.primary.unwrap()).unwrap();
    let f1_true = 8934.0;
    assert!(p.cls == "AM beat" && (p.f - f1_true * 1.01).abs() < 3.0);
    let tv = s.candidates.iter().find(|c| c.cls == "TV line").unwrap();
    assert!(
        (tv.f - 15625.0 * 1.01).abs() < 5.0 && tv.tape == Some(Some(true)) && s.coherence[0].tape
    );
    assert_eq!(r.recommend.r#ref.0.num(), Some(9000.0));
    assert_eq!(r.recommend.abs.as_ref().unwrap().0.as_str(), "line:15625");
    assert!((survey::plan::implied_true(p, tv).unwrap() - f1_true).abs() < 2.0);
    assert!(25.0 < p.cn0 && p.cn0 < 60.0 && p.verdict.starts_with("trackable"));
}

/// Survey from the CLI argv: recommended --ref, planned outputs and survey file names
/// (am9 12 s, 3 x 4 s, -o o.flac)
#[test]
fn test_survey_cli_no_audio() {
    let c = case_small("am9cli");
    let (r, json, ..) = run_case(&c, &Replay::load(&c.dir));
    let j: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(j["recommend"]["ref"][0], 9000.0);
    assert!(
        j["plan"]["resolved"]["ref_src"]
            .as_str()
            .unwrap()
            .starts_with("from survey")
    );
    assert_eq!(
        r.plan.resolved.as_ref().unwrap().outputs[0],
        ("audio".into(), "o.flac".into())
    );
    assert_eq!(
        (r.json_path.as_str(), r.png_path.as_str()),
        ("o_survey.json", "o_survey.png")
    );
}

/// The plan matches the run (plan part; the run values come from the stored reference)
#[test]
fn test_plan_matches_run() {
    let c = case_small("am9plan");
    let (r, ..) = run_case(&c, &Replay::load(&c.dir));
    let p = r.plan.resolved.unwrap();
    assert_eq!(p.centre, 9000.0);
    assert_eq!(p.abs_mode, "assume:9100.0");
    assert_eq!(p.corr_limit, "on");
    assert_eq!(p.notch_hz, None);
    assert_eq!(p.extra_notches, vec![5000.0]);
    assert!((p.search_hz[1] - 9000.0 * 1.04).abs() < 1e-9);
    assert_eq!(p.outputs[1], ("curve".into(), "r_curve.csv".into()));
}

#[test]
fn one_thread_equals_many() {
    let c = case_small("two8");
    let rp = Replay::load(&c.dir);
    let go = || run_case(&c, &rp).1;
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(go);
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(go);
    assert_eq!(one, many, "thread count changed the survey");
}
