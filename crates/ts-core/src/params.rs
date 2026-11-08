//! params: parameter resolution shared by the pipeline run and
//! the survey plan.
//!
//! [`RunArgs`] holds the parsed-argument fields the resolution reads (the CLI maps its `Cli`
//! onto it). Paths are handled as POSIX paths: normalised text, `stem`, `with_name`.

use crate::absolute::{Abs, AbsError, parse_abs, parse_float};
use crate::physics::{SpeedError, parse_speed};
use ts_io::fmt_general;

/// The command-line fields used by parameter resolution (`a.<name>`; CLI defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct RunArgs {
    pub r#ref: f64,
    pub output: String,
    pub abs: String,
    pub tape_speed: Option<String>,
    pub play_speed: Option<String>,
    pub bw: String,
    pub check_line: Option<f64>,
    pub weak_rf: bool,
    pub corr_limit: String,
    pub curve: Option<String>,
    pub report: Option<String>,
    pub plot: Option<String>,
    pub no_plot: bool,
    pub lowpass: Option<f64>,
    pub lowpass_width: f64,
    pub keep_ref: bool,
    pub notch_at: Option<String>,
    pub notch_width: f64,
}

impl Default for RunArgs {
    fn default() -> Self {
        RunArgs {
            r#ref: 9000.0,
            output: String::new(),
            abs: "ref".into(),
            tape_speed: None,
            play_speed: None,
            bw: "auto".into(),
            check_line: None,
            weak_rf: false,
            corr_limit: "auto".into(),
            curve: None,
            report: None,
            plot: None,
            no_plot: false,
            lowpass: None,
            lowpass_width: 500.0,
            keep_ref: false,
            notch_at: None,
            notch_width: 150.0,
        }
    }
}

/// Errors by class (exit with message, value, division by zero).
#[derive(Debug, thiserror::Error)]
pub enum ParamError {
    /// Fatal usage error: exit with the message.
    #[error("{0}")]
    Exit(String),
    /// Invalid value (float / --abs / tape speed parsing, paths).
    #[error("{0}")]
    Value(String),
    /// Division by zero.
    #[error("{0}")]
    ZeroDivision(String),
}

impl From<AbsError> for ParamError {
    fn from(e: AbsError) -> Self {
        ParamError::Value(e.to_string())
    }
}

impl From<SpeedError> for ParamError {
    fn from(e: SpeedError) -> Self {
        match e {
            SpeedError::Value(m) => ParamError::Value(m),
            SpeedError::ZeroDivision(m) => ParamError::ZeroDivision(m),
        }
    }
}

fn truthy(s: &Option<String>) -> Option<&str> {
    s.as_deref().filter(|v| !v.is_empty())
}

/// `play_factor(a)` → (recording speed, playback speed [m/s] or None, play factor
/// F = play / tape).
pub fn play_factor(a: &RunArgs) -> Result<(Option<f64>, Option<f64>, f64), ParamError> {
    let v_rec = truthy(&a.tape_speed).map(parse_speed).transpose()?;
    let v_play = truthy(&a.play_speed).map(parse_speed).transpose()?;
    let play_t = v_play.filter(|&v| v != 0.0);
    if play_t.is_some() && v_rec.is_none_or(|v| v == 0.0) {
        return Err(ParamError::Exit(
            "--play-speed needs --tape-speed (the recording speed)".into(),
        ));
    }
    let f = match play_t {
        Some(p) => p / v_rec.unwrap(),
        None => 1.0,
    };
    Ok((v_rec, v_play, f))
}

/// `--bw`: `auto` or a fixed loop bandwidth [Hz].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bw {
    Auto,
    Hz(f64),
}

/// `resolve(a)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub abs: Abs,
    pub v_rec: Option<f64>,
    pub v_play: Option<f64>,
    pub play: f64,
    /// Search centre `ref * play`.
    pub centre: f64,
    pub bw: Bw,
    /// Second line (`line:` arg or `--check-line`).
    pub line_hz: Option<f64>,
    /// Physics speed (`v_play or v_rec`).
    pub v_phys: Option<f64>,
    /// `on` with `--weak-rf`, else `--corr-limit`.
    pub corr_limit: String,
}

/// `resolve(a)`: L1 mode/arg, speeds, play factor, search centre, pass-1 bw, second line,
/// physics speed, bandwidth policy.
pub fn resolve(a: &RunArgs) -> Result<Resolved, ParamError> {
    let abs = parse_abs(&a.abs)?;
    let (v_rec, v_play, play) = play_factor(a)?;
    let bw = if a.bw == "auto" {
        Bw::Auto
    } else {
        Bw::Hz(parse_float(&a.bw).map_err(ParamError::Value)?)
    };
    let line_hz = match abs {
        Abs::Line(v) => Some(v),
        _ => a.check_line,
    };
    Ok(Resolved {
        v_rec,
        v_play,
        play,
        centre: a.r#ref * play,
        bw,
        line_hz,
        v_phys: v_play.filter(|&v| v != 0.0).or(v_rec),
        corr_limit: if a.weak_rf {
            "on".into()
        } else {
            a.corr_limit.clone()
        },
        abs,
    })
}

// ------------------------------------------------------------------ pathlib (posix)

/// `str(PurePosixPath(s))`: collapse `//`, drop `.` parts and trailing `/`; exactly two
/// leading slashes are kept; empty → `.`.
pub fn norm_path(s: &str) -> String {
    let root = if s.starts_with("//") && !s.starts_with("///") {
        "//"
    } else if s.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = s
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    let out = format!("{root}{}", parts.join("/"));
    if out.is_empty() { ".".into() } else { out }
}

/// `PurePosixPath(s).name`.
fn path_name(p: &str) -> &str {
    let root_only = p.chars().all(|c| c == '/');
    if root_only || p == "." {
        return "";
    }
    p.rsplit('/').next().unwrap_or("")
}

/// POSIX path stem (a name with only leading dots has no suffix).
pub fn path_stem(s: &str) -> String {
    let p = norm_path(s);
    let name = path_name(&p);
    match name.rfind('.') {
        Some(i) if !name[..i].trim_start_matches('.').is_empty() => name[..i].to_string(),
        _ => name.to_string(),
    }
}

/// `str(PurePosixPath(s).with_name(name))`.
pub fn with_file_name(s: &str, name: &str) -> Result<String, ParamError> {
    let p = norm_path(s);
    if path_name(&p).is_empty() {
        return Err(ParamError::Value(format!(
            "PurePosixPath({}) has an empty name",
            crate::absolute::quote_str(&p)
        )));
    }
    Ok(match p.rfind('/') {
        Some(i) => format!("{}{}", &p[..=i], name),
        None => name.to_string(),
    })
}

/// Files a run writes (`out_paths(a)`); paths as normalised POSIX text.
#[derive(Debug, Clone, PartialEq)]
pub struct OutPaths {
    pub audio: String,
    pub curve: String,
    pub report: String,
    /// None with `--no-plot`.
    pub plot: Option<String>,
    /// Only with `--plot` (and not `--no-plot`).
    pub model_plot: Option<String>,
}

/// `out_paths(a)`: audio, curve, report, speed plot, model plot.
pub fn out_paths(a: &RunArgs) -> Result<OutPaths, ParamError> {
    let out = &a.output;
    let stem = path_stem(out);
    let plot_arg = truthy(&a.plot);
    let pp = match plot_arg {
        Some(p) => norm_path(p),
        None => with_file_name(out, &format!("{stem}_speed.png"))?,
    };
    Ok(OutPaths {
        audio: norm_path(out),
        curve: match truthy(&a.curve) {
            Some(c) => norm_path(c),
            None => with_file_name(out, &format!("{stem}_curve.csv"))?,
        },
        report: match truthy(&a.report) {
            Some(r) => norm_path(r),
            None => with_file_name(out, &format!("{stem}_report.json"))?,
        },
        plot: (!a.no_plot).then(|| pp.clone()),
        model_plot: if plot_arg.is_some() && !a.no_plot {
            Some(with_file_name(
                &pp,
                &format!("{}_model.png", path_stem(&pp)),
            )?)
        } else {
            None
        },
    })
}

/// `--lowpass` given (and non-zero).
fn lowpass_on(a: &RunArgs) -> Option<f64> {
    a.lowpass.filter(|&v| v != 0.0)
}

/// `lowpass_stop(a)`: start of the lowpass stopband [Hz] (inf without --lowpass).
pub fn lowpass_stop(a: &RunArgs) -> f64 {
    match lowpass_on(a) {
        Some(lp) => lp + a.lowpass_width / 2.0,
        None => f64::INFINITY,
    }
}

/// `ref_notch(a, true_hz)` → (notch frequency or None, reason if None).
pub fn ref_notch(a: &RunArgs, true_hz: f64) -> (Option<f64>, String) {
    if a.keep_ref {
        return (None, "kept (--keep-ref)".into());
    }
    if true_hz >= lowpass_stop(a) {
        return (
            None,
            format!(
                "not needed: in lowpass stopband (>= {} Hz)",
                fmt_general(lowpass_stop(a), 6)
            ),
        );
    }
    (Some(true_hz), String::new())
}

/// `manual_notches(a)`: `--notch-at` list (spaces removed, empty items skipped).
pub fn manual_notches(a: &RunArgs) -> Result<Vec<f64>, ParamError> {
    let s = a.notch_at.clone().unwrap_or_default().replace(' ', "");
    s.split(',')
        .filter(|h| !h.is_empty())
        .map(|h| parse_float(h).map_err(ParamError::Value))
        .collect()
}

/// `notch_skip(a, hz, notch_hz)`: why an extra notch at `hz` is not applied (None = applied).
pub fn notch_skip(a: &RunArgs, hz: f64, notch_hz: Option<f64>) -> Option<&'static str> {
    if hz >= lowpass_stop(a) {
        return Some("lowpass");
    }
    if let Some(n) = notch_hz.filter(|&n| n != 0.0)
        && (hz - n).abs() < a.notch_width / 2.0
    {
        return Some("reference notch");
    }
    None
}
