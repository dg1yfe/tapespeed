//! Command line: options, defaults and help text of the one-shot command.
use clap::{Arg, ArgAction, Command, CommandFactory, FromArgMatches, Parser};

/// Model-step constants (`MIN_EV`, `B_DEFAULT`, `TA_DEFAULT`).
pub const MIN_EV: f64 = 0.70;
pub const B_DEFAULT: f64 = 20.0;
pub const TA_DEFAULT: f64 = 5.0;

// Field names are the option destinations (long option name, `-` -> `_`).
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(
    name = "tapespeed",
    about = "One-shot: analyze (pass 1) -> model step -> evidence / bandwidth policy -> L1 -> correct."
)]
pub struct Cli {
    // input files; several = batch (AFTER_HELP); no help text
    #[arg(id = "input", value_name = "input", required = true, num_args = 1..)]
    pub inputs: Vec<String>,
    // the input of this run (`inputs[0]`, or one batch file)
    #[arg(skip)]
    pub input: String,

    #[arg(
        long = "max-ram",
        value_name = "SIZE",
        help = "memory target (planning budget, not a hard cap: the measured peak may exceed it somewhat): bytes with K/M/G/T suffix or N% of the available RAM (default 80 % of available; an explicit value may also raise it, capped at physical RAM / cgroup limit); beyond it the input spills to --tmp-dir and fewer bw candidates run in parallel"
    )]
    pub max_ram: Option<String>,

    #[arg(long = "threads", help = "worker threads (default: all cores)")]
    pub threads: Option<usize>,

    #[arg(
        long = "tmp-dir",
        value_name = "DIR",
        help = "spill files (default $TAPESPEED_TMPDIR, else the system temp dir); unlinked at creation, never left behind"
    )]
    pub tmp_dir: Option<String>,

    #[arg(
        long = "ref",
        id = "ref",
        value_name = "REF",
        help = "detection frequency (search centre) [Hz]; optional with --survey"
    )]
    pub r#ref: Option<f64>,

    #[arg(
        long = "output",
        short = 'o',
        help = "corrected audio (.wav f32, .flac 24 bit, ...); optional with --survey"
    )]
    pub output: Option<String>,

    #[arg(
        long = "overwrite",
        help = "allow an output path (audio, curve, report, plots, survey files) to be an input file; default: refuse (exit 2). Other existing outputs are always overwritten"
    )]
    pub overwrite: bool,

    #[arg(
        long = "survey",
        help = "no audio: report reference-line candidates (sampled blocks), the recommended options with a ready cmd: line, and the steps of that run (with --ref: of a run with the typed options); writes <stem>_survey.json / .png next to -o (or the input stem in the cwd)"
    )]
    pub survey: bool,

    #[arg(
        long = "survey-blocks",
        help = "survey: number of blocks spread over the file",
        default_value = "12"
    )]
    pub survey_blocks: i64,

    #[arg(
        long = "survey-block-s",
        help = "survey: block length [s]",
        default_value_t = 20.0
    )]
    pub survey_block_s: f64,

    #[arg(
        long = "curve",
        help = "curve CSV path (default: <output stem>_curve.csv)"
    )]
    pub curve: Option<String>,

    #[arg(
        long = "report",
        help = "report JSON (default: <output stem>_report.json)"
    )]
    pub report: Option<String>,

    #[arg(
        long = "plot",
        help = "speed plot path (default <output stem>_speed.png); model plot next to it (<plot stem>_model.png)"
    )]
    pub plot: Option<String>,

    #[arg(long = "no-plot", help = "skip the speed plot")]
    pub no_plot: bool,

    #[arg(
        long = "keep-ref",
        help = "leave the reference tone in (default: notch it out at its corrected frequency)"
    )]
    pub keep_ref: bool,

    #[arg(long = "lowpass", help = "AM: lowpass the output (default 8000 Hz if given without value) against neighbour-channel interference above the transmitted audio", value_name = "HZ", num_args = 0..=1, default_missing_value = "8000")]
    pub lowpass: Option<f64>,

    #[arg(
        long = "notch-at",
        help = "extra notches (output frequencies) over the whole file, e.g. 8021,3361"
    )]
    pub notch_at: Option<String>,

    #[arg(long = "auto-notch", help = "detect stationary narrowband lines in the corrected output and notch them only where present: keyed (default) = on/off keyed signals (CW) only, steady carriers kept; all = every line", value_parser = ["keyed", "all"], num_args = 0..=1, default_missing_value = "keyed")]
    pub auto_notch: Option<String>,

    #[arg(
        long = "auto-notch-min",
        help = "lowest frequency searched by --auto-notch [Hz]",
        default_value_t = 3000.0
    )]
    pub auto_notch_min: f64,

    #[arg(
        long = "notch-width",
        help = "-3 dB width of extra notches [Hz]",
        default_value_t = 150.0
    )]
    pub notch_width: f64,

    #[arg(
        long = "lowpass-width",
        help = "lowpass transition width [Hz]",
        default_value_t = 500.0
    )]
    pub lowpass_width: f64,

    #[arg(
        long = "notch-q",
        help = "notch Q (30: line -26..-52 dB, program ±1 kHz -1 dB)",
        default_value_t = 30.0
    )]
    pub notch_q: f64,

    #[arg(
        long = "search",
        help = "max |speed offset| (fraction)",
        default_value_t = 0.03
    )]
    pub search: f64,

    #[arg(
        long = "bw",
        help = "pass-1 loop bandwidth [Hz] or auto",
        default_value = "auto"
    )]
    pub bw: String,

    #[arg(long = "channel", value_parser = ["mid", "L", "R", "S"], default_value = "mid")]
    pub channel: String,

    #[arg(long = "taps", help = "resampler kernel length", default_value = "64")]
    pub taps: i64,

    #[arg(long = "model", help = "model step (see README)", value_parser = ["auto", "on", "off"], default_value = "auto")]
    pub model: String,

    #[arg(long = "model-min-ev", help = "min explained variance for model-aided tracking", default_value_t = MIN_EV)]
    pub model_min_ev: f64,

    #[arg(long = "model-band", help = "aided residual band [Hz]", default_value_t = B_DEFAULT)]
    pub model_band: f64,

    #[arg(long = "model-ta", help = "L3 amplitude knot spacing [s]", default_value_t = TA_DEFAULT)]
    pub model_ta: f64,

    #[arg(
        long = "abs",
        help = "L1 source: ref | none | assume:HZ | line:HZ | value:R | percent:P | curve:CSV | pair[:HZ] (AM: both adjacent-channel beats, midpoint = HZ x speed; HZ = channel spacing, default 9000)",
        default_value = "ref"
    )]
    pub abs: String,

    #[arg(
        long = "pair-dmax",
        help = "--abs pair: max |station offset from the raster| [Hz]",
        default_value_t = 300.0
    )]
    pub pair_dmax: f64,

    #[arg(
        long = "no-switch-repair",
        help = "--abs pair: keep pass-1 spans locked onto the other beat (default: detect and divide out)"
    )]
    pub no_switch_repair: bool,

    #[arg(
        long = "check-line",
        help = "second line [Hz] for coherence evidence only"
    )]
    pub check_line: Option<f64>,

    #[arg(
        long = "max-corr-hz",
        help = "correction band limit when the policy applies",
        default_value_t = 6.0
    )]
    pub max_corr_hz: f64,

    #[arg(long = "corr-limit", help = "bandwidth policy", value_parser = ["auto", "on", "off"], default_value = "auto")]
    pub corr_limit: String,

    #[arg(
        long = "weak-rf",
        help = "known weak reception (fading, co-channel interference): same as --corr-limit on"
    )]
    pub weak_rf: bool,

    #[arg(
        long = "tape-speed",
        help = "nominal (recording) tape speed, e.g. 19.05cm, 7.5ips"
    )]
    pub tape_speed: Option<String>,

    #[arg(
        long = "play-speed",
        help = "playback speed of the transfer if different, e.g. 19.05cm"
    )]
    pub play_speed: Option<String>,
}

/// The clap command supports `-h/--help` text, unique-prefix long options and negative numbers as values.
/// `--help` epilogue: batch mode and output naming.
pub const AFTER_HELP: &str = "\
BATCH: several inputs run in parallel within one memory budget (at most ceil(threads / 2.5) at once).
  -o with one input: the output file, as before; an existing directory (or one given with a
  trailing /, created if missing) gives DIR/<stem>.flac. Missing parent directories of any
  output path are created.
  -o with several inputs: a directory (existing, or given with a trailing / or without an audio
  extension; created if missing), outputs DIR/<input stem>.flac, or a template containing {stem}
  (e.g. -o out/{stem}.wav); -o all.flac is an error. Curve, report and plots follow each output
  (<stem>_curve.csv, <stem>_report.json, <stem>_speed.png); explicit --curve/--report/--plot need
  {stem} with several inputs. Input stems must be unique. Each file's lines print as one block when
  it finishes; the exit status is 1 if any file failed.
INPUTS ARE PROTECTED: an output path (audio, curve, report, plots, survey files) that is the same
  file as an input (also via symlinks, ./ or hard links; checked for all files before anything
  runs) is refused with exit status 2 unless --overwrite is given. Existing outputs are overwritten.
MEMORY: the 'memory:' line shows the plan (mode ram|spill, bw candidates in parallel, estimated
  peak, budget, jobs, threads). See docs/MEMORY.md.";

/// Per-file copies of `a` (input, output, curve, report, plot resolved; see [`AFTER_HELP`]).
pub fn expand(a: &Cli) -> Result<Vec<Cli>, String> {
    use std::path::Path;
    let multi = a.inputs.len() > 1;
    let stem = |f: &str| {
        Path::new(f)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| f.to_string())
    };
    let tpl = |v: &Option<String>, st: &str, what: &str| -> Result<Option<String>, String> {
        match v {
            Some(t) if t.contains("{stem}") => Ok(Some(t.replace("{stem}", st))),
            Some(_) if multi => Err(format!(
                "{what} with several inputs needs {{stem}} in the path"
            )),
            o => Ok(o.clone()),
        }
    };
    let mut out = vec![];
    let mut seen = std::collections::HashSet::new();
    for f in &a.inputs {
        let st = stem(f);
        let mut b = a.clone();
        b.input = f.clone();
        b.curve = tpl(&a.curve, &st, "--curve")?;
        b.report = tpl(&a.report, &st, "--report")?;
        b.plot = tpl(&a.plot, &st, "--plot")?;
        b.output = match &a.output {
            Some(o) if o.contains("{stem}") => Some(o.replace("{stem}", &st)),
            // a directory: several inputs, an existing directory, or a trailing `/` (created)
            Some(o) if multi || Path::new(o).is_dir() || o.ends_with('/') => {
                if multi
                    && (Path::new(o).is_file()
                        || (!Path::new(o).is_dir() && !o.ends_with('/') && audio_ext(o)))
                {
                    return Err(format!(
                        "-o {o}: with several inputs -o is a directory or contains {{stem}} \
                         (e.g. -o {d}/ or -o '{d}/{{stem}}.flac')",
                        d = Path::new(o).with_extension("").to_string_lossy()
                    ));
                }
                std::fs::create_dir_all(o).map_err(|e| format!("-o {o}: {e}"))?;
                Some(
                    Path::new(o)
                        .join(format!("{st}.flac"))
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            o => o.clone(),
        };
        let key = b.output.clone().unwrap_or_else(|| st.clone());
        if multi && !seen.insert(key) {
            return Err(format!(
                "{f}: output name collides with another input (same stem {st:?})"
            ));
        }
        out.push(b);
    }
    Ok(out)
}

/// Extensions that name an audio file (with several inputs, `-o x.<ext>` is not a directory).
const AUDIO_EXT: [&str; 20] = [
    "wav", "flac", "mp3", "ogg", "oga", "opus", "m4a", "m4b", "aac", "mp4", "aif", "aiff", "aifc",
    "wma", "ape", "wv", "caf", "mka", "w64", "alac",
];

fn audio_ext(p: &str) -> bool {
    std::path::Path::new(p)
        .extension()
        .is_some_and(|e| AUDIO_EXT.contains(&e.to_string_lossy().to_lowercase().as_str()))
}

/// Identity of a path for the input guard: the canonical path (an existing file, else the
/// canonical parent + file name) and, if it exists, (device, inode) for hard links.
fn ident(p: &str) -> (Option<std::path::PathBuf>, Option<(u64, u64)>) {
    use std::path::Path;
    let path = Path::new(p);
    #[cfg(unix)]
    let ino = {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(path).ok().map(|m| (m.dev(), m.ino()))
    };
    #[cfg(not(unix))]
    let ino = None;
    let canon = path.canonicalize().ok().or_else(|| {
        let name = path.file_name()?;
        let parent = match path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d.canonicalize().ok()?,
            _ => std::env::current_dir().ok()?,
        };
        Some(parent.join(name))
    });
    (canon, ino)
}

/// Refuse output paths that are the same file as any input (all files of a batch, and
/// `--abs curve:CSV`), before anything runs. `outs`: per expanded file, its output paths.
/// `--overwrite` disables the check.
pub fn protect_inputs(
    top: &Cli,
    outs: &[(String, Vec<(&'static str, String)>)],
) -> Result<(), String> {
    if top.overwrite {
        return Ok(());
    }
    let mut ins: Vec<String> = top.inputs.clone();
    if let Some(c) = top.abs.strip_prefix("curve:") {
        ins.push(c.to_string());
    }
    let ins: Vec<_> = ins.iter().map(|p| (p, ident(p))).collect();
    for (file, list) in outs {
        for (what, o) in list {
            let (oc, oi) = ident(o);
            for (ip, (ic, ii)) in &ins {
                let same = (oc.is_some() && oc == *ic) || (oi.is_some() && oi == *ii);
                if same {
                    return Err(format!(
                        "{file}: {what} output {o} is the input file {ip}: refusing to overwrite \
                         an input (choose another -o, e.g. <name>_corrected.flac, or pass --overwrite)"
                    ));
                }
            }
        }
    }
    Ok(())
}

pub fn command() -> Command {
    Cli::command()
        .after_help(AFTER_HELP)
        .disable_help_flag(true)
        .disable_version_flag(true)
        .version(env!("CARGO_PKG_VERSION"))
        .infer_long_args(true)
        .allow_negative_numbers(true)
        .arg(
            Arg::new("help")
                .short('h')
                .long("help")
                .action(ArgAction::Help)
                .help("show this help message and exit"),
        )
        .arg(
            Arg::new("version")
                .short('V')
                .long("version")
                .action(ArgAction::Version)
                .help("show the version and exit"),
        )
}

/// Parse like `parser.parse_args(argv)`: `argv[0]` is the program name; errors exit with status 2.
pub fn parse_from<I, T>(argv: I) -> Cli
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let m = command().get_matches_from(argv);
    let mut a = Cli::from_arg_matches(&m).unwrap_or_else(|e| e.exit());
    a.input = a.inputs[0].clone();
    a
}

/// Post-parse check of `main()`: `p.error(...)` -> message, caller exits 2.
pub fn check_required(a: &Cli) -> Result<(), String> {
    if !a.survey && (a.r#ref.is_none() || a.output.is_none()) {
        return Err("--ref and -o are required (except with --survey)".into());
    }
    Ok(())
}

/// The `usage:` line printed before `error:` messages.
pub fn usage() -> String {
    command().render_usage().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    type Row = (String, Option<String>, Option<String>);

    fn outs(args: &[&str]) -> Result<Vec<Row>, String> {
        let mut v = vec!["tapespeed"];
        v.extend_from_slice(args);
        let a = parse_from(v);
        Ok(expand(&a)?
            .into_iter()
            .map(|b| (b.input, b.output, b.curve))
            .collect())
    }

    #[test]
    fn output_naming() {
        let d = std::env::temp_dir().join(format!("ts-expand-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let ds = d.to_string_lossy().into_owned();
        // one input: -o as given
        assert_eq!(
            outs(&["a/x.mp3", "-o", "y.wav"]).unwrap(),
            [("a/x.mp3".into(), Some("y.wav".into()), None)]
        );
        // several: a directory (created), <stem>.flac
        let r = outs(&["a/x.mp3", "b/y.wav", "-o", &ds]).unwrap();
        assert_eq!(r[1].1.as_deref(), Some(d.join("y.flac").to_str().unwrap()));
        assert!(d.is_dir());
        // one input into an existing directory
        assert_eq!(
            outs(&["a/x.mp3", "-o", &ds]).unwrap()[0].1.as_deref(),
            d.join("x.flac").to_str()
        );
        // template, also for the curve
        let r = outs(&[
            "x.mp3",
            "y.mp3",
            "-o",
            "o/{stem}.wav",
            "--curve",
            "c/{stem}.csv",
        ])
        .unwrap();
        assert_eq!(r[0].1.as_deref(), Some("o/x.wav"));
        assert_eq!(r[1].2.as_deref(), Some("c/y.csv"));
        // errors: explicit --curve without {stem}, colliding stems, -o a file
        assert!(outs(&["x.mp3", "y.mp3", "-o", &ds, "--curve", "c.csv"]).is_err());
        assert!(outs(&["a/x.mp3", "b/x.wav", "-o", &ds]).is_err());
        std::fs::write(d.join("f.flac"), b"").unwrap();
        assert!(outs(&["x.mp3", "y.mp3", "-o", d.join("f.flac").to_str().unwrap()]).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
