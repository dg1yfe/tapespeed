//! Subcommands with their own argument parsing:
//! - `tapespeed correct IN CURVE -o OUT [--notch] [--notch-q Q] [--taps N]` ;
//!   streams correct -> notch -> encoder, no output buffer;
//! - `tapespeed synth [--preset P] -o OUT [--truth CSV] [--<field> V ...]`;
//! - `tapespeed analyze IN --ref HZ -o CSV [--plot PNG] [--search S] [--bw B] [--fmax F]
//!   [--channel C]`.

use clap::Parser;
use ts_core::correct::{self, Params, Post};

/// tapespeed pass 2: correct speed from curve
#[derive(Parser, Debug)]
#[command(name = "tapespeed correct", infer_long_args = true)]
struct CorrectArgs {
    input: String,
    curve: String,
    #[arg(short = 'o', long = "output", required = true)]
    output: String,
    /// notch out ref_hz after correction
    #[arg(long)]
    notch: bool,
    #[arg(long = "notch-q", default_value_t = 30.0)]
    notch_q: f64,
    #[arg(long, default_value_t = 64)]
    taps: usize,
}

fn run_correct(argv: &[String]) -> Result<(), String> {
    let a = CorrectArgs::parse_from(argv.iter().skip(1));
    let audio = ts_io::load(&a.input, None, None).map_err(|e| e.to_string())?;
    let c = ts_core::curve::read_curve(&a.curve).map_err(|e| e.to_string())?;
    let mut post = Post::default();
    if a.notch {
        let hz = correct::default_notch_hz(&c).ok_or("KeyError: 'ref_hz'")?;
        post.notches.push((hz, a.notch_q));
    }
    let p = Params {
        taps: a.taps,
        ..Params::default()
    };
    let mut w = ts_io::Writer::create(&a.output, audio.sr, audio.channels, None)
        .map_err(|e| e.to_string())?;
    let r = correct::run(
        &audio.data,
        audio.channels,
        audio.sr,
        &c,
        &p,
        &post,
        |b: &[f32]| w.write(b),
    );
    match r {
        Ok(_) => w.finish().map_err(|e| e.to_string()),
        // a broken pipe means ffmpeg died: prefer its stderr
        Err(e) => Err(match w.finish() {
            Err(fe) => fe.to_string(),
            Ok(()) => e.to_string(),
        }),
    }
}

/// `tapespeed synth --help`: usage, options, fields with defaults.
pub fn synth_help() -> Result<String, String> {
    let p = ts_core::synth::make_params("custom", &[]).map_err(|e| e.to_string())?;
    let mut h = String::from(
        "usage: tapespeed synth [-h] [--preset P] -o OUT [--truth CSV] [--FIELD VALUE ...]\n\n\
         Synthetic test recording with a known speed curve: writes OUT (.wav f32, .flac 24 bit)\n\
         and the truth curve CSV.\n\noptions:\n\
         \x20 -h, --help         show this help message and exit\n",
    );
    h += &format!(
        "  --preset P         {{{},custom}}, applied before the field overrides [default: custom]\n",
        ts_core::synth::PRESET_NAMES.join(",")
    );
    h += "  -o, --out OUT      output audio (required)\n";
    h += "  --truth CSV        truth curve [default: <OUT without extension>_truth.csv]\n";
    h += "\nfields (--FIELD VALUE or --FIELD=VALUE; '-' and '_' are equivalent in names):\n";
    for (k, v) in p.meta() {
        if k != "source" && k != "preset" {
            h += &format!("  --{k:<16} [default: {}]\n", v.to_text());
        }
    }
    Ok(h)
}

/// `tapespeed synth`: `--preset`, `-o/--out`, `--truth` and one `--<field>` per
/// synth parameter (value typing in `synth::Params::set_cli`).
fn run_synth(argv: &[String]) -> Result<(), String> {
    let (mut preset, mut out, mut truth) = ("custom".to_string(), None, None);
    let mut kw: Vec<(String, String)> = vec![];
    let mut it = argv.iter().skip(2);
    while let Some(k) = it.next() {
        let (key, inline) = match k.split_once('=') {
            Some((a, b)) if k.starts_with("--") => (a.to_string(), Some(b.to_string())),
            _ => (k.clone(), None),
        };
        let mut val = || {
            inline
                .clone()
                .or_else(|| it.next().cloned())
                .ok_or(format!("argument {key}: expected one argument"))
        };
        match key.as_str() {
            "-h" | "--help" => {
                print!("{}", synth_help()?);
                return Ok(());
            }
            "--preset" => preset = val()?,
            "-o" | "--out" => out = Some(val()?),
            "--truth" => truth = Some(val()?),
            o if o.starts_with("--") => {
                let name = o[2..].replace('-', "_");
                let v = val()?;
                kw.retain(|(n, _)| *n != name);
                kw.push((name, v));
            }
            o => return Err(format!("unrecognized arguments: {o}")),
        }
    }
    let out = out.ok_or("the following arguments are required: -o/--out")?;
    ts_core::synth::cli(
        &preset,
        &kw,
        std::path::Path::new(&out),
        truth.as_deref().map(std::path::Path::new),
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// tapespeed pass 1: reference tone -> speed curve
#[derive(Parser, Debug)]
#[command(
    name = "tapespeed analyze",
    infer_long_args = true,
    allow_negative_numbers = true
)]
struct AnalyzeArgs {
    input: String,
    /// nominal reference frequency [Hz]
    #[arg(long = "ref", required = true)]
    r#ref: f64,
    /// max |speed offset| (fraction)
    #[arg(long, default_value_t = 0.05)]
    search: f64,
    /// PLL bandwidth [Hz] or 'auto' (default)
    #[arg(long, default_value = "auto")]
    bw: String,
    /// max flutter frequency in curve [Hz]
    #[arg(long, default_value_t = 25.0)]
    fmax: f64,
    #[arg(long, default_value = "mid", value_parser = ["mid", "L", "R", "S"])]
    channel: String,
    #[arg(short = 'o', long = "output", required = true)]
    output: String,
    #[arg(long)]
    plot: Option<String>,
}

fn run_analyze(argv: &[String]) -> Result<(), String> {
    use ts_core::analyze::api::cli_summary;
    use ts_core::analyze::{AnalyzeOpts, analyze};
    let a = AnalyzeArgs::parse_from(argv.iter().skip(1));
    let audio = ts_io::load(&a.input, None, None).map_err(|e| e.to_string())?;
    let o = AnalyzeOpts {
        search: a.search,
        bw: a.bw.parse().map_err(|e: ts_dsp::ValueError| e.0)?,
        channel: a.channel.parse().map_err(|e: ts_dsp::ValueError| e.0)?,
        fmax: a.fmax,
    };
    let mut meta = indexmap::IndexMap::new();
    meta.insert("source".to_string(), a.input.as_str().into());
    let c = analyze(
        &audio.data,
        audio.channels,
        audio.sr,
        a.r#ref,
        &o,
        Some(&meta),
    )
    .map_err(|e| e.0)?;
    ts_core::curve::write_curve(&a.output, &c).map_err(|e| e.to_string())?;
    if let Some(p) = &a.plot {
        // the model plot shows the analysed curve
        ts_plot::plot_model(&c, &c, None, std::path::Path::new(p)).map_err(|e| e.to_string())?;
    }
    println!("{}", cli_summary(&c));
    Ok(())
}

/// If `argv[1]` names a subcommand, run it and return its exit status.
pub fn dispatch(argv: &[String]) -> Option<i32> {
    match argv.get(1).map(String::as_str) {
        Some(name @ ("synth" | "analyze")) => Some(
            match if name == "synth" {
                run_synth(argv)
            } else {
                run_analyze(argv)
            } {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("tapespeed {name}: {e}");
                    1
                }
            },
        ),
        Some("correct") => Some(match run_correct(argv) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("tapespeed correct: {e}");
                1
            }
        }),
        _ => None,
    }
}
