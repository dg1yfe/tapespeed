//! `--survey`: survey, recommend, plan, report text, plot, JSON.
use std::collections::BTreeMap;
use std::path::Path;

use ts_core::survey::{self, AnalyzeTracker, plan::SurveyArgs, report};
use ts_plot::{SurveyCandidate, SurveyClass, SurveyDet, SurveyPlotInput};

use crate::cli::Cli;
use crate::pipeline::Fail;

/// `Cli` → the fields `--survey` reads.
pub fn survey_args(a: &Cli) -> SurveyArgs {
    SurveyArgs {
        input: a.input.clone(),
        output: a.output.clone(),
        r#ref: a.r#ref,
        survey_blocks: a.survey_blocks.max(0) as usize,
        survey_block_s: a.survey_block_s,
        curve: a.curve.clone(),
        report: a.report.clone(),
        plot: a.plot.clone(),
        no_plot: a.no_plot,
        keep_ref: a.keep_ref,
        lowpass: a.lowpass,
        notch_at: a.notch_at.clone(),
        auto_notch: a.auto_notch.clone(),
        auto_notch_min: a.auto_notch_min,
        notch_width: a.notch_width,
        lowpass_width: a.lowpass_width,
        notch_q: a.notch_q,
        search: a.search,
        bw: a.bw.clone(),
        channel: a.channel.clone(),
        taps: a.taps,
        model: a.model.clone(),
        model_min_ev: a.model_min_ev,
        model_band: a.model_band,
        model_ta: a.model_ta,
        abs: a.abs.clone(),
        pair_dmax: a.pair_dmax,
        check_line: a.check_line,
        max_corr_hz: a.max_corr_hz,
        corr_limit: a.corr_limit.clone(),
        weak_rf: a.weak_rf,
        tape_speed: a.tape_speed.clone(),
        play_speed: a.play_speed.clone(),
        mem_peak: None,
    }
}

fn class(s: &str) -> SurveyClass {
    match s {
        "FM pilot" => SurveyClass::FmPilot,
        "AM beat" => SurveyClass::AmBeat,
        "TV line" => SurveyClass::TvLine,
        "codec" => SurveyClass::Codec,
        "hum" => SurveyClass::Hum,
        _ => SurveyClass::Unknown,
    }
}

/// `report::plot_data` → ts-plot's input (one-to-one).
pub fn plot_input(d: report::PlotData) -> SurveyPlotInput {
    SurveyPlotInput {
        file: d.file,
        f: d.f,
        d: d.d,
        d_blocks: d.d_blocks,
        prom: d.prom,
        blocks: d.blocks,
        candidates: d
            .candidates
            .into_iter()
            .map(|c| SurveyCandidate {
                id: c.id,
                f: c.f,
                cls: class(&c.cls),
                low: c.low,
                primary: c.primary,
                ch: c.ch,
                blocks: c
                    .blocks
                    .into_iter()
                    .map(|(k, f, cn0)| (k, SurveyDet { f, cn0 }))
                    .collect::<BTreeMap<_, _>>(),
            })
            .collect(),
        step: d.step,
        lock_db: d.lock_db,
        margin_db: d.margin_db,
    }
}

/// `survey.main(a)` for each input: print the report, write `<stem>_survey.png` / `.json`,
/// print the last line. The PLAN gets the memory planner's dry-run line for a run with these
/// options. Several inputs run one after another.
pub fn main_survey(a: &Cli) -> Result<(), Fail> {
    let files = crate::cli::expand(a).map_err(Fail::Usage)?;
    let outs: Vec<_> = files
        .iter()
        .filter_map(|b| {
            let (j, p) = survey::survey_paths(&b.input, b.output.as_deref()).ok()?;
            Some((b.input.clone(), vec![("survey json", j), ("survey png", p)]))
        })
        .collect();
    crate::cli::protect_inputs(a, &outs).map_err(Fail::Usage)?;
    let one = |f: &Cli| -> Result<String, Fail> {
        // dry run of a run with these options (no --ref yet: estimated at 9 kHz)
        let mut g = f.clone();
        let assumed = g.r#ref.is_none();
        g.r#ref.get_or_insert(9000.0);
        let (mem, peak) = match crate::pipeline::plan_files(std::slice::from_ref(&g), a) {
            Ok((ctx, jobs)) => match &jobs[0].plan {
                Ok(p) => (
                    ctx.line(p) + if assumed { " (for --ref 9000)" } else { "" },
                    Some(p.peak),
                ),
                Err(e) => (format!("memory: no plan ({})", e.message()), None),
            },
            Err(e) => (format!("memory: no plan ({})", e.message()), None),
        };
        survey_one(f, &mem, peak)
    };
    // sequential on purpose: each survey already uses the pool (~5x CPU/wall); 17
    // files with ceil(threads/2.5) = 4 surveys at once took 11.6-14.6 s vs 10.9-11.2 s one after
    // another, at 30-70 % more CPU (measured)
    for f in &files {
        print!("{}", one(f)?);
    }
    Ok(())
}

/// Insert `line` as the last row of the PLAN section of the survey text.
pub fn with_plan_line(text: &str, line: &str) -> String {
    match text.find("\nPLAN").map(|i| i + 1) {
        Some(i) => {
            let end = text[i..].find("\n\n").map_or(text.len(), |e| i + e);
            format!("{}\n  {line}{}", &text[..end], &text[end..])
        }
        None => format!("{text}\n{line}"),
    }
}

/// One survey: the stdout text (report with the plan line, final line) after writing the files.
fn survey_one(a: &Cli, mem: &str, peak: Option<u64>) -> Result<String, Fail> {
    let fixed = std::env::var_os("TAPESPEED_FIXED_CLOCK").map(|_| 0.0);
    let mut sa = survey_args(a);
    sa.mem_peak = peak;
    let sv = survey::run(&sa, &AnalyzeTracker, fixed)
        .map_err(|e| Fail::Error("ValueError".into(), e.to_string()))?;
    crate::pipeline::make_parents([&sv.png_path, &sv.json_path])?;
    let text = format!("{}\n", with_plan_line(&sv.text, mem));
    ts_plot::plot_survey(
        &plot_input(report::plot_data(&sv.survey)),
        Path::new(&sv.png_path),
    )
    .map_err(|e| Fail::Error("RuntimeError".into(), e.to_string()))?;
    std::fs::write(&sv.json_path, &sv.json)
        .map_err(|e| Fail::Error("OSError".into(), e.to_string()))?;
    Ok(format!("{text}{}\n", sv.final_line()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn plan_line_goes_into_plan() {
        let t = "A\n\nPLAN (x)\n   1 decode  y\n  13 runtime z\n\nWARNINGS\n  !";
        assert_eq!(
            super::with_plan_line(t, "memory: m"),
            "A\n\nPLAN (x)\n   1 decode  y\n  13 runtime z\n  memory: m\n\nWARNINGS\n  !"
        );
        assert_eq!(super::with_plan_line("A", "memory: m"), "A\nmemory: m");
    }
}
