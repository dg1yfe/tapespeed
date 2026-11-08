//! CLI behaviour: `synth --help`, missing output
//! directories, the survey `cmd:` string, several survey inputs = the single runs in order.
mod common;
use common::*;
use std::path::Path;

/// A 20 s am9 test signal in `d` (synth, deterministic).
fn signal(d: &Path, name: &str) -> String {
    let p = d.join(name);
    if !p.exists() {
        let (st, _, err) = cli(&[
            "synth",
            "--preset",
            "am9",
            "--duration",
            "20",
            "--channels",
            "1",
            "-o",
            p.to_str().unwrap(),
        ]);
        assert_eq!(st, 0, "{err}");
    }
    p.to_string_lossy().into_owned()
}

#[test]
fn synth_help_exits_0() {
    let (st, out, err) = cli(&["synth", "--help"]);
    assert_eq!(st, 0, "{err}");
    assert!(out.starts_with("usage: tapespeed synth"), "{out}");
    assert!(
        out.contains("--ref_hz           [default: 9000.0]"),
        "{out}"
    );
    let (st, out2, _) = cli(&["synth", "-h"]);
    assert_eq!((st, &out2), (0, &out));
}

#[test]
fn missing_output_directories_are_created() {
    let d = scratch("cli_fixes_dirs");
    let _ = std::fs::remove_dir_all(d.join("o"));
    let a = signal(&d, "a.wav");
    let s = |p: &str| d.join(p).to_string_lossy().into_owned();
    // template, nested path, trailing slash (one input), curve/plot elsewhere
    for (args, want) in [
        (
            vec!["-o", &s("o/t/{stem}.wav")],
            vec!["o/t/a.wav", "o/t/a_curve.csv"],
        ),
        (
            vec!["-o", &s("o/n/x.wav"), "--plot", &s("o/p/x.png")],
            vec![
                "o/n/x.wav",
                "o/n/x_report.json",
                "o/p/x.png",
                "o/p/x_model.png",
            ],
        ),
        (
            vec!["-o", &(s("o/dir") + "/")],
            vec!["o/dir/a.flac", "o/dir/a_speed.png"],
        ),
    ] {
        let mut v = vec![a.as_str(), "--ref", "9000"];
        v.extend(args.iter().map(|x| &x[..]));
        let (st, _, err) = cli(&v);
        assert_eq!(st, 0, "{v:?}: {err}");
        for w in want {
            assert!(d.join(w).is_file(), "{v:?}: {w} missing");
        }
    }
    // survey outputs
    let (st, _, err) = cli(&[&a, "--survey", "-o", &s("o/sv/x.flac")]);
    assert_eq!(st, 0, "{err}");
    assert!(d.join("o/sv/x_survey.json").is_file());
    assert!(d.join("o/sv/x_survey.png").is_file());
}

/// Survey stdout under the fixed clock, without the machine-dependent `memory:` row and step 13.
fn survey_text(args: &[&str]) -> String {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_tapespeed"))
        .args(args)
        .env("TAPESPEED_FIXED_CLOCK", "1")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.trim_start().starts_with("memory:")) // available RAM varies between runs
        .map(|l| format!("{l}\n"))
        .collect()
}

#[test]
fn survey_cmd_and_several_inputs() {
    let d = scratch("cli_fixes_survey");
    let (a, b) = (signal(&d, "a.wav"), signal(&d, "b.wav"));
    let o = d.join("sv").to_string_lossy().into_owned() + "/";
    let both = survey_text(&[&a, &b, "--survey", "-o", &o]);
    let one = survey_text(&[&a, "--survey", "-o", &o]) + &survey_text(&[&b, "--survey", "-o", &o]);
    assert_eq!(both, one, "two inputs = the two single runs in order");
    let cmd: Vec<&str> = both
        .lines()
        .filter(|l| l.trim_start().starts_with("cmd:"))
        .collect();
    assert_eq!(cmd.len(), 2);
    assert!(
        cmd[0]
            .trim_start()
            .starts_with(&format!("cmd: tapespeed '{a}' --ref 9000")),
        "{}",
        cmd[0]
    );
    assert!(
        cmd.iter()
            .all(|l| l.trim_start().starts_with("cmd: tapespeed ")),
        "{cmd:?}"
    );
    let rt: Vec<&str> = both.lines().filter(|l| l.contains("13 runtime")).collect();
    assert!(
        rt[0].contains("min/h measured, M1 Max 10 threads; one thread ~")
            && rt[0].contains("GiB (planner peak estimate"),
        "{}",
        rt[0]
    );
}
