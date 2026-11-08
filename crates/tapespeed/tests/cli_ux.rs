//! UX / safety behaviour: inputs are never overwritten, survey `cmd:` and PLAN,
//! lossy output defaults, batch `-o` rule, survey classes (AM beat outside --search, FM pilot
//! without shared wow), `--channel` on a mono file.
mod common;
use common::*;
use std::path::Path;

fn synth(d: &Path, name: &str, extra: &[&str]) -> String {
    let p = d.join(name);
    let mut v = vec![
        "synth",
        "--preset",
        "am9",
        "--duration",
        "20",
        "--channels",
        "1",
        "-o",
        p.to_str().unwrap(),
    ];
    v.extend_from_slice(extra);
    let (st, _, err) = cli(&v);
    assert_eq!(st, 0, "{err}");
    p.to_string_lossy().into_owned()
}

/// Survey of `args[0]`; survey files go next to the input unless `-o` is given.
fn survey(args: &[&str]) -> String {
    let mut v = args.to_vec();
    let o = Path::new(args[0]).with_extension("sv.flac");
    let o = o.to_string_lossy().into_owned();
    if !args.contains(&"-o") {
        v.extend_from_slice(&["-o", &o]);
    }
    v.extend_from_slice(&["--survey", "--survey-blocks", "3", "--survey-block-s", "8"]);
    let (st, out, err) = cli(&v);
    assert_eq!(st, 0, "{v:?}: {err}");
    out
}

fn line<'a>(text: &'a str, start: &str) -> &'a str {
    text.lines()
        .map(str::trim_start)
        .find(|l| l.starts_with(start))
        .unwrap_or_else(|| panic!("no line {start:?} in\n{text}"))
}

#[test]
fn inputs_are_never_overwritten() {
    let d = scratch("cli_ux_guard");
    let a = synth(&d, "a.flac", &[]);
    let b = d.join("b.flac").to_string_lossy().into_owned();
    std::fs::copy(&a, &b).unwrap();
    std::os::unix::fs::symlink(&a, d.join("link.flac")).unwrap();
    std::fs::hard_link(&a, d.join("hard.flac")).unwrap();
    let orig = std::fs::read(&a).unwrap();
    let s = |p: &str| d.join(p).to_string_lossy().into_owned();
    let dot = format!("{}/./a.flac", d.display());
    let tpl = s("{stem}.flac");
    for args in [
        vec![a.as_str(), "--ref", "9000", "-o", a.as_str()],
        vec![a.as_str(), "--ref", "9000", "-o", &dot],
        vec![a.as_str(), "--ref", "9000", "-o", &s("link.flac")],
        vec![a.as_str(), "--ref", "9000", "-o", &s("hard.flac")],
        // batch: b's output would be b itself; checked before a runs
        vec![a.as_str(), b.as_str(), "--ref", "9000", "-o", &tpl],
        // side files: curve / plot onto an input
        vec![
            a.as_str(),
            "--ref",
            "9000",
            "-o",
            &s("x.flac"),
            "--curve",
            &a,
        ],
        vec![
            a.as_str(),
            "--ref",
            "9000",
            "-o",
            &s("x.flac"),
            "--plot",
            &a,
        ],
    ] {
        let (st, out, err) = cli(&args);
        assert_eq!(st, 2, "{args:?}: {out}{err}");
        assert!(
            err.starts_with("tapespeed: error: ") && err.contains("refusing to overwrite an input"),
            "{args:?}: {err}"
        );
        assert_eq!(std::fs::read(&a).unwrap(), orig, "{args:?}: input changed");
        assert!(!d.join("x.flac").exists() && !d.join("a_curve.csv").exists());
    }
    // a survey JSON onto an input (an audio file named like a survey output)
    let j = s("k_survey.json");
    std::fs::copy(&a, &j).unwrap();
    let (st, _, err) = cli(&[&j, "--survey", "-o", &s("k.flac")]);
    assert_eq!(st, 2, "{err}");
    // --abs curve:CSV is an input too
    let (st, _, _) = cli(&[&a, "--ref", "9000", "-o", &s("c.flac")]);
    assert_eq!(st, 0);
    let (st, _, err) = cli(&[
        &a,
        "--ref",
        "9000",
        "-o",
        &s("c.flac"),
        "--abs",
        &format!("curve:{}", s("c_curve.csv")),
    ]);
    assert_eq!(st, 2, "{err}");
    // --overwrite: allowed (the input is decoded before the output opens)
    let (st, out, err) = cli(&[&b, "--ref", "9000", "-o", &b, "--overwrite"]);
    assert_eq!(st, 0, "{err}");
    assert!(out.contains("audio "), "{out}");
    // existing outputs that are not inputs are overwritten, as before
    let (st, _, err) = cli(&[&a, "--ref", "9000", "-o", &s("c.flac")]);
    assert_eq!(st, 0, "{err}");
}

#[test]
fn survey_cmd_names_a_new_output_and_has_every_recommended_option() {
    let d = scratch("cli_ux_cmd");
    let wav = synth(&d, "w.wav", &[]);
    let flac = synth(&d, "f.flac", &[]);
    let mp3 = d.join("m.mp3").to_string_lossy().into_owned();
    let o = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i", &wav, "-b:a", "192k", &mp3])
        .status()
        .unwrap();
    assert!(o.success());
    for (inp, want) in [
        (&wav, "-o 'w_corrected.wav'"),
        (&flac, "-o 'f_corrected.flac'"),
        (&mp3, "-o 'm_corrected.flac'"),
    ] {
        let t = survey(&[inp.as_str(), "-o", &d.join("sv/x.flac").to_string_lossy()]);
        let cmd = line(&t, "cmd:");
        assert!(cmd.ends_with(want), "{cmd}");
        // am9: AM beat primary -> lowpass and auto-notch recommended, and in the cmd
        assert!(line(&t, "--lowpass ").contains("8000"), "{t}");
        assert!(line(&t, "--auto-notch ").contains("keyed"), "{t}");
        assert!(cmd.contains(" --lowpass --auto-notch -o "), "{cmd}");
        // no --ref: PLAN plans the cmd line
        assert!(
            line(&t, "PLAN (").starts_with("PLAN (recommended options = the cmd: line"),
            "{t}"
        );
        assert!(line(&t, "10 lowpass").contains("8000 Hz"), "{t}");
        assert!(line(&t, "11 notches").contains("--auto-notch keyed"), "{t}");
        assert!(!t.contains("survey recommends"), "{t}");
    }
    // --ref given: PLAN plans the typed options and names what is missing
    let t = survey(&[&wav, "--ref", "9000"]);
    assert!(line(&t, "PLAN (").starts_with("PLAN (typed options"), "{t}");
    assert!(line(&t, "10 lowpass").contains("off"), "{t}");
    assert!(
        t.contains("! survey recommends --lowpass (see RECOMMEND)"),
        "{t}"
    );
    assert!(
        t.contains("! survey recommends --auto-notch (see RECOMMEND)"),
        "{t}"
    );
}

#[test]
fn am_beat_outside_search_is_the_am_beat() {
    let d = scratch("cli_ux_search");
    // 9 kHz beat at -3.1 % (just outside the default +-3 %)
    let a = synth(&d, "s.wav", &["--offset", "-3.1"]);
    let t = survey(&[&a]);
    let c1 = line(&t, "1 ");
    assert!(c1.contains(" AM ") && c1.contains("9000"), "{c1}");
    assert!(line(&t, "--ref ").contains("needs --search 0.04"), "{t}");
    assert!(
        line(&t, "--search ").starts_with("--search      0.04"),
        "{t}"
    );
    assert!(
        line(&t, "cmd:").contains("--ref 9000 --search 0.04 "),
        "{}",
        line(&t, "cmd:")
    );
    assert!(line(&t, "3 analyze").contains("search +-4 %"), "{t}");
    assert!(!t.contains("near the --search limit"), "{t}");
    // typed --search 0.04: no recommendation needed, plain AM class
    let t = survey(&[&a, "--search", "0.04"]);
    assert!(!t.contains("  --search "), "{t}");
}

#[test]
fn fm_pilot_needs_the_tape_wow() {
    let d = scratch("cli_ux_fm");
    // AM recording plus a stationary (post-tape) line at 19.1 kHz: no FM pilot
    let a = synth(&d, "am.wav", &["--lines", "19100:-12:post"]);
    let t = survey(&[&a]);
    let c = t
        .lines()
        .find(|l| l.contains("19100.0"))
        .unwrap_or_else(|| panic!("{t}"));
    assert!(c.contains(" ? ") && c.contains("no FM signature"), "{c}");
    assert!(!line(&t, "cmd:").contains("--check-line 19100"), "{t}");
    // the synthetic FM recording keeps its pilot
    let p = d.join("fm.wav").to_string_lossy().into_owned();
    let (st, _, err) = cli(&["synth", "--preset", "fm19", "--duration", "20", "-o", &p]);
    assert_eq!(st, 0, "{err}");
    let t = survey(&[&p]);
    let c1 = line(&t, "1 ");
    assert!(c1.contains(" FM ") && c1.contains("19000"), "{c1}");
    assert!(line(&t, "cmd:").contains("--ref 19000 -o"), "{t}");
}

#[test]
fn several_inputs_need_a_directory_or_stem() {
    let d = scratch("cli_ux_batch");
    let a = synth(&d, "a.wav", &[]);
    let b = synth(&d, "b.wav", &[]);
    let all = d.join("all.flac");
    let (st, _, err) = cli(&[&a, &b, "--ref", "9000", "-o", all.to_str().unwrap()]);
    assert_eq!(st, 2, "{err}");
    assert!(
        err.contains("with several inputs -o is a directory or contains {stem}")
            && err.contains("/all/"),
        "{err}"
    );
    assert!(!all.exists(), "no directory all.flac");
    // a name without an audio extension is a directory
    let out = d.join("out");
    let (st, _, err) = cli(&[&a, &b, "--ref", "9000", "-o", out.to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    assert!(out.join("a.flac").is_file() && out.join("b.flac").is_file());
}

#[test]
fn lossy_output_gets_a_note_and_sane_settings() {
    let d = scratch("cli_ux_mp3");
    let a = synth(&d, "a.wav", &[]);
    let m = d.join("a.mp3").to_string_lossy().into_owned();
    let (st, out, err) = cli(&[&a, "--ref", "9000", "-o", &m]);
    assert_eq!(st, 0, "{err}");
    assert!(
        out.contains(&format!(
            "note: {m} is lossy (MP3, LAME VBR -q:a 2, ~190 kb/s); .flac or .wav give a lossless master"
        )),
        "{out}"
    );
    // VBR: the Xing/Info header of LAME's VBR mode
    let b = std::fs::read(&m).unwrap();
    assert!(b.windows(4).any(|w| w == b"Xing"), "no VBR header");
    let (_, out, _) = cli(&[
        &a,
        "--ref",
        "9000",
        "-o",
        &d.join("a.flac").to_string_lossy(),
    ]);
    assert!(!out.contains("note:"), "{out}");
}

#[test]
fn channel_on_a_mono_file_warns_and_uses_mid() {
    let d = scratch("cli_ux_mono");
    let a = synth(&d, "m.wav", &[]);
    let o = d.join("o.flac").to_string_lossy().into_owned();
    let (st, out, err) = cli(&[&a, "--ref", "9000", "--channel", "S", "-o", &o]);
    assert_eq!(st, 0, "{err}");
    assert!(
        out.contains(
            "warning: --channel S on a mono file: analysing the mono signal (as --channel mid)"
        ),
        "{out}"
    );
    let csv = std::fs::read_to_string(d.join("o_curve.csv")).unwrap();
    assert!(csv.contains("# channel=mid"), "{}", &csv[..400]);
    let t = survey(&[&a, "--ref", "9000", "--channel", "L"]);
    assert!(t.contains("! --channel L on a mono file"), "{t}");
}
