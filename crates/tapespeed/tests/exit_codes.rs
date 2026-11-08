//! Exit codes and messages of the CLI.
mod common;
use common::*;

/// Exit codes and messages: usage error (2), run error (1).
#[test]
fn exit_codes() {
    let (st, _, err) = cli(&["in.wav"]);
    assert_eq!(st, 2);
    assert!(
        err.contains("--ref and -o are required (except with --survey)"),
        "{err}"
    );
    let d = scratch("exit_codes");
    let x = vec![0.0f32; 48000];
    let src = d.join("z.wav");
    ts_io::save(&src, &x, 1, 48000, None).unwrap();
    let o = d.join("o.wav");
    let (st, _, err) = cli(&[
        src.to_str().unwrap(),
        "--ref",
        "9000",
        "--play-speed",
        "19.05cm",
        "-o",
        o.to_str().unwrap(),
    ]);
    assert_eq!(
        (st, err.trim()),
        (1, "--play-speed needs --tape-speed (the recording speed)")
    );
}
