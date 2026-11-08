//! The clap command must match the stored CLI definition (tests/data/parser.json): option
//! strings, help texts, choices, required flags, value counts and defaults.
#[path = "../../testdata.rs"]
mod testdata;
use serde_json::Value;
use tapespeed::cli;

fn fixture() -> Value {
    let p = testdata::dir("parser.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

/// Help texts that intentionally differ from the stored CLI definition: dest -> text.
const HELP_FIXED: [(&str, &str); 2] = [
    (
        "plot",
        "speed plot path (default <output stem>_speed.png); model plot next to it (<plot stem>_model.png)",
    ),
    // the PLAN section plans the recommended options without --ref
    (
        "survey",
        "no audio: report reference-line candidates (sampled blocks), the recommended options with a ready cmd: line, and the steps of that run (with --ref: of a run with the typed options); writes <stem>_survey.json / .png next to -o (or the input stem in the cwd)",
    ),
];

#[test]
fn option_set_defaults_help_match_reference() {
    let reference = fixture();
    let mut cmd = cli::command();
    cmd.build();
    let mut seen = 0;
    for a in reference["actions"].as_array().unwrap() {
        let dest = a["dest"].as_str().unwrap();
        if dest == "help" {
            let h = cmd
                .get_arguments()
                .find(|x| x.get_id() == "help")
                .expect("help");
            assert_eq!(h.get_short(), Some('h'));
            assert_eq!(h.get_long(), Some("help"));
            assert_eq!(
                h.get_help().unwrap().to_string(),
                a["help"].as_str().unwrap()
            );
            seen += 1;
            continue;
        }
        let arg = cmd
            .get_arguments()
            .find(|x| x.get_id() == dest)
            .unwrap_or_else(|| panic!("missing option {dest}"));
        seen += 1;
        // option strings
        let mut want: Vec<String> = a["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let mut got: Vec<String> = arg
            .get_long()
            .map(|l| format!("--{l}"))
            .into_iter()
            .chain(arg.get_short().map(|c| format!("-{c}")))
            .collect();
        want.sort();
        got.sort();
        assert_eq!(got, want, "{dest}: option strings");
        // help text (with the intentional differences of HELP_FIXED)
        match HELP_FIXED
            .iter()
            .find(|(d, _)| *d == dest)
            .map(|(_, h)| *h)
            .or(a["help"].as_str())
        {
            Some(h) => assert_eq!(
                arg.get_help().map(|s| s.to_string()).as_deref(),
                Some(h),
                "{dest}: help"
            ),
            None => assert!(arg.get_help().is_none(), "{dest}: unexpected help"),
        }
        // choices
        let choices: Vec<String> = arg
            .get_possible_values()
            .iter()
            .map(|p| p.get_name().to_string())
            .collect();
        match a["choices"].as_array() {
            Some(c) => assert_eq!(
                choices,
                c.iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>(),
                "{dest}: choices"
            ),
            None => assert!(
                choices.is_empty() || !arg.get_action().takes_values(),
                "{dest}: choices"
            ),
        }
        // required
        assert_eq!(
            arg.is_required_set(),
            a["required"].as_bool().unwrap(),
            "{dest}: required"
        );
        // nargs / kind
        let kind = a["kind"].as_str().unwrap();
        let nargs = &a["nargs"];
        if kind == "StoreTrueAction" {
            assert!(!arg.get_action().takes_values(), "{dest}: flag");
            assert_eq!(a["default"], Value::Bool(false));
            continue;
        }
        let n = arg.get_num_args().unwrap();
        if dest == "input" {
            // several inputs = batch
            assert_eq!(
                (n.min_values(), n.max_values()),
                (1, usize::MAX),
                "input: nargs"
            );
        } else if nargs == "?" {
            assert_eq!((n.min_values(), n.max_values()), (0, 1), "{dest}: nargs ?");
            // clap has no getter for default_missing_value: observe it by passing the bare option
            let opt = a["options"][0].as_str().unwrap();
            let m = cmd
                .clone()
                .try_get_matches_from(["tapespeed", "x", opt])
                .unwrap();
            let got = m
                .get_raw(dest)
                .expect("bare value")
                .next()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            match &a["const"] {
                Value::Number(c) => assert_eq!(
                    got.parse::<f64>().unwrap(),
                    c.as_f64().unwrap(),
                    "{dest}: const"
                ),
                Value::String(c) => assert_eq!(&got, c, "{dest}: const"),
                other => panic!("{dest}: const {other}"),
            }
        } else {
            assert_eq!((n.min_values(), n.max_values()), (1, 1), "{dest}: nargs");
        }
        // default
        let defs: Vec<String> = arg
            .get_default_values()
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        match &a["default"] {
            Value::Null => assert!(defs.is_empty(), "{dest}: default {defs:?} vs None"),
            Value::Number(d) => {
                assert_eq!(defs.len(), 1, "{dest}: default missing");
                assert_eq!(
                    defs[0].parse::<f64>().unwrap(),
                    d.as_f64().unwrap(),
                    "{dest}: default"
                );
            }
            Value::String(d) => assert_eq!(defs, vec![d.clone()], "{dest}: default"),
            other => panic!("{dest}: default {other}"),
        }
    }
    // no extra options beyond the stored definition except the memory/batch ones and --overwrite
    const EXTRA: [&str; 5] = ["max_ram", "threads", "tmp_dir", "overwrite", "version"];
    assert!(
        EXTRA
            .iter()
            .all(|r| cmd.get_arguments().any(|x| x.get_id() == *r))
    );
    assert_eq!(
        cmd.get_arguments().count(),
        seen + EXTRA.len(),
        "option count (unexpected extra arguments)"
    );
}

fn parse(args: &[&str]) -> cli::Cli {
    let mut v = vec!["tapespeed"];
    v.extend_from_slice(args);
    cli::command()
        .try_get_matches_from(v.clone())
        .map(|m| <cli::Cli as clap::FromArgMatches>::from_arg_matches(&m).unwrap())
        .unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn optional_values_and_prefixes() {
    let a = parse(&["in.wav", "--lowpass", "--ref", "9000", "-o", "o.flac"]);
    assert_eq!(
        (a.lowpass, a.r#ref, a.auto_notch.as_deref()),
        (Some(8000.0), Some(9000.0), None)
    );
    assert_eq!(
        parse(&["in.wav", "--lowpass", "5000"]).lowpass,
        Some(5000.0)
    );
    assert_eq!(parse(&["in.wav", "--lowpass=6000"]).lowpass, Some(6000.0));
    assert_eq!(
        parse(&["in.wav", "--auto-notch"]).auto_notch.as_deref(),
        Some("keyed")
    );
    assert_eq!(
        parse(&["in.wav", "--auto-notch", "all"])
            .auto_notch
            .as_deref(),
        Some("all")
    );
    // unique-prefix long options
    let a = parse(&[
        "in.wav",
        "--sea",
        "0.05",
        "--no-p",
        "--keep",
        "--model-m",
        "0.5",
        "--chan",
        "S",
    ]);
    assert_eq!(
        (
            a.search,
            a.no_plot,
            a.keep_ref,
            a.model_min_ev,
            a.channel.as_str()
        ),
        (0.05, true, true, 0.5, "S")
    );
    // exact match beats prefix (--lowpass vs --lowpass-width)
    assert_eq!(
        parse(&["in.wav", "--lowpass", "--lowpass-width", "300"]).lowpass_width,
        300.0
    );
    // negative numbers are values
    assert_eq!(parse(&["in.wav", "--search", "-0.5"]).search, -0.5);
    // defaults
    let a = parse(&["in.wav"]);
    assert_eq!(
        (
            a.taps,
            a.notch_q,
            a.bw.as_str(),
            a.abs.as_str(),
            a.survey_blocks
        ),
        (64, 30.0, "auto", "ref", 12)
    );
}

#[test]
fn parse_errors() {
    let bad = |args: &[&str]| {
        let mut v = vec!["tapespeed"];
        v.extend_from_slice(args);
        cli::command().try_get_matches_from(v).is_err()
    };
    assert!(bad(&[]), "input required");
    assert!(bad(&["x", "--ambiguous-nonexistent"]));
    assert!(bad(&["x", "--model-m", "--model-"]), "ambiguous prefix");
    assert!(bad(&["x", "--channel", "Q"]));
    assert!(bad(&["x", "--auto-notch", "some"]));
    let a = parse(&["x"]);
    assert!(cli::check_required(&a).is_err());
    assert!(cli::check_required(&parse(&["x", "--survey"])).is_ok());
}
