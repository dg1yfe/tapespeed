//! Doc drift check: the option tables of docs/MANUAL.md (between `<!-- drift:NAME:begin -->` and
//! `<!-- drift:NAME:end -->`) against the CLI itself:
//! - `main`: option set = long options of `tapespeed --help` = the clap definition; default cell
//!   = clap default (`` `X` ``), `` off; bare: `X` `` for optional values, `off` for flags, `–`
//!   otherwise; type cell = `` `{a,b}` `` for options with possible values;
//! - `analyze`, `correct`: names and `[default: X]` of `tapespeed analyze|correct --help`;
//! - `synth`: fields and defaults in the truth-CSV header of `tapespeed synth -o x.wav` and in
//!   `tapespeed synth --help`;
//! - `--help` of the main command shows every clap default as `[default: X]`.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use clap::ArgAction;

const BIN: &str = env!("CARGO_BIN_EXE_tapespeed");
const NONE: &str = "–";

fn manual() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/MANUAL.md");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Option (or synth field) name -> table cells (first cell included), for one drift table.
fn table(doc: &str, name: &str) -> BTreeMap<String, Vec<String>> {
    let (b, e) = (
        format!("<!-- drift:{name}:begin -->"),
        format!("<!-- drift:{name}:end -->"),
    );
    let i = doc
        .find(&b)
        .unwrap_or_else(|| panic!("MANUAL.md: marker {b} missing"));
    let j = doc[i..]
        .find(&e)
        .unwrap_or_else(|| panic!("MANUAL.md: marker {e} missing"))
        + i;
    let mut out = BTreeMap::new();
    for line in doc[i + b.len()..j].lines().map(str::trim) {
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .replace("\\|", "\u{1}")
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().replace('\u{1}', "|"))
            .collect();
        // option rows start with a code span (header and separator rows do not)
        let Some(k) = cells[0].find("--").filter(|_| cells[0].starts_with('`')) else {
            continue;
        };
        let key: String = cells[0][k + 2..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        assert!(
            out.insert(key.clone(), cells).is_none(),
            "MANUAL.md drift:{name}: duplicate row --{key}"
        );
    }
    assert!(!out.is_empty(), "MANUAL.md drift:{name}: no rows");
    out
}

fn run(args: &[&str]) -> String {
    let o = Command::new(BIN)
        .args(args)
        .output()
        .expect("run tapespeed");
    assert!(o.status.success(), "tapespeed {args:?}: {:?}", o.status);
    String::from_utf8(o.stdout).unwrap()
}

/// Long options of a `--help` text -> `[default: X]` if shown (on the option line or in its
/// description below). Option lines are indented by at most 6 spaces; description lines (which
/// may start with `--`) by more.
fn help_options(text: &str) -> BTreeMap<String, Option<String>> {
    let mut out = BTreeMap::new();
    let mut cur: Option<String> = None;
    let def = |t: &str| {
        t.find("[default: ").map(|d| {
            let r = &t[d + 10..];
            r[..r.find(']').unwrap()].to_string()
        })
    };
    for line in text.lines() {
        let ind = line.len() - line.trim_start().len();
        let t = line.trim_start();
        if ind > 6 || !t.starts_with('-') {
            // description line of the current option
            if let (Some(c), Some(d)) = (&cur, def(t))
                && ind > 6
            {
                out.insert(c.clone(), Some(d));
            }
            if ind <= 6 {
                cur = None;
            }
            continue;
        }
        let Some(k) = t.find("--") else { continue };
        let name: String = t[k + 2..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        out.insert(name.clone(), def(t));
        cur = Some(name);
    }
    out
}

fn code(v: &str) -> String {
    format!("`{v}`")
}

#[test]
fn main_options_match_help_and_clap() {
    let doc = table(&manual(), "main");
    let help = help_options(&run(&["--help"]));
    let cmd = tapespeed::cli::command();
    let mut clap_names = vec![];
    for a in cmd.get_arguments().filter(|a| !a.is_positional()) {
        let name = a.get_long().expect("long option").to_string();
        clap_names.push(name.clone());
        let Some(row) = doc.get(&name) else { continue };
        let want = match a.get_action() {
            ArgAction::SetTrue => "off".to_string(),
            ArgAction::Help => NONE.to_string(),
            _ if !a.get_default_values().is_empty() => code(
                &a.get_default_values()
                    .iter()
                    .map(|v| v.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            _ => {
                // optional value: the value a bare `--name` gets
                let m = tapespeed::cli::command().try_get_matches_from([
                    "tapespeed",
                    "x",
                    &format!("--{name}"),
                ]);
                match m.ok().and_then(|m| {
                    m.get_raw(a.get_id().as_str())
                        .and_then(|mut v| v.next().map(|s| s.to_string_lossy().into_owned()))
                }) {
                    Some(v) => format!("off; bare: {}", code(&v)),
                    None => NONE.to_string(),
                }
            }
        };
        assert_eq!(row[2], want, "MANUAL.md §5.1 --{name}: default cell");
        // `--help` shows the clap default
        if !a.get_default_values().is_empty() {
            assert_eq!(
                help.get(&name).cloned().flatten().map(|v| code(&v)),
                Some(want.clone()),
                "`tapespeed --help` --{name}: [default: …]"
            );
        }
        if matches!(a.get_action(), ArgAction::Set | ArgAction::Append) {
            let pv: Vec<String> = a
                .get_possible_values()
                .iter()
                .map(|p| p.get_name().to_string())
                .collect();
            if !pv.is_empty() {
                assert_eq!(
                    row[1],
                    code(&format!("{{{}}}", pv.join(","))),
                    "MANUAL.md §5.1 --{name}: type cell"
                );
            }
        }
    }
    clap_names.sort();
    let doc_names: Vec<String> = doc.keys().cloned().collect();
    let help_names: Vec<String> = help.keys().cloned().collect();
    assert_eq!(
        doc_names, help_names,
        "MANUAL.md §5.1 options vs `tapespeed --help`"
    );
    assert_eq!(
        doc_names, clap_names,
        "MANUAL.md §5.1 options vs clap definition"
    );
}

fn check_sub(sub: &str, section: &str) {
    let doc = table(&manual(), sub);
    let help = help_options(&run(&[sub, "--help"]));
    let dn: Vec<&String> = doc.keys().collect();
    let hn: Vec<&String> = help.keys().collect();
    assert_eq!(
        dn, hn,
        "MANUAL.md {section} options vs `tapespeed {sub} --help`"
    );
    for (name, def) in &help {
        let want = def.as_deref().map_or(NONE.to_string(), code);
        assert_eq!(
            doc[name][2], want,
            "MANUAL.md {section} --{name}: default cell"
        );
    }
}

#[test]
fn analyze_options_match_help() {
    check_sub("analyze", "§7.2");
}

#[test]
fn correct_options_match_help() {
    check_sub("correct", "§7.3");
}

#[test]
fn synth_help_matches_table() {
    let doc = table(&manual(), "synth");
    let help = help_options(&run(&["synth", "--help"]));
    let mut want: BTreeMap<String, String> = help
        .into_iter()
        .filter(|(k, _)| !matches!(k.as_str(), "help" | "out" | "truth"))
        .map(|(k, v)| (k, v.unwrap_or_default()))
        .collect();
    // the field list repeats nothing; --preset is listed once among the options
    assert_eq!(want.remove("preset").as_deref(), Some("custom"));
    let got: BTreeMap<String, String> = doc
        .iter()
        .filter(|(k, _)| *k != "preset")
        .map(|(k, row)| {
            let v = row[1].trim_matches('`');
            (
                k.clone(),
                if v == "\"\"" {
                    String::new()
                } else {
                    v.to_string()
                },
            )
        })
        .collect();
    assert_eq!(
        got, want,
        "MANUAL.md §7.4 synth fields/defaults vs `tapespeed synth --help`"
    );
}

#[test]
fn synth_fields_match_truth_header() {
    let doc = table(&manual(), "synth");
    let dir = std::env::temp_dir().join(format!("ts-manual-drift-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("x.wav");
    run(&["synth", "-o", out.to_str().unwrap()]);
    let hdr = std::fs::read_to_string(dir.join("x_truth.csv")).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let mut want = BTreeMap::new();
    for l in hdr.lines().take_while(|l| l.starts_with('#')) {
        if let Some((k, v)) = l[1..].trim_start().split_once('=')
            && k != "source"
        {
            want.insert(k.to_string(), v.to_string());
        }
    }
    let got: BTreeMap<String, String> = doc
        .iter()
        .map(|(k, row)| {
            let v = row[1].trim_matches('`');
            (
                k.clone(),
                if v == "\"\"" {
                    String::new()
                } else {
                    v.to_string()
                },
            )
        })
        .collect();
    assert_eq!(
        got, want,
        "MANUAL.md §7.4 synth fields/defaults vs truth CSV header"
    );
}
