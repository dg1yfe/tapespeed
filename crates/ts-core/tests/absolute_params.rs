//! Against stored reference results (tests/data/h2): absolute (parse_abs, apply_l1 all
//! modes, pair_measure, pair_table, pair_info, ghost_rho, switch_repair), params (resolve,
//! out_paths, notches) and physics (parse_speed, classify, geometry, plausibility, infer,
//! constrain).
//!
//! Acceptance: abs_l1 <= 1e-8; candidates / verdicts / rejection reasons / ambiguity equal;
//! repaired spans equal; params and physics equal (floats 1e-9 relative, strings exact).
#[path = "../../testdata.rs"]
mod testdata;
use std::path::PathBuf;

use serde_json::Value as J;
use ts_core::absolute::{
    self as ab, Abs, Candidate, PairParams, PairResult, PairSolution, Side, XCheck,
};
use ts_core::curve::{Curve, MetaValue};
use ts_core::hetdec::Channel;
use ts_core::model::law::{Comp, Kind};
use ts_core::params::{self as pr, RunArgs};
use ts_core::physics as ph;
use ts_io::json::Value as V;

// ------------------------------------------------------------------ data helpers

fn dir() -> PathBuf {
    testdata::dir("h2")
}

/// serde_json without `float_roundtrip` may parse floats 1 ulp off: quote every float token
/// (outside strings) so `jf` parses it with Rust's correctly rounded `str::parse`.
fn quote_floats(s: &str) -> String {
    let b = s.as_bytes();
    let mut o = String::with_capacity(s.len() + s.len() / 8);
    let (mut i, mut in_str) = (0, false);
    while i < b.len() {
        let c = b[i];
        if in_str {
            o.push(c as char);
            if c == b'\\' {
                o.push(b[i + 1] as char);
                i += 2;
                continue;
            }
            in_str = c != b'"';
            i += 1;
        } else if c == b'"' {
            in_str = true;
            o.push('"');
            i += 1;
        } else if c == b'-' || c.is_ascii_digit() {
            let j0 = i;
            while i < b.len() && matches!(b[i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                i += 1;
            }
            let tok = &s[j0..i];
            if tok.contains(['.', 'e', 'E']) {
                o.push('"');
                o.push_str(tok);
                o.push('"');
            } else {
                o.push_str(tok);
            }
        } else {
            o.push(c as char);
            i += 1;
        }
    }
    o
}

fn json(p: PathBuf) -> J {
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    serde_json::from_str(&quote_floats(&s)).unwrap()
}

/// Key paths of our value in insertion order (as listed in the reference `keys` records).
fn key_paths(v: &V, pre: &str, out: &mut Vec<String>) {
    match v {
        V::Obj(o) => {
            for (k, x) in o {
                let q = format!("{pre}.{k}");
                out.push(q.clone());
                key_paths(x, &q, out);
            }
        }
        V::Arr(a) => {
            for (i, x) in a.iter().enumerate() {
                key_paths(x, &format!("{pre}[{i}]"), out);
            }
        }
        _ => {}
    }
}

fn check_keys(what: &str, e: &J, parts: Vec<(&str, V)>) {
    let mut got = Vec::new();
    key_paths(
        &V::Obj(parts.into_iter().map(|(k, v)| (k.to_string(), v)).collect()),
        "",
        &mut got,
    );
    let exp: Vec<&str> = e
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(got, exp, "{what}: reference key order");
}

fn npy_f64(p: PathBuf) -> Vec<f64> {
    let f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let n = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    let d = n.dtype().descr();
    if d.contains("b1") {
        n.into_vec::<bool>()
            .unwrap()
            .into_iter()
            .map(|b| b as u8 as f64)
            .collect()
    } else if d.contains("f4") {
        n.into_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    } else {
        n.into_vec::<f64>().unwrap()
    }
}

fn npy_bool(p: PathBuf) -> Vec<bool> {
    npy_f64(p).into_iter().map(|v| v != 0.0).collect()
}

fn npy_f32(p: PathBuf) -> (Vec<usize>, Vec<f32>) {
    let f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let n = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    let shape = n.shape().iter().map(|&s| s as usize).collect();
    (shape, n.into_vec::<f32>().unwrap())
}

/// JSON float (quoted by `quote_floats`; non-finite as "nan" / "inf" / "-inf").
fn jf(v: &J) -> f64 {
    match v {
        J::String(s) => s.parse().unwrap_or_else(|_| panic!("float {s:?}")),
        _ => v.as_f64().unwrap_or_else(|| panic!("float {v}")),
    }
}

fn jof(v: &J) -> Option<f64> {
    if v.is_null() { None } else { Some(jf(v)) }
}

fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= atol + rtol * b.abs()
}

/// Recursive comparison of a reference JSON record with our value: strings / ints / bools / null
/// exact, floats within tolerance. Returns failures; tracks the largest relative float error.
struct Cmp {
    rtol: f64,
    atol: f64,
    fails: Vec<String>,
    worst: f64,
}

impl Cmp {
    fn new(rtol: f64, atol: f64) -> Self {
        Cmp {
            rtol,
            atol,
            fails: vec![],
            worst: 0.0,
        }
    }

    fn num(&mut self, path: &str, e: f64, g: f64) {
        if e.is_finite() && g.is_finite() && e != g {
            self.worst = self.worst.max((e - g).abs() / e.abs().max(1e-300));
        }
        if !close(g, e, self.rtol, self.atol) {
            self.fails.push(format!("{path}: ref {e:?} got {g:?}"));
        }
    }

    fn val(&mut self, path: &str, e: &J, g: &V) {
        match (e, g) {
            (J::Null, V::Null) => {}
            (J::Bool(a), V::Bool(b)) if a == b => {}
            (J::String(a), V::Str(b)) if a == b => {}
            (J::String(_), V::Float(b)) => self.num(path, jf(e), *b),
            (J::Number(a), V::Int(b)) if a.is_i64() => {
                if a.as_i64() != Some(*b) {
                    self.fails.push(format!("{path}: ref {a} got {b}"));
                }
            }
            (J::Number(a), V::Float(b)) if a.is_f64() => self.num(path, a.as_f64().unwrap(), *b),
            (J::Array(a), V::Arr(b)) => {
                if a.len() != b.len() {
                    self.fails
                        .push(format!("{path}: len ref {} got {}", a.len(), b.len()));
                    return;
                }
                for (i, (x, y)) in a.iter().zip(b).enumerate() {
                    self.val(&format!("{path}[{i}]"), x, y);
                }
            }
            (J::Object(a), V::Obj(b)) => {
                let ka: Vec<&String> = a.keys().collect();
                let kb: Vec<&String> = b.iter().map(|(k, _)| k).collect();
                if !same_keys(&ka, &kb) {
                    self.fails
                        .push(format!("{path}: keys ref {ka:?} got {kb:?}"));
                    return;
                }
                for (k, x) in a {
                    self.val(&format!("{path}.{k}"), x, g.get(k).unwrap());
                }
            }
            _ => self.fails.push(format!("{path}: ref {e} got {g:?}")),
        }
    }

    fn check(&self, what: &str) {
        assert!(
            self.fails.is_empty(),
            "{what}: {} mismatches:\n{}",
            self.fails.len(),
            self.fails
                .iter()
                .take(40)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// serde_json (no `preserve_order`) sorts object keys, so key sets are compared here; the
/// reference insertion order is checked separately (`keys` lists in the records).
fn same_keys(ka: &[&String], kb: &[&String]) -> bool {
    let mut a = ka.to_vec();
    let mut b = kb.to_vec();
    a.sort();
    b.sort();
    a == b
}

fn meta_to_v(m: &MetaValue) -> V {
    match m {
        MetaValue::Str(s) => V::Str(s.clone()),
        MetaValue::Int(i) => V::Int(*i),
        MetaValue::Float(f) => V::Float(*f),
        MetaValue::Bool(b) => V::Bool(*b),
    }
}

fn info_to_v(info: &ab::Info) -> V {
    V::Obj(
        info.iter()
            .map(|(k, v)| (k.clone(), meta_to_v(v)))
            .collect(),
    )
}

fn opt_v(v: Option<f64>) -> V {
    v.map(V::Float).unwrap_or(V::Null)
}

// ------------------------------------------------------------------ parse / params

#[test]
fn parse_abs_and_speed() {
    let d = json(dir().join("parse.json"));
    for r in d["abs"].as_array().unwrap() {
        let spec = r["spec"].as_str().unwrap();
        match ab::parse_abs(spec) {
            Ok(a) => {
                assert!(r.get("error").is_none(), "{spec}: reference raised {r}");
                assert_eq!(a.mode(), r["mode"].as_str().unwrap(), "{spec}");
                match &a {
                    Abs::Curve(p) => assert_eq!(p, r["arg"].as_str().unwrap()),
                    _ => {
                        let e = jof(&r["arg"]);
                        let g = a.arg();
                        assert!(
                            e.is_none() && g.is_none()
                                || close(g.unwrap(), e.unwrap(), 0.0, 0.0)
                                    && g.unwrap().is_sign_negative()
                                        == e.unwrap().is_sign_negative(),
                            "{spec}: ref {e:?} got {g:?}"
                        );
                    }
                }
            }
            Err(e) => assert_eq!(e.to_string(), r["msg"].as_str().unwrap_or("<ok>"), "{spec}"),
        }
    }
    for r in d["speed"].as_array().unwrap() {
        let s = r["s"].as_str().unwrap();
        match ph::parse_speed(s) {
            Ok(v) => {
                assert_eq!(v, jf(&r["v"]), "{s}");
                assert_eq!(ph::fmt_speed(v), r["fmt"].as_str().unwrap());
            }
            Err(e) => assert_eq!(e.to_string(), r["msg"].as_str().unwrap_or("<ok>"), "{s}"),
        }
    }
}

/// Minimal option parser for the option set used in params.json (prefixes not needed).
fn run_args(argv: &[String]) -> RunArgs {
    let mut a = RunArgs::default();
    let mut i = 1; // argv[0] = input
    let take = |i: &mut usize| -> String {
        *i += 1;
        argv[*i].clone()
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--ref" => a.r#ref = take(&mut i).parse().unwrap(),
            "-o" => a.output = take(&mut i),
            "--abs" => a.abs = take(&mut i),
            "--weak-rf" => a.weak_rf = true,
            "--no-plot" => a.no_plot = true,
            "--keep-ref" => a.keep_ref = true,
            "--lowpass" => {
                a.lowpass = Some(match argv.get(i + 1).and_then(|v| v.parse::<f64>().ok()) {
                    Some(v) => {
                        i += 1;
                        v
                    }
                    None => 8000.0,
                })
            }
            "--lowpass-width" => a.lowpass_width = take(&mut i).parse().unwrap(),
            "--notch-at" => a.notch_at = Some(take(&mut i)),
            "--notch-width" => a.notch_width = take(&mut i).parse().unwrap(),
            "--search" | "--pair-dmax" => {
                take(&mut i);
            }
            "--check-line" => a.check_line = Some(take(&mut i).parse().unwrap()),
            "--bw" => a.bw = take(&mut i),
            "--tape-speed" => a.tape_speed = Some(take(&mut i)),
            "--play-speed" => a.play_speed = Some(take(&mut i)),
            "--plot" => a.plot = Some(take(&mut i)),
            "--curve" => a.curve = Some(take(&mut i)),
            "--report" => a.report = Some(take(&mut i)),
            "--corr-limit" => a.corr_limit = take(&mut i),
            o => panic!("option {o}"),
        }
        i += 1;
    }
    a
}

fn err_eq(what: &str, e: &J, kind: &str, msg: String) {
    assert_eq!(e["msg"].as_str(), Some(msg.as_str()), "{what}");
    assert_eq!(e["error"].as_str(), Some(kind), "{what}");
}

fn perr_kind(e: &pr::ParamError) -> &'static str {
    match e {
        pr::ParamError::Exit(_) => "SystemExit",
        pr::ParamError::Value(_) => "ValueError",
        pr::ParamError::ZeroDivision(_) => "ZeroDivisionError",
    }
}

#[test]
fn params_resolution() {
    let d = json(dir().join("params.json"));
    let mut n = 0;
    for r in d.as_array().unwrap() {
        let argv: Vec<String> = r["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let a = run_args(&argv);
        let what = argv.join(" ");
        let e = &r["resolve"];
        match pr::resolve(&a) {
            Ok(p) => {
                assert!(e.get("error").is_none(), "{what}: reference raised {e}");
                assert_eq!(p.abs.mode(), e["mode"].as_str().unwrap(), "{what}");
                match &p.abs {
                    Abs::Curve(c) => assert_eq!(Some(c.as_str()), e["arg"].as_str()),
                    a => assert_eq!(opt_v(a.arg()), opt_v(jof(&e["arg"])), "{what}"),
                }
                assert_eq!(p.v_rec, jof(&e["v_rec"]), "{what}");
                assert_eq!(p.v_play, jof(&e["v_play"]), "{what}");
                assert_eq!(p.play, jf(&e["play"]), "{what}");
                assert_eq!(p.centre, jf(&e["centre"]), "{what}");
                match p.bw {
                    pr::Bw::Auto => assert_eq!(e["bw"].as_str(), Some("auto")),
                    pr::Bw::Hz(v) => assert_eq!(v, jf(&e["bw"])),
                }
                assert_eq!(p.line_hz, jof(&e["line_hz"]), "{what}");
                assert_eq!(p.v_phys, jof(&e["v_phys"]), "{what}");
                assert_eq!(p.corr_limit, e["corr_limit"].as_str().unwrap(), "{what}");
            }
            Err(x) => err_eq(&what, e, perr_kind(&x), x.to_string()),
        }
        match pr::play_factor(&a) {
            Ok((vr, vp, f)) => {
                let pf = r["play_factor"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{what}: {}", r["play_factor"]));
                assert_eq!(
                    (vr, vp, f),
                    (jof(&pf[0]), jof(&pf[1]), jf(&pf[2])),
                    "{what}"
                );
            }
            Err(x) => err_eq(&what, &r["play_factor"], perr_kind(&x), x.to_string()),
        }
        let op = pr::out_paths(&a).unwrap();
        let eo = &r["out_paths"];
        assert_eq!(op.audio, eo["audio"].as_str().unwrap(), "{what}");
        assert_eq!(op.curve, eo["curve"].as_str().unwrap(), "{what}");
        assert_eq!(op.report, eo["report"].as_str().unwrap(), "{what}");
        assert_eq!(op.plot.as_deref(), eo["plot"].as_str(), "{what}");
        assert_eq!(
            op.model_plot.as_deref(),
            eo["model_plot"].as_str(),
            "{what}"
        );
        assert_eq!(pr::lowpass_stop(&a), jf(&r["lowpass_stop"]), "{what}");
        let mn = pr::manual_notches(&a).unwrap();
        let emn: Vec<f64> = r["manual_notches"]
            .as_array()
            .unwrap()
            .iter()
            .map(jf)
            .collect();
        assert_eq!(mn, emn, "{what}");
        for rn in r["ref_notch"].as_array().unwrap() {
            let (nh, why) = pr::ref_notch(&a, jf(&rn["true_hz"]));
            assert_eq!(nh, jof(&rn["notch_hz"]), "{what}");
            assert_eq!(why, rn["why"].as_str().unwrap(), "{what}");
            for sk in rn["skip"].as_array().unwrap() {
                assert_eq!(
                    pr::notch_skip(&a, jf(&sk[0]), nh),
                    sk[1].as_str(),
                    "{what} {sk}"
                );
            }
        }
        n += 1;
    }
    println!("params: {n} argv cases equal");
}

// ------------------------------------------------------------------ physics

fn comp_from(d: &J) -> Comp {
    let mut c = Comp::new(
        d["name"].as_str().unwrap(),
        jf(&d["a"]),
        jf(&d["b"]),
        jf(&d["beta"]),
    );
    c.harmonics = d["harmonics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    c.kind = if d["kind"] == "chirp" {
        Kind::Chirp
    } else {
        Kind::Fixed
    };
    c.f_mean = jf(&d["f_mean"]);
    c.amp_db = jf(&d["amp_db"]);
    c.pts = d["pts"].as_array().map(|p| {
        let v = |i: usize| {
            p[i].as_array()
                .unwrap()
                .iter()
                .map(jf)
                .collect::<Vec<f64>>()
        };
        (v(0), v(1), v(2))
    });
    c
}

fn comp_to_v(c: &Comp) -> V {
    let pts = match &c.pts {
        Some((a, b, w)) => V::Arr(vec![a.clone().into(), b.clone().into(), w.clone().into()]),
        None => V::Null,
    };
    V::obj()
        .with("name", c.name.as_str())
        .with("a", c.a)
        .with("b", c.b)
        .with("beta", c.beta)
        .with(
            "harmonics",
            V::Arr(c.harmonics.iter().map(|&h| V::Int(h as i64)).collect()),
        )
        .with("kind", c.kind.as_str())
        .with("f_mean", c.f_mean)
        .with("amp_db", c.amp_db)
        .with("pts", pts)
}

fn classes_to_v(c: &ph::Classes) -> V {
    let o = |v: Option<usize>| v.map(|k| V::Int(k as i64)).unwrap_or(V::Null);
    V::obj()
        .with("take", o(c.take))
        .with("supply", o(c.supply))
        .with("clutch", o(c.clutch))
        .with(
            "fixed",
            V::Arr(c.fixed.iter().map(|&k| V::Int(k as i64)).collect()),
        )
}

fn plaus_to_v(p: &[(&str, Vec<String>)]) -> V {
    V::Obj(
        p.iter()
            .map(|(f, v)| (f.to_string(), v.clone().into()))
            .collect(),
    )
}

#[test]
fn physics_inference() {
    let d = json(dir().join("physics.json"));
    let mut cmp = Cmp::new(1e-12, 0.0);
    let (mut ninf, mut ncon) = (0, 0);
    for r in d.as_array().unwrap() {
        let name = r["name"].as_str().unwrap();
        let comps: Vec<Comp> = r["comps"]
            .as_array()
            .unwrap()
            .iter()
            .map(comp_from)
            .collect();
        let span = (jf(&r["span"][0]), jf(&r["span"][1]));
        if let Some(v) = r.get("constrain_v") {
            let v = jf(v);
            match ph::constrain(&comps, v, span) {
                Ok((out, info)) => {
                    let got = V::Arr(out.iter().map(comp_to_v).collect());
                    let mut c2 = Cmp::new(1e-9, 1e-15); // SVD lstsq (gelsd vs faer)
                    c2.val(&format!("{name} constrain {v}"), &r["out"], &got);
                    c2.val(
                        &format!("{name} constrain {v} info"),
                        &r["info"],
                        &info.to_value(),
                    );
                    check_keys(name, &r["keys"], vec![("info", info.to_value())]);
                    cmp.fails.extend(c2.fails);
                    cmp.worst = cmp.worst.max(c2.worst);
                }
                Err(e) => panic!("{name} constrain {v}: {e}"),
            }
            ncon += 1;
            continue;
        }
        let amps: Option<ph::Amps> = r["amps"].as_object().map(|m| {
            m.iter()
                .map(|(k, v)| (k.parse::<usize>().unwrap(), jf(v)))
                .collect()
        });
        let vg = jof(&r["v_given"]);
        let p = format!("{name} amps={} vg={vg:?}", r["amps"]);
        let inf = ph::infer(&comps, span, amps.as_ref(), vg);
        let geo: Vec<(String, V)> = ph::SPEEDS
            .iter()
            .map(|&v| {
                (
                    ts_io::repr_shortest(v),
                    ph::geometry(&comps, v, span, amps.as_ref()).to_value(),
                )
            })
            .collect();
        check_keys(
            &p,
            &r["keys"],
            vec![("infer", inf.to_value()), ("geometry", V::Obj(geo))],
        );
        cmp.val(
            &format!("{p} classify"),
            &r["classify"],
            &classes_to_v(&ph::classify(&comps)),
        );
        cmp.val(
            &format!("{p} infer"),
            &r["infer"],
            &ph::infer(&comps, span, amps.as_ref(), vg).to_value(),
        );
        for v in ph::SPEEDS {
            let key = ts_io::repr_shortest(v);
            let g = ph::geometry(&comps, v, span, amps.as_ref());
            cmp.val(
                &format!("{p} geometry {key}"),
                &r["geometry"][&key],
                &g.to_value(),
            );
            cmp.val(
                &format!("{p} plaus {key}"),
                &r["plausibility"][&key],
                &plaus_to_v(&ph::plausibility(&g)),
            );
        }
        ninf += 1;
    }
    cmp.check("physics");
    println!(
        "physics: {ninf} infer cases, {ncon} constrain cases; worst rel {:.1e}",
        cmp.worst
    );
}

// ------------------------------------------------------------------ apply_l1

fn curve_from(t: Vec<f64>, r: Vec<f64>, v: Vec<bool>) -> Curve {
    let n = t.len();
    Curve {
        t,
        ratio: r,
        confidence: vec![1.0; n],
        valid: v,
        meta: Default::default(),
    }
}

fn pm_from(d: &J) -> PairResult {
    let cands = d["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Candidate {
            f: c.get("f").map_or(0.0, jf),
            prom: jf(&c["prom"]),
            s: c.get("s").map_or(0.0, jf),
            l1_pct: c.get("l1_pct").map_or(0.0, jf),
            delta: c.get("delta").map_or(0.0, jf),
            side: if c.get("side").and_then(|s| s.as_str()) == Some("hi") {
                Side::Hi
            } else {
                Side::Lo
            },
            why: c["why"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| w.as_str().unwrap().to_string())
                .collect(),
            pres: jf(&c["pres"]),
            drift: c.get("drift").map_or(0.0, jf),
            k_hat: jf(&c["k_hat"]),
            gain_db: c.get("gain_db").map_or(0.0, jf),
            k_contrast: c.get("k_contrast").map_or(0.0, jf),
            mirror_db: c.get("mirror_db").map_or(0.0, jf),
            cochan: None,
            comb: None,
            sub_d: None,
        })
        .collect();
    let xcheck = d
        .get("xcheck")
        .and_then(|x| x.as_object())
        .map(|m| {
            m.iter()
                .map(|(k, v)| XCheck {
                    nom: k.parse().unwrap(),
                    f: jf(&v["f"]),
                    prom: jf(&v["prom"]),
                    s: jof(&v["s"]),
                    ppm: jof(&v["ppm"]),
                })
                .collect()
        })
        .unwrap_or_default();
    let solution = d.get("s").map(|_| {
        let bl: Vec<(f64, f64, f64)> = d["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| (jf(&b[0]), jf(&b[1]), jf(&b[2])))
            .collect();
        PairSolution {
            best: d["best"].as_u64().unwrap() as usize,
            s: jf(&d["s"]),
            f_lo: jf(&d["f_lo"]),
            f_hi: jf(&d["f_hi"]),
            delta: jf(&d["delta"]),
            side: if d["side"] == "lo" {
                Side::Lo
            } else {
                Side::Hi
            },
            f_true: jf(&d["f_true"]),
            offset_hz: jf(&d["offset_hz"]),
            blocks: bl,
            blk_stats: d.get("blk_ftrue_ptp_ppm").map(|_| {
                (
                    jf(&d["blk_ftrue_ptp_ppm"]),
                    jf(&d["blk_ftrue_std_ppm"]),
                    jf(&d["blk_delta_ptp"]),
                )
            }),
        }
    });
    PairResult {
        hz: d.get("hz").map_or(9000.0, jf),
        f_primary: d.get("f_primary").map_or(0.0, jf),
        prom_primary: d.get("prom_primary").map_or(0.0, jf),
        r0: d.get("r0").map_or(1.0, jf),
        candidates: cands,
        xcheck,
        n_blocks: d.get("n_blocks").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        sub_lines: vec![],
        ambiguity: d
            .get("ambiguity")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        error: d.get("error").and_then(|v| v.as_str()).map(String::from),
        solution,
    }
}

#[test]
fn apply_l1_modes() {
    let d = json(dir().join("l1.json"));
    let ld = |n: &str| npy_f64(dir().join(format!("l1__{n}.npy")));
    let c1 = curve_from(
        ld("c1_t"),
        ld("c1_ratio"),
        npy_bool(dir().join("l1__c1_valid.npy")),
    );
    let c2 = curve_from(
        ld("c2_t"),
        ld("c2_ratio"),
        npy_bool(dir().join("l1__c2_valid.npy")),
    );
    let mut c2bad = c2.clone();
    c2bad.valid = vec![false; c2.t.len()];
    let pm = pm_from(&d["pm"]);
    // xcheck dict order: serde_json sorts keys, the reference keeps insertion order (15625, 15734)
    let mut worst = 0.0f64;
    for (i, r) in d["cases"].as_array().unwrap().iter().enumerate() {
        let spec = r["spec"].as_str().unwrap();
        let ex = &r["extra"];
        let abs = ab::parse_abs(spec).unwrap();
        let line = if ex.get("bad").is_some() { &c2bad } else { &c2 };
        let pme = PairResult {
            error: Some("--abs pair: no partner line passing the criteria within ...".into()),
            solution: None,
            ..pm.clone()
        };
        let pair = if ex.get("pm").is_some() {
            Some(&pm)
        } else if ex.get("pm_err").is_some() {
            Some(&pme)
        } else {
            None
        };
        let window = ex.get("window").map_or(60.0, jf);
        let play = jf(&r["play"]);
        match ab::apply_l1(&c1, 9000.0, &abs, play, Some(line), window, pair) {
            Ok((s, info)) => {
                assert!(
                    r.get("error").is_none(),
                    "{spec}: reference raised {}",
                    r["msg"]
                );
                let es = npy_f64(dir().join(format!("l1__s{i}.npy")));
                assert_eq!(s.len(), es.len());
                let md = s
                    .iter()
                    .zip(&es)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(md <= 1e-12, "{spec}: s max diff {md:e}");
                let mut cmp = Cmp::new(1e-12, 1e-12);
                let got = info_to_v(&info);
                // abs_l1 at 1e-8 absolute, others tight
                let l1d = (jf(&r["info"]["abs_l1"])
                    - info["abs_l1"].to_text().parse::<f64>().unwrap())
                .abs();
                assert!(l1d <= 1e-8, "{spec}: abs_l1 diff {l1d:e}");
                worst = worst.max(l1d);
                if let J::Object(m) = &r["info"] {
                    let ka: Vec<&String> = m.keys().collect();
                    let kb: Vec<&String> = info.keys().collect();
                    assert!(same_keys(&ka, &kb), "{spec}: keys {ka:?} vs {kb:?}");
                    for (k, v) in m {
                        cmp.val(&format!("{spec}.{k}"), v, got.get(k).unwrap());
                    }
                }
                check_keys(
                    spec,
                    &r["keys"],
                    info.iter()
                        .map(|(k, v)| (k.as_str(), meta_to_v(v)))
                        .collect(),
                );
                cmp.check(spec);
            }
            Err(e) => {
                assert_eq!(Some(e.to_string().as_str()), r["msg"].as_str(), "{spec}");
            }
        }
    }
    for h in d["huber"].as_array().unwrap() {
        let v: Vec<f64> = h["v"].as_array().unwrap().iter().map(jf).collect();
        assert_eq!(ab::huber_mean(&v, 1.5, 5), jf(&h["m"]));
    }
    println!("apply_l1: all modes equal, worst abs_l1 diff {worst:.1e}");
}

/// Key order of apply_l1 info (serde_json maps are sorted, so check the order of our keys
/// against the documented insertion order for the line and pair modes).
#[test]
fn apply_l1_key_order() {
    let d = json(dir().join("l1.json"));
    let ld = |n: &str| npy_f64(dir().join(format!("l1__{n}.npy")));
    let c1 = curve_from(
        ld("c1_t"),
        ld("c1_ratio"),
        npy_bool(dir().join("l1__c1_valid.npy")),
    );
    let c2 = curve_from(
        ld("c2_t"),
        ld("c2_ratio"),
        npy_bool(dir().join("l1__c2_valid.npy")),
    );
    let (_, info) =
        ab::apply_l1(&c1, 9000.0, &Abs::Line(15625.0), 1.0, Some(&c2), 60.0, None).unwrap();
    let k: Vec<&str> = info.keys().map(|s| s.as_str()).collect();
    assert_eq!(
        k,
        [
            "abs_mode",
            "play_factor",
            "line_line_valid",
            "line_common",
            "line_q",
            "line_q_std_pct",
            "line_q_drift_pct",
            "line_n_blocks",
            "line_hz",
            "line_disagree_pct",
            "abs_l1",
            "abs_l1_pct",
            "abs_ref_true_hz"
        ]
    );
    let pm = pm_from(&d["pm"]);
    let (_, info) =
        ab::apply_l1(&c1, 9000.0, &Abs::Pair(9000.0), 1.0, None, 60.0, Some(&pm)).unwrap();
    let k: Vec<&str> = info.keys().map(|s| s.as_str()).collect();
    assert_eq!(
        k,
        [
            "abs_mode",
            "play_factor",
            "pair_hz",
            "pair_lo_hz",
            "pair_hi_hz",
            "pair_delta_hz",
            "pair_side",
            "pair_offset_hz",
            "pair_ambiguity",
            "pair_prom_db",
            "pair_pres",
            "pair_k",
            "pair_n_cand",
            "pair_n_rejected",
            "pair_n_blocks",
            "pair_blk_ftrue_std_ppm",
            "pair_blk_ftrue_ptp_ppm",
            "pair_blk_delta_ptp",
            "pair_xcheck_ppm",
            "abs_l1",
            "abs_l1_pct",
            "abs_ref_true_hz"
        ]
    );
}

// ------------------------------------------------------------------ pair / switch

/// Compare a pair_measure result with the reference dict (keys compared as sets: serde_json
/// sorts object keys; our order matches the reference by construction). Floats: mixed 1e-6 rel / 1e-6 abs
/// (float32 PSD; FFT rounding differs at 1e-16), strings and decisions exact; s and f_true
/// checked separately at 1e-9 relative.
fn cmp_pair(what: &str, e: &J, g: &PairResult, table: &J) -> (f64, f64) {
    let gv = g.to_value();
    let mut cmp = Cmp::new(1e-6, 1e-6);
    cmp_sorted(&mut cmp, what, e, &gv);
    let tab = ab::pair_table(g);
    let et: Vec<&str> = table
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for (i, (a, b)) in et.iter().zip(&tab).enumerate() {
        if *a != b {
            cmp.fails
                .push(format!("{what} table line {i}:\n ref {a}\n got {b}"));
        }
    }
    if et.len() != tab.len() {
        cmp.fails
            .push(format!("{what}: table len {} vs {}", et.len(), tab.len()));
    }
    let mut srel = 0.0;
    if let Some(sol) = &g.solution {
        srel = (sol.s / jf(&e["s"]) - 1.0)
            .abs()
            .max((sol.f_true / jf(&e["f_true"]) - 1.0).abs());
        assert!(srel <= 1e-9, "{what}: s / f_true rel {srel:e}");
    }
    cmp.check(what);
    (cmp.worst, srel)
}

fn cmp_sorted(cmp: &mut Cmp, path: &str, e: &J, g: &V) {
    match (e, g) {
        (J::Object(a), V::Obj(b)) => {
            let ka: Vec<&String> = a.keys().collect();
            let kb: Vec<&String> = b.iter().map(|(k, _)| k).collect();
            if !same_keys(&ka, &kb) {
                cmp.fails
                    .push(format!("{path}: keys ref {ka:?} got {kb:?}"));
                return;
            }
            for (k, x) in a {
                cmp_sorted(cmp, &format!("{path}.{k}"), x, g.get(k).unwrap());
            }
        }
        (J::Array(a), V::Arr(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                cmp_sorted(cmp, &format!("{path}[{i}]"), x, y);
            }
        }
        _ => cmp.val(path, e, g),
    }
}

fn params_from(kw: &J) -> PairParams {
    let mut p = PairParams::default();
    if let Some(o) = kw.as_object() {
        for (k, v) in o {
            match k.as_str() {
                "blk_s" => p.blk_s = jof(v),
                "seg_s" => p.seg_s = jf(v),
                "hz" => p.hz = jf(v),
                "dmax" => p.dmax = jf(v),
                "dmin" => p.dmin = jf(v),
                "play" => p.play = jf(v),
                "smax" => p.smax = jf(v),
                "xcheck" => p.xcheck = v.as_array().unwrap().iter().map(jf).collect(),
                "channel" => p.channel = v.as_str().unwrap().parse().unwrap(),
                _ => panic!("kw {k}"),
            }
        }
    }
    p
}

#[test]
fn pair_and_switch_synthetic() {
    let calls = json(dir().join("pair.json"));
    let sigs = ["pairsig", "switch", "nopartner"];
    let mut worst = (0.0f64, 0.0f64);
    for sig in sigs {
        let (shape, x) = npy_f32(dir().join(format!("pair_{sig}__x.npy")));
        let ch = if shape.len() > 1 { shape[1] } else { 1 };
        let ld = |n: &str| npy_f64(dir().join(format!("pair_{sig}__{n}.npy")));
        let c1 = Curve {
            t: ld("c1_t"),
            ratio: ld("c1_ratio"),
            confidence: ld("c1_confidence"),
            valid: npy_bool(dir().join(format!("pair_{sig}__c1_valid.npy"))),
            meta: Default::default(),
        };
        let mut csw: Option<Curve> = None;
        for (i, c) in calls
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, c)| c["sig"] == sig)
        {
            let what = format!("{sig}#{i} {} {}", c["fn"], c.get("kw").unwrap_or(&J::Null));
            match c["fn"].as_str().unwrap() {
                "pair_measure" => {
                    let cur = if c["curve"] == "sw" {
                        csw.as_ref().unwrap()
                    } else {
                        &c1
                    };
                    let pm = ab::pair_measure(&x, ch, 24000.0, cur, 9000.0, &params_from(&c["kw"]))
                        .unwrap();
                    let w = cmp_pair(&what, &c["res"], &pm, &c["table"]);
                    check_keys(&what, &c["keys"], vec![("res", pm.to_value())]);
                    worst = (worst.0.max(w.0), worst.1.max(w.1));
                    let gr = ab::ghost_rho(&pm, 2.5);
                    match jof(&c["ghost"]) {
                        Some(e) => assert!(
                            (gr.unwrap() / e - 1.0).abs() <= 1e-9,
                            "{what}: ghost {gr:?} vs {e}"
                        ),
                        None => assert!(gr.is_none(), "{what}: ghost {gr:?}"),
                    }
                }
                "pair_info" => {
                    let pm = ab::pair_measure(&x, ch, 24000.0, &c1, 9000.0, &params_from(&c["kw"]))
                        .unwrap();
                    let mut cmp = Cmp::new(1e-6, 1e-6);
                    cmp_sorted(&mut cmp, &what, &c["info"], &info_to_v(&ab::pair_info(&pm)));
                    cmp.check(&what);
                    check_keys(
                        &what,
                        &c["keys"],
                        vec![("info", info_to_v(&ab::pair_info(&pm)))],
                    );
                }
                "switch_repair" => {
                    let rho = jf(&c["rho"]);
                    let kw = c.get("kw").cloned().unwrap_or(J::Null);
                    let blk = kw.get("blk_s").map_or(5.0, jf);
                    let seg = kw.get("seg_s").map_or(2.0, jf);
                    let (cr, info) = ab::switch_repair(
                        &x,
                        ch,
                        24000.0,
                        &c1,
                        9000.0,
                        rho,
                        Channel::Mid,
                        blk,
                        seg,
                    )
                    .unwrap();
                    let mut cmp = Cmp::new(1e-6, 1e-9);
                    cmp_sorted(&mut cmp, &what, &c["info"], &info.to_value());
                    cmp.check(&what);
                    check_keys(&what, &c["keys"], vec![("info", info.to_value())]);
                    let out = c.get("out").and_then(|o| o.as_str()).unwrap_or("sw");
                    let er = ld(&format!("{out}_ratio"));
                    let md = cr
                        .ratio
                        .iter()
                        .zip(&er)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    assert!(md <= 1e-12, "{what}: repaired ratio max diff {md:e}");
                    if out == "sw" {
                        csw = Some(cr);
                    }
                }
                f => panic!("fn {f}"),
            }
        }
    }
    println!(
        "pair synthetic: worst float rel {:.1e}, worst s/f_true rel {:.1e}",
        worst.0, worst.1
    );
}

// ------------------------------------------------------------------ determinism

/// 1 thread == N threads, bit for bit: pair_measure (parallel FFT segments and
/// warp chunks, segment-ordered float32 means) and switch_repair on the switch signal.
#[test]
fn thread_determinism() {
    let (shape, x) = npy_f32(dir().join("pair_switch__x.npy"));
    let ch = if shape.len() > 1 { shape[1] } else { 1 };
    let ld = |n: &str| npy_f64(dir().join(format!("pair_switch__{n}.npy")));
    let c1 = Curve {
        t: ld("c1_t"),
        ratio: ld("c1_ratio"),
        confidence: ld("c1_confidence"),
        valid: npy_bool(dir().join("pair_switch__c1_valid.npy")),
        meta: Default::default(),
    };
    let p = PairParams {
        blk_s: Some(16.0),
        xcheck: vec![15625.0],
        ..Default::default()
    };
    let run = || {
        let pm = ab::pair_measure(&x, ch, 24000.0, &c1, 9000.0, &p).unwrap();
        let sw = ab::switch_repair(
            &x,
            ch,
            24000.0,
            &c1,
            9000.0,
            9066.0 / 8934.0,
            Channel::Mid,
            5.0,
            2.0,
        )
        .unwrap();
        (pm, sw)
    };
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(run);
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(run);
    assert_eq!(one.0, many.0);
    assert_eq!(one.1.1, many.1.1);
    assert!(
        one.1
            .0
            .ratio
            .iter()
            .zip(&many.1.0.ratio)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
