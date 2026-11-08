//! Shared helpers for the model tests: npy / JSON loading (tests/data/g) and comparison of
//! comps / models / explained against stored reference results.
#![allow(dead_code)]
#[path = "../../../testdata.rs"]
pub mod testdata;
use std::path::{Path, PathBuf};

use serde_json::Value;
use ts_core::model::{Amp, Comp, Hier, Kind, Model};

pub fn data(name: &str) -> PathBuf {
    testdata::dir("g").join(name)
}

pub fn npy_f64(p: &Path) -> Vec<f64> {
    let f = std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let npy = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    let d = npy.dtype().descr();
    if d.contains("b1") || d.contains("|b") {
        npy.into_vec::<bool>()
            .unwrap()
            .into_iter()
            .map(|b| if b { 1.0 } else { 0.0 })
            .collect()
    } else if d.contains("i8") {
        npy.into_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        npy.into_vec::<f64>().unwrap()
    }
}

pub fn ld(case: &str, name: &str) -> Vec<f64> {
    npy_f64(&data(&format!("{case}__{name}.npy")))
}

pub fn ld_bool(case: &str, name: &str) -> Vec<bool> {
    ld(case, name).into_iter().map(|v| v != 0.0).collect()
}

pub fn json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

pub fn f(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) if s == "nan" => f64::NAN,
        _ => panic!("not a number: {v}"),
    }
}

pub fn fv(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

pub fn comp_from(v: &Value) -> Comp {
    let mut c = Comp::new(
        v["name"].as_str().unwrap(),
        f(&v["a"]),
        f(&v["b"]),
        f(&v["beta"]),
    );
    c.harmonics = v["harmonics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h.as_u64().unwrap() as u32)
        .collect();
    c.kind = if v["kind"] == "chirp" {
        Kind::Chirp
    } else {
        Kind::Fixed
    };
    c.f_mean = f(&v["f_mean"]);
    c.amp_db = f(&v["amp_db"]);
    c
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Worst errors of a comparison.
#[derive(Debug, Default, Clone, Copy)]
pub struct Errs {
    pub f_mean: f64,
    pub law: f64,
    pub amp_db: f64,
    pub r0: f64,
    pub drift: f64,
    pub amp2: f64,
    pub amp3: f64,
    /// worst per-key relative amplitude error of keys ≥ 1 % of the strongest (L2 and L3)
    pub amp_strong: f64,
    pub ev: f64,
}

impl Errs {
    pub fn max(self, o: Errs) -> Errs {
        Errs {
            f_mean: self.f_mean.max(o.f_mean),
            law: self.law.max(o.law),
            amp_db: self.amp_db.max(o.amp_db),
            r0: self.r0.max(o.r0),
            drift: self.drift.max(o.drift),
            amp2: self.amp2.max(o.amp2),
            amp3: self.amp3.max(o.amp3),
            amp_strong: self.amp_strong.max(o.amp_strong),
            ev: self.ev.max(o.ev),
        }
    }
}

/// Comps: count, kind, harmonics, name exact; returns (max |Δf_mean| Hz, max rel Δ law
/// coefficients, max |Δamp_db|).
pub fn cmp_comps(case: &str, got: &[Comp], exp: &Value) -> (f64, f64, f64) {
    let exp = exp.as_array().unwrap();
    let names: Vec<&str> = got.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(got.len(), exp.len(), "{case}: comp count {names:?}");
    let (mut df, mut dl, mut da) = (0.0f64, 0.0f64, 0.0f64);
    for (c, e) in got.iter().zip(exp) {
        let ce = comp_from(e);
        assert_eq!(c.kind, ce.kind, "{case}: kind of {}", c.name);
        assert_eq!(c.harmonics, ce.harmonics, "{case}: harmonics of {}", c.name);
        assert_eq!(c.name, ce.name, "{case}: name");
        df = df.max((c.f_mean - ce.f_mean).abs());
        for (a, b) in [(c.a, ce.a), (c.b, ce.b), (c.beta, ce.beta)] {
            dl = dl.max((a - b).abs() / b.abs().max(1e-300).max(if b == 0.0 { 1.0 } else { 0.0 }));
        }
        da = da.max((c.amp_db - ce.amp_db).abs());
    }
    (df, dl, da)
}

/// Model: r0 abs, drift coefficients rel (to max |coef|), amplitudes: max |Δa| over all keys
/// and knots relative to the model's largest |a| (global), and the worst per-key relative error
/// among keys with max |a| ≥ 1 % of the largest (strong). Returns (r0, drift, global, strong).
pub fn cmp_model(case: &str, m: &Model, e: &Value) -> (f64, f64, f64, f64) {
    assert_eq!(m.level as u64, e["level"].as_u64().unwrap());
    let r0 = (m.r0 - f(&e["r0"])).abs();
    let mut dd = 0.0f64;
    if let Some(dc) = &m.drift_coef {
        let de = fv(&e["drift_coef"]);
        let kn = fv(&e["drift_knots"]);
        // the reference JSON stores 16 significant digits only
        let mk = m.drift_knots.as_ref().unwrap();
        assert_eq!(mk.len(), kn.len(), "{case}: knots");
        for (a, b) in mk.iter().zip(&kn) {
            assert!(
                (a - b).abs() <= 1e-15 * b.abs().max(1.0),
                "{case}: knot {a} vs {b}"
            );
        }
        let s = max_abs(&de);
        dd = dc
            .iter()
            .zip(&de)
            .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()))
            / s;
    }
    let ea = e["amp"].as_object().unwrap();
    assert_eq!(m.amp.len(), ea.len(), "{case}: amp keys");
    let mut pairs = vec![];
    for ((k, h), a) in &m.amp {
        let ev = &ea[&format!("{k},{h}")];
        let pr: Vec<(f64, f64)> = match (a, ev) {
            (Amp::L2(z), Value::Array(v)) => vec![(z.re, f(&v[0])), (z.im, f(&v[1]))],
            (Amp::L3(z), Value::Object(o)) => {
                let (er, ei) = (fv(&o["re"]), fv(&o["im"]));
                assert_eq!(z.len(), er.len(), "{case}: knots of ({k},{h})");
                z.iter()
                    .zip(er.iter().zip(&ei))
                    .flat_map(|(a, (r, i))| [(a.re, *r), (a.im, *i)])
                    .collect()
            }
            _ => panic!("{case}: amp kind mismatch at ({k},{h})"),
        };
        let mx = pr.iter().fold(0.0f64, |m, (_, b)| m.max(b.abs()));
        let d = pr.iter().fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        if std::env::var("G_DEBUG").is_ok() {
            eprintln!(
                "  {case} L{} ({k},{h}): max|a| {mx:.3e} max|d| {d:.3e} rel {:.2e}",
                m.level,
                d / mx
            );
        }
        pairs.push((d, mx));
    }
    let gmax = pairs.iter().fold(0.0f64, |m, p| m.max(p.1)).max(1e-300);
    let glob = pairs.iter().fold(0.0f64, |m, p| m.max(p.0)) / gmax;
    let strong = pairs
        .iter()
        .filter(|p| p.1 >= 0.01 * gmax)
        .fold(0.0f64, |m, p| m.max(p.0 / p.1));
    (r0, dd, glob, strong)
}

/// Full fit_hier comparison against a reference JSON with keys comps, m1, m2, m3, explained.
pub fn cmp_hier(case: &str, h: &Hier, ex: &indexmap::IndexMap<String, f64>, e: &Value) -> Errs {
    let (df, dl, da) = cmp_comps(case, &h.comps, &e["comps"]);
    let (r0, ..) = cmp_model(case, &h.m1, &e["m1"]);
    let (r2, d2, a2, s2) = cmp_model(case, &h.m2, &e["m2"]);
    let (r3, d3, a3, s3) = cmp_model(case, &h.m3, &e["m3"]);
    let mut dev = 0.0f64;
    for (k, v) in ex {
        dev = dev.max((v - f(&e["explained"][k])).abs());
    }
    Errs {
        f_mean: df,
        law: dl,
        amp_db: da,
        r0: r0.max(r2).max(r3),
        drift: d2.max(d3),
        amp2: a2,
        amp3: a3,
        amp_strong: s2.max(s3),
        ev: dev,
    }
}
