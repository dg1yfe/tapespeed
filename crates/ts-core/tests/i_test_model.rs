//! Model tests: model fill vs PCHIP, time-map composition, aided tracking of flutter, plus the
//! gap / long-gap / compose / coarse_track outputs against stored reference results
//! (tests/data/i).
mod i_common;
use i_common::*;

use ts_core::curve::Curve;
use ts_core::model::aided::{AidedOpts, CoarseOpts, coarse_track, track_aided};
use ts_core::model::gaps::{
    compose, fill_long_gaps_default, long_gaps_default, model_fill, pchip_fill, time_integral,
};
use ts_core::model::{Amp, Comp, FitOpts, HierOpts, IdentOpts, fit, fit_hier};

fn ld(case: &str, k: &str) -> Vec<f64> {
    npy_f64(&data(&format!("{case}__{k}.npy")))
}

fn ldb(case: &str, k: &str) -> Vec<bool> {
    npy_bool(&data(&format!("{case}__{k}.npy")))
}

fn g_data(k: &str) -> Vec<f64> {
    npy_f64(&testdata::dir("g").join(format!("mech300__{k}.npy")))
}

fn ones_curve(t: Vec<f64>, r: Vec<f64>) -> Curve {
    let n = t.len();
    Curve {
        t,
        ratio: r,
        confidence: vec![1.0; n],
        valid: vec![true; n],
        meta: Default::default(),
    }
}

fn gap_rms(a: &[f64], b: &[f64], gap: &[bool]) -> f64 {
    let d: Vec<f64> = (0..a.len())
        .filter(|&i| gap[i])
        .map(|i| a[i] - b[i])
        .collect();
    rms(&d)
}

/// The model fill beats PCHIP in the gaps (mech300 fixture, tests/data/g).
#[test]
fn test_model_fill_beats_pchip() {
    let (t, r, rt) = (g_data("t"), g_data("r"), g_data("rt"));
    let v: Vec<bool> = g_data("w").iter().map(|&x| x > 0.0).collect();
    let w: Vec<f64> = v.iter().map(|&b| b as u8 as f64).collect();
    let o = HierOpts {
        ident: IdentOpts {
            win: 60.0,
            hop: 15.0,
            ..IdentOpts::default()
        },
        ..HierOpts::default()
    };
    let h = fit_hier(&t, &r, &w, &o).unwrap();
    let rp: Vec<f64> = r
        .iter()
        .zip(&v)
        .map(|(&x, &b)| if b { x } else { 0.0 })
        .collect();
    let gap: Vec<bool> = v.iter().map(|b| !b).collect();
    let pf = pchip_fill(&t, &rp, &v).unwrap();
    let mf = model_fill(&t, &rp, &v, &h.m3, 0.1);
    let (e_p, e_m) = (gap_rms(&pf, &rt, &gap), gap_rms(&mf, &rt, &gap));
    let e = json(&data("fill.json"));
    eprintln!(
        "model_fill e_m {e_m:.4e} (ref {}), pchip e_p {e_p:.4e} (ref {})",
        e["e_m"], e["e_p"]
    );
    assert!(e_m < 0.25 * e_p, "{e_m} {e_p}");
    // pchip is exact arithmetic on identical inputs; the model fill inherits the fit_hier tolerance
    assert!(max_abs_diff(&pf, &ld("fill", "pchip")) < 1e-15);
    assert!((e_m - f(&e["e_m"])).abs() < 1e-3 * f(&e["e_m"]));
    assert!(max_abs_diff(&mf, &ld("fill", "model")) < 1e-8);
}

/// compose() is the composition of the time maps (+ valid / confidence mapping of a partly
/// invalid second curve, against the stored reference).
#[test]
fn test_compose_is_time_map_composition() {
    let dt = ts_core::model::DT;
    let t: Vec<f64> = (0..4000).map(|i| i as f64 * dt).collect();
    let tau: Vec<f64> = (0..4200).map(|i| i as f64 * dt).collect();
    // grids [0, 20) / [0, 21) in steps of DT: lengths as in the reference
    assert_eq!(t.len(), ld("compose", "r1").len());
    assert_eq!(tau.len(), ld("compose", "r2").len());
    let r1 = ld("compose", "r1");
    let r2 = ld("compose", "r2");
    let c1 = ones_curve(t.clone(), r1.clone());
    let mut c2 = ones_curve(tau.clone(), r2.clone());
    let ct_plain = compose(&c1, &c2).unwrap();
    let t1 = time_integral(&t, &r1).unwrap();
    let ttot = time_integral(&t, &ct_plain.ratio).unwrap();
    let t2 = time_integral(&tau, &r2).unwrap();
    let err = t.iter().fold(0.0f64, |m, &x| {
        m.max((ttot.eval(x) - t2.eval(t1.eval(x))).abs())
    });
    eprintln!(
        "compose max |Ttot - T2(T1)| {err:.3e} (ref {})",
        json(&data("compose.json"))["max_err"]
    );
    assert!(err < 1e-7, "{err}");
    let t1v: Vec<f64> = t.iter().map(|&x| t1.eval(x)).collect();
    assert!(max_abs_diff(&t1v, &ld("compose", "T1")) < 1e-12);
    c2.confidence = ld("compose", "q2");
    c2.valid = ldb("compose", "v2");
    let ct = compose(&c1, &c2).unwrap();
    assert!(max_abs_diff(&ct.ratio, &ld("compose", "ratio")) < 1e-14);
    assert!(max_abs_diff(&ct.confidence, &ld("compose", "confidence")) < 1e-12);
    assert_eq!(ct.valid, ldb("compose", "valid"));
    assert_eq!(ct.meta["composed"].to_text(), "r1*r2(T1)");
}

/// Aided tracking recovers flutter (stored synthetic audio and truth).
#[test]
fn test_track_aided_recovers_flutter() {
    let e = json(&data("flutter.json"));
    let x: Vec<f32> = {
        let p = data("flutter__x.npy");
        let f = std::fs::File::open(&p).unwrap();
        npyz::NpyFile::new(std::io::BufReader::new(f))
            .unwrap()
            .into_vec::<f32>()
            .unwrap()
    };
    let (tt, tr) = (ld("flutter", "truth_t"), ld("flutter", "truth_ratio"));
    let mut cap = Comp::new("cap", 7.46, 0.0, 0.0);
    cap.harmonics = vec![1];
    let mut rol = Comp::new("rol", 1.48, 0.0, 0.0);
    rol.harmonics = vec![1];
    let ones = vec![1.0; tt.len()];
    let mut m = fit(
        &tt,
        &tr,
        &ones,
        &[cap, rol],
        &FitOpts {
            level: 3,
            ta: 5.0,
            ..FitOpts::default()
        },
    )
    .unwrap();
    for a in m.amp.values_mut() {
        match a {
            Amp::L3(v) => v.iter_mut().for_each(|z| *z *= 1.1),
            Amp::L2(z) => *z *= 1.1,
        }
    }
    let sr = e["sr"].as_u64().unwrap() as u32;
    let o = AidedOpts {
        b: 2.0,
        refine: 2,
        ..AidedOpts::default()
    };
    let (c, _) = track_aided(&x, 1, sr, f(&e["ref_hz"]), &m, &o).unwrap();
    let core: Vec<bool> = (0..c.t.len())
        .map(|i| c.valid[i] && c.t[i] > 1.0 && c.t[i] < 19.0)
        .collect();
    let cm = core.iter().filter(|&&b| b).count() as f64 / core.len() as f64;
    assert!(cm > 0.8, "{cm}");
    let ec: Vec<f64> = (0..c.t.len())
        .filter(|&i| core[i])
        .map(|i| c.ratio[i] - tr[i])
        .collect();
    let pm = m.predict(&c.t);
    let em: Vec<f64> = (0..c.t.len())
        .filter(|&i| core[i])
        .map(|i| pm[i] - tr[i])
        .collect();
    let (er, emr) = (rms(&ec), rms(&em));
    eprintln!(
        "flutter: core {cm:.4} err {er:.4e} model err {emr:.4e} (ref {} {} {})",
        e["core_mean"], e["err"], e["err_model"]
    );
    assert!(er < 1e-4 && er < 0.2 * emr, "{er} {emr}");
    // agreement with the reference curve on the same inputs (fit agreement ~1e-10 → rounding level)
    assert_eq!(mismatches(&c.valid, &ldb("flutter", "valid")), 0);
    let d: Vec<f64> = c
        .ratio
        .iter()
        .zip(&ld("flutter", "ratio"))
        .map(|(a, b)| a - b)
        .collect();
    assert!(rms(&d) < 1e-9, "{}", rms(&d));
    assert!((cm - f(&e["core_mean"])).abs() < 1e-12 && (er - f(&e["err"])).abs() < 1e-3 * er);
}

/// coarse_track on the flutter audio vs the stored reference.
#[test]
fn coarse_track_ref() {
    let e = json(&data("flutter.json"));
    let f32s: Vec<f32> = {
        let f = std::fs::File::open(data("flutter__x.npy")).unwrap();
        npyz::NpyFile::new(std::io::BufReader::new(f))
            .unwrap()
            .into_vec::<f32>()
            .unwrap()
    };
    let c = coarse_track(
        &f32s,
        1,
        e["sr"].as_u64().unwrap() as u32,
        f(&e["ref_hz"]),
        &CoarseOpts::default(),
    )
    .unwrap();
    let mm = mismatches(&c.valid, &ldb("coarse", "valid"));
    let d = max_abs_diff(&c.ratio, &ld("coarse", "ratio"));
    eprintln!(
        "coarse_track: valid mismatch {mm}/{}, max |dr| {d:.2e}",
        c.valid.len()
    );
    assert_eq!(mm, 0);
    assert!(d < 1e-12, "{d}");
}

/// long_gaps / fill_long_gaps (interior: edge-linear, ends: cosine ramp to L1) and the
/// model_fill / pchip_fill edge cases (gaps at both file ends) vs the stored reference.
#[test]
fn gaps_ref() {
    let (t, r) = (ld("gaps", "t"), ld("gaps", "r"));
    let e = json(&data("gaps.json"));
    for (vk, gk, rk, vok, l1k) in [
        ("v", "gaps", "ratio", "valid", "L1"),
        ("v2", "gaps2", "ratio2", "valid2", "L1_2"),
    ] {
        let v = ldb("gaps", vk);
        let g = long_gaps_default(&t, &v).unwrap();
        let ge: Vec<(usize, usize)> = ld("gaps", gk)
            .chunks(2)
            .map(|c| (c[0] as usize, c[1] as usize))
            .collect();
        assert_eq!(g, ge, "{vk}: long_gaps");
        let (rr, vv, l1) = fill_long_gaps_default(&t, &r, &v, &g).unwrap();
        assert_eq!(vv, ldb("gaps", vok));
        let d = max_abs_diff(&rr, &ld("gaps", rk));
        eprintln!(
            "{vk}: gaps {g:?}, L1 {l1} (ref {}), max |dr| {d:.2e}",
            e[l1k]
        );
        assert!((l1 - f(&e[l1k])).abs() < 1e-12 && d < 1e-12);
    }
    let v = ldb("gaps", "v");
    let w: Vec<f64> = v.iter().map(|&b| b as u8 as f64).collect();
    let mut c = Comp::new("w", 0.025, 0.0, 0.0);
    c.harmonics = vec![1, 2];
    let m = fit(
        &t,
        &r,
        &w,
        &[c],
        &FitOpts {
            level: 3,
            ta: 20.0,
            ..FitOpts::default()
        },
    )
    .unwrap();
    let mf = model_fill(&t, &r, &v, &m, 0.1);
    let dm = max_abs_diff(&mf, &ld("gaps", "model_fill"));
    let dp = max_abs_diff(&pchip_fill(&t, &r, &v).unwrap(), &ld("gaps", "pchip"));
    eprintln!("model_fill max diff {dm:.2e}, pchip_fill {dp:.2e}");
    assert!(dm < 1e-9 && dp < 1e-14);
}
