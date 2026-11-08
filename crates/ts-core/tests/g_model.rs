//! Model fitting against stored reference results (tests/data/g): fit_law, identify, fit
//! L1-L3, level_outliers, fit_hier, explained; behaviour tests (reel law, identify and levels,
//! sparse fit) and thread determinism.

mod g_common;
use g_common::*;
use ts_core::model::{
    self, Comp, FitOpts, HierOpts, IdentOpts, Kind, explained, fit, fit_hier, fit_law,
    level_outliers,
};

fn hier_opts(e: &serde_json::Value) -> HierOpts {
    let mut o = HierOpts {
        ta: f(&e["Ta"]),
        ..HierOpts::default()
    };
    if let Some(ik) = e["ident_kw"].as_object() {
        o.ident = IdentOpts {
            win: f(&ik["win"]),
            hop: f(&ik["hop"]),
            ..IdentOpts::default()
        };
    }
    o.ident.keep_p = true;
    o
}

/// fit_hier result, explained, t, r, w
type CaseOut = (
    model::Hier,
    indexmap::IndexMap<String, f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
);

fn run_case(case: &str) -> CaseOut {
    let e = json(&data(&format!("{case}.json")));
    let (t, r, w) = (ld(case, "t"), ld(case, "r"), ld(case, "w"));
    let h = fit_hier(&t, &r, &w, &hier_opts(&e)).unwrap();
    let vv: Vec<f64> = w
        .iter()
        .zip(&h.rejected)
        .map(|(&a, &j)| if a > 0.0 && !j { 1.0 } else { 0.0 })
        .collect();
    let ex = explained(&[("m1", &h.m1), ("m2", &h.m2), ("m3", &h.m3)], &t, &r, &vv);
    (h, ex, t, r, w)
}

fn check_case(case: &str) -> Errs {
    let e = json(&data(&format!("{case}.json")));
    let t0 = std::time::Instant::now();
    let (h, ex, _, _, _) = run_case(case);
    let el = t0.elapsed().as_secs_f64();
    // level_outliers exact
    let rej = ld_bool(case, "rejected");
    assert_eq!(h.rejected, rej, "{case}: rejected");
    // spectrogram of identify
    let info = h.info.as_ref().unwrap();
    let (tc, vf, p) = (ld(case, "tc"), ld(case, "vf"), ld(case, "P"));
    assert_eq!(info.sg.tc, tc, "{case}: tc");
    let dvf = info
        .sg
        .vf
        .iter()
        .zip(&vf)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let nbp = p.len() / tc.len();
    let nb = info.sg.f.len();
    let mut dp = 0.0f64;
    for i in 0..tc.len() {
        let pr = &p[i * nbp..(i + 1) * nbp];
        let mx = pr.iter().fold(0.0f64, |m, v| m.max(*v));
        for (a, b) in info.sg.p[i * nb..i * nb + nbp].iter().zip(pr) {
            dp = dp.max((a - b).abs() / mx);
        }
    }
    let err = cmp_hier(case, &h, &ex, &e);
    println!(
        "{case}: {} comps, EV {:.6} | vf {dvf:.1e} P {dp:.1e} | f_mean {:.1e} Hz law {:.1e} amp_db {:.1e} | r0 {:.1e} drift {:.1e} amp2 {:.1e} amp3 {:.1e} strong {:.1e} | EV {:.1e} | {el:.2} s (ref {:.2} s)",
        h.comps.len(),
        ex["expl_m3"],
        err.f_mean,
        err.law,
        err.amp_db,
        err.r0,
        err.drift,
        err.amp2,
        err.amp3,
        err.amp_strong,
        err.ev,
        f(&e["seconds"])
    );
    assert!(dvf < 1e-12 && dp < 1e-9, "{case}: spectrogram");
    assert!(err.f_mean <= 1e-6, "{case}: f_mean {:e}", err.f_mean);
    // global amplitude error (relative to the strongest |a|) <= 1e-6; per strong key 5e-6:
    // the fit itself reproduces the reference to ~1e-10 given the reference comps, the residual
    // comes from the comp laws (bounded scalar minimisation on a flat SSE(beta): the minimiser
    // position is set by rounding noise of the 2-column lstsq, beta differs up to ~1e-6 rel,
    // f_mean ~1e-10 Hz)
    assert!(err.amp2 <= 1e-6 && err.amp3 <= 1e-6, "{case}: amps {err:?}");
    assert!(err.amp_strong <= 5e-6, "{case}: strong amps {err:?}");
    assert!(err.ev <= 1e-5, "{case}: EV {:e}", err.ev);
    err
}

#[test]
fn fit_hier_ref_mech300() {
    check_case("mech300");
}

#[test]
fn fit_hier_ref_mech() {
    check_case("mech");
}

#[test]
fn fit_hier_ref_am120() {
    check_case("am120");
}

#[test]
fn fit_hier_ref_real_a() {
    check_case("real_a");
}

#[test]
fn fit_law_ref() {
    let cases = json(&data("fit_law.json"));
    for c in cases.as_array().unwrap() {
        let (t, ff) = (fv(&c["t"]), fv(&c["f"]));
        let w = c["w"].as_array().map(|_| fv(&c["w"]));
        let sp = fv(&c["span"]);
        let l = fit_law(&t, &ff, w.as_deref(), Some((sp[0], sp[1])), 161);
        let name = c["name"].as_str().unwrap();
        assert_eq!(l.kind.as_str(), c["kind"].as_str().unwrap(), "{name}");
        // a, b relative to the law scale |a| + |b| (the reel law has a ≈ 0); rms relative to
        // the mean frequency (an exact law has rms ≈ 0)
        let sc = f(&c["a"]).abs() + f(&c["b"]).abs();
        let fs = ff.iter().map(|v| v.abs()).sum::<f64>() / ff.len() as f64;
        let rb = (l.beta - f(&c["beta"])).abs() / f(&c["beta"]).abs().max(1e-300);
        let e = [
            (l.a - f(&c["a"])).abs() / sc,
            (l.b - f(&c["b"])).abs() / sc,
            if l.beta == 0.0 { 0.0 } else { rb },
            (l.rms - f(&c["rms"])).abs() / fs,
        ];
        println!(
            "fit_law {name} {}: rel a/b/beta/rms {:.1e} {:.1e} {:.1e} {:.1e}",
            l.kind.as_str(),
            e[0],
            e[1],
            e[2],
            e[3]
        );
        assert!(e.iter().all(|&v| v < 1e-7), "{name}: {e:?}");
    }
}

/// fit_law recovers a reel law.
#[test]
fn test_fit_law_reel() {
    let (v, h, r0) = (0.1905, 31.5e-6, 0.0288);
    let t = ts_dsp::window::linspace(0.0, 2400.0, 80);
    let fr: Vec<f64> = t
        .iter()
        .map(|&x| {
            v / (2.0 * std::f64::consts::PI * (r0 * r0 + v * h / std::f64::consts::PI * x).sqrt())
        })
        .collect();
    let l = fit_law(&t, &fr, None, Some((0.0, 2400.0)), 161);
    assert_eq!(l.kind, Kind::Chirp);
    let c = Comp::new("T", l.a, l.b, l.beta);
    let e = t
        .iter()
        .zip(&fr)
        .fold(0.0f64, |m, (&x, &y)| m.max((c.freq_at(x) - y).abs()));
    assert!(e < 2e-3, "{e}");
    // phase is the integral of frequency
    let tt = ts_dsp::window::linspace(0.0, 2400.0, 200001);
    let fq = c.freq(&tt);
    let mut ph = 0.0;
    let mut worst = 0.0f64;
    for i in 1..tt.len() {
        ph += (fq[i] + fq[i - 1]) / 2.0 * (tt[i] - tt[i - 1]);
        worst = worst.max((ph - c.phase_at(tt[i])).abs());
    }
    assert!(worst < 1e-6, "{worst}");
}

/// identify finds the mechanical components; the levels L1 < L2 < L3 improve the fit
/// (the same fit_hier call as the model-fill test in i_test_model.rs)
#[test]
fn test_identify_and_levels() {
    let (h, _, t, _, _) = run_case("mech300");
    let rt = ld("mech300", "rt");
    let fm: Vec<f64> = h.comps.iter().map(|c| c.f_mean).collect();
    let near = |v: &[f64], x: f64| v.iter().fold(f64::INFINITY, |m, a| m.min((a - x).abs()));
    for fx in [7.46, 1.484, 4.98] {
        assert!(near(&fm, fx) < 0.01, "{fx} {fm:?}");
    }
    let f0: Vec<f64> = h
        .comps
        .iter()
        .filter(|c| c.kind == Kind::Chirp)
        .map(|c| c.freq_at(0.0))
        .collect();
    assert!(near(&f0, 1.053) < 0.03, "take-up reel {f0:?}");
    assert!(near(&f0, 3.097) < 0.03, "slip clutch {f0:?}");
    let rms = |m: &model::Model| {
        let p = m.predict(&t);
        (p.iter()
            .zip(&rt)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / t.len() as f64)
            .sqrt()
    };
    let (e1, e2, e3) = (rms(&h.m1), rms(&h.m2), rms(&h.m3));
    assert!(e3 < 0.25 * e2 && 0.25 * e2 < 0.25 * e1, "{e1} {e2} {e3}");
    let mu = rt.iter().sum::<f64>() / rt.len() as f64;
    let var0 = rt.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / rt.len() as f64;
    assert!(1.0 - e3 * e3 / var0 > 0.95);
    let k = fm
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - 7.46).abs().partial_cmp(&(b.1 - 7.46).abs()).unwrap())
        .unwrap()
        .0;
    let model::Amp::L3(a) = &h.m3.amp[&(k, 1)] else {
        panic!("L3 amp")
    };
    let a3 = (a.iter().map(|z| z.norm_sqr()).sum::<f64>() / a.len() as f64).sqrt();
    assert!((a3 / 0.003 - 1.0).abs() < 0.2, "{a3}");
}

/// A sparse fit stays bounded (+ agreement of the prediction with the reference)
#[test]
fn test_sparse_fit_bounded() {
    let (t, r, w) = (ld("sparse", "t"), ld("sparse", "r"), ld("sparse", "w"));
    let mut c = Comp::new("cap", 6.8, 0.0, 0.0);
    c.harmonics = vec![1, 2];
    let o = FitOpts {
        level: 3,
        ta: 5.0,
        ..FitOpts::default()
    };
    let m = fit(&t, &r, &w, &[c], &o).unwrap();
    let p = m.predict(&t);
    assert!(p.iter().all(|v| v.is_finite()));
    assert!(p.iter().fold(0.0f64, |a, v| a.max((v - 0.98).abs())) < 0.005);
    let last = p[p.len() - 1];
    let tail = t
        .iter()
        .zip(&p)
        .filter(|&(&x, _)| x > 200.0)
        .fold(0.0f64, |a, (_, v)| a.max((v - last).abs()));
    assert!(tail < 1e-4, "{tail}");
    let pe = ld("sparse", "pred");
    let d = p
        .iter()
        .zip(&pe)
        .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
    let e = json(&data("sparse.json"));
    let (r0, dd, da, _) = cmp_model("sparse", &m, &e["m3"]);
    println!("sparse: pred max abs {d:.1e}, r0 {r0:.1e} drift {dd:.1e} amp {da:.1e}");
    assert!(d < 1e-9 && da < 1e-6, "{d} {da}");
}

#[test]
fn level_outliers_ref() {
    let t = ld("mech300", "t");
    let (r, w) = (ld("outliers", "r"), ld("outliers", "w"));
    let lo = level_outliers(&t, &r, &w, &model::OutlierOpts::default()).unwrap();
    let exp = ld_bool("outliers", "lo");
    assert!(exp.iter().any(|&v| v));
    assert_eq!(lo, exp);
}

/// RAYON_NUM_THREADS=1 ≡ default: fit_hier in a 1-thread pool is bit-identical.
#[test]
fn thread_determinism() {
    let case = "am120";
    let (h, ex, ..) = run_case(case);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (h1, ex1, ..) = pool.install(|| run_case(case));
    assert_eq!(h.comps, h1.comps);
    for k in ex.keys() {
        assert_eq!(ex[k].to_bits(), ex1[k].to_bits(), "{k}");
    }
    assert_eq!(h.m3.drift_coef, h1.m3.drift_coef);
    assert_eq!(h.m3.amp, h1.m3.amp);
    assert_eq!(h.m2.amp, h1.m2.amp);
}
