//! Convolution, spectral, filter, peak, interpolation and statistics primitives against the
//! stored reference vectors (tests/data/b).
//! `cargo test -p ts-dsp --test prims_ref -- --nocapture` prints the worst error per function.

mod b_common;

use b_common::*;
use num_complex::Complex64;
use ts_dsp::conv::{self, ConvMode};
use ts_dsp::ndimage::{self, Mode};
use ts_dsp::{interp, peaks, spectral, stats, window};

fn conv_mode(s: &str) -> ConvMode {
    match s {
        "full" => ConvMode::Full,
        "same" => ConvMode::Same,
        "valid" => ConvMode::Valid,
        _ => panic!("{s}"),
    }
}

#[test]
fn windows() {
    let mut w = Worst::new("window");
    for c in load("window") {
        let n = c.pu("n");
        let got = match c.p("kind") {
            "hann_periodic" => window::hann_periodic(n),
            "hann_symmetric" => window::hann_symmetric(n),
            "hanning" => window::hanning(n),
            "kaiser" => window::kaiser(n, c.pf("beta"), c.p("sym") == "1"),
            k => panic!("{k}"),
        };
        let want = c.f64("out");
        let e = max_abs_diff(&got, &want, &c.id) / max_abs(&want).max(1e-300);
        w.add(e, DESIGN_REL, &c.id);
    }
}

#[test]
fn fftconvolve() {
    let mut w = Worst::new("fftconvolve");
    for c in load("fftconvolve") {
        let (a, v) = (c.f64("a"), c.f64("v"));
        let got = conv::fftconvolve(&a, &v, conv_mode(c.p("mode")));
        let e = max_abs_diff(&got, &c.f64("out"), &c.id)
            / (rms(&a) * rms(&v) * v.len() as f64).max(1e-300);
        w.add(e, OUT_REL, &c.id);
    }
    let mut w = Worst::new("oaconvolve (complex)");
    for c in load("oaconvolve_c") {
        let (a, v) = (c.c128("a"), c.f64("v"));
        let got = conv::oaconvolve_c(&a, &v, conv_mode(c.p("mode")));
        let e = max_abs_diff(&reim(&got), &reim(&c.c128("out")), &c.id)
            / (rms_c(&a) * rms(&v) * v.len() as f64).max(1e-300);
        w.add(e, OUT_REL, &c.id);
    }
    let mut w = Worst::new("fftconvolve axes=0");
    for c in load("fftconvolve_axis0") {
        let (d, k) = (c.f64("data"), c.f64("ker"));
        let (got, nout) =
            conv::fftconvolve_axis0(&d, c.pu("nrows"), c.pu("ncols"), &k, ConvMode::Same).unwrap();
        assert_eq!(nout, c.pu("nrows"));
        let e = max_abs_diff(&got, &c.f64("out"), &c.id) / rms(&d);
        w.add(e, OUT_REL, &c.id);
    }
}

#[test]
fn upfirdn() {
    let mut w = Worst::new("upfirdn");
    for c in load("upfirdn") {
        let (h, x) = (c.f64("h"), c.f64("x"));
        let got = conv::upfirdn(&h, &x, c.pu("up"), c.pu("down")).unwrap();
        let e = max_abs_diff(&got, &c.f64("out"), &c.id) / (rms(&x) * rms(&h) * h.len() as f64);
        w.add(e, OUT_REL, &c.id);
    }
}

#[test]
fn resample_poly() {
    let mut w = Worst::new("resample_poly (cplx)");
    for c in load("resample_poly_c") {
        let (x, h) = (c.c128("x"), c.f64("h"));
        let got = conv::resample_poly_down_c(&x, c.pu("down"), &h).unwrap();
        let want = c.c128("out");
        assert_eq!(
            got.len(),
            conv::resample_poly_len(x.len(), c.pu("down")),
            "{}",
            c.id
        );
        let e =
            max_abs_diff(&reim(&got), &reim(&want), &c.id) / (rms_c(&x) * rms(&h) * h.len() as f64);
        w.add(e, OUT_REL, &c.id);
    }
    let mut w = Worst::new("resample_poly (real)");
    for c in load("resample_poly") {
        let (x, h) = (c.f64("x"), c.f64("h"));
        let d = c.pu("down");
        if c.params.contains_key("default") && d > 1 {
            let (nt, cut, beta) = conv::resample_poly_default_spec(d);
            assert_eq!((nt, cut, beta), (h.len(), 1.0 / d as f64, 5.0));
        }
        let got = conv::resample_poly_down(&x, d, &h).unwrap();
        let e = max_abs_diff(&got, &c.f64("out"), &c.id) / (rms(&x) * rms(&h) * h.len() as f64);
        w.add(e, OUT_REL, &c.id);
    }
}

#[test]
fn uniform_filter1d() {
    let mut w = Worst::new("uniform_filter1d");
    for c in load("uniform_filter1d") {
        let x = c.f64("x");
        let mode = match c.p("mode") {
            "nearest" => Mode::Nearest,
            "reflect" => Mode::Reflect,
            "constant" => Mode::Constant(0.0),
            m => panic!("{m}"),
        };
        let got = ndimage::uniform_filter1d(&x, c.pu("size"), mode).unwrap();
        // bit-exact against the reference vectors (same recurrence)
        w.add(max_abs_diff(&got, &c.f64("out"), &c.id), 0.0, &c.id);
    }
}

#[test]
fn median_filters() {
    let mut w = Worst::new("median_filter");
    for c in load("median_filter") {
        let got = ndimage::median_filter(&c.f64("x"), c.pu("size"), Mode::Nearest).unwrap();
        w.add(max_abs_diff(&got, &c.f64("out"), &c.id), 0.0, &c.id);
    }
    let mut w = Worst::new("medfilt");
    for c in load("medfilt") {
        let got = ndimage::medfilt(&c.f64("x"), c.pu("size")).unwrap();
        w.add(max_abs_diff(&got, &c.f64("out"), &c.id), 0.0, &c.id);
    }
    assert!(ndimage::medfilt(&[1.0, 2.0], 4).is_err());
}

#[test]
fn binary_morphology() {
    let mut w = Worst::new("binary morphology");
    for c in load("binary") {
        let (x, n) = (c.bools("x"), c.pu("n"));
        assert_eq!(
            ndimage::binary_dilation(&x, n).unwrap(),
            c.bools("dil"),
            "{} dil",
            c.id
        );
        assert_eq!(
            ndimage::binary_erosion(&x, n).unwrap(),
            c.bools("ero"),
            "{} ero",
            c.id
        );
        assert_eq!(
            ndimage::binary_opening(&x, n).unwrap(),
            c.bools("ope"),
            "{} ope",
            c.id
        );
        assert_eq!(
            ndimage::binary_closing(&x, n).unwrap(),
            c.bools("clo"),
            "{} clo",
            c.id
        );
        w.add(0.0, 0.0, &c.id);
    }
}

#[test]
fn welch() {
    let mut w = Worst::new("welch");
    for c in load("welch") {
        let x = c.f64("x");
        let (f, p) = spectral::welch(&x, c.pf("fs"), c.pu("nperseg")).unwrap();
        assert_eq!(max_abs_diff(&f, &c.f64("f"), &c.id), 0.0, "{} f", c.id);
        let want = c.f64("p");
        let e = max_abs_diff(&p, &want, &c.id) / max_abs(&want).max(rms(&x).powi(2) / c.pf("fs"));
        w.add(e, OUT_REL, &c.id);
    }
}

#[test]
fn stft() {
    let mut w = Worst::new("stft");
    for c in load("stft") {
        let x = c.f64("x");
        let nov = c.p("noverlap").parse::<i64>().unwrap();
        let nov = (nov >= 0).then_some(nov as usize);
        let s = spectral::stft(&x, c.pf("fs"), c.pu("nperseg"), nov).unwrap();
        let shape = &c.arr("z").shape;
        assert_eq!((s.nfreq, s.nframes), (shape[0], shape[1]), "{}", c.id);
        assert_eq!(max_abs_diff(&s.f, &c.f64("f"), &c.id), 0.0, "{} f", c.id);
        assert!(
            max_abs_diff(&s.t, &c.f64("t"), &c.id) <= 1e-15 * max_abs(&s.t),
            "{} t",
            c.id
        );
        let e = max_abs_diff(&reim(&s.z), &reim(&c.c128("z")), &c.id) / rms(&x);
        w.add(e, OUT_REL, &c.id);
    }
    // float32 input: the reference is computed in complex64, `stft` in f64 (see its docs)
    let mut w = Worst::new("stft f32 input (f64 math)");
    for c in load("stft_f32") {
        let x = c.f64("x");
        let s = spectral::stft(&x, c.pf("fs"), c.pu("nperseg"), Some(c.pu("noverlap"))).unwrap();
        let e = max_abs_diff(&reim(&s.z), &reim(&c.c128("z")), &c.id) / rms(&x);
        w.add(e, 1e-6, &c.id);
    }
}

#[test]
fn rfft_frames() {
    let mut w = Worst::new("rfft_frames");
    for c in load("rfft_frames") {
        let (x, win) = (c.f64("x"), c.f64("win"));
        let (got, nf) = spectral::rfft_frames(&x, &win, c.pu("hop"), c.pu("nfft")).unwrap();
        assert_eq!(nf, c.arr("out").shape[0], "{}", c.id);
        let e = max_abs_diff(&reim(&got), &reim(&c.c128("out")), &c.id) / rms(&x);
        w.add(e, OUT_REL, &c.id);
    }
}

#[test]
fn fftfreq() {
    let mut w = Worst::new("fftfreq/rfftfreq");
    for c in load("fftfreq") {
        let (n, d) = (c.pu("n"), c.pf("d"));
        assert_eq!(
            max_abs_diff(&spectral::fftfreq(n, d), &c.f64("f"), &c.id),
            0.0
        );
        assert_eq!(
            max_abs_diff(&spectral::rfftfreq(n, d), &c.f64("rf"), &c.id),
            0.0
        );
        let sh = spectral::fftshift(&spectral::fftfreq(n, d));
        assert_eq!(max_abs_diff(&sh, &c.f64("sh"), &c.id), 0.0);
        w.add(0.0, 0.0, &c.id);
    }
}

#[test]
fn find_peaks() {
    let mut w = Worst::new("find_peaks");
    let heights = [None, Some(0.0), Some(1.0)];
    let dists = [None, Some(1.0), Some(2.0), Some(5.0), Some(50.5)];
    for c in load("find_peaks") {
        let x = c.f64("x");
        for (hi, h) in heights.iter().enumerate() {
            for (di, d) in dists.iter().enumerate() {
                let got: Vec<i64> = peaks::find_peaks(&x, *h, *d)
                    .unwrap()
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                assert_eq!(
                    got,
                    c.i64(&format!("pk_{hi}_{di}")),
                    "{} h={h:?} d={d:?}",
                    c.id
                );
                w.add(0.0, 0.0, &c.id);
            }
        }
    }
}

#[test]
fn argsort_tie_order() {
    let mut w = Worst::new("argsort (numpy order)");
    for c in load("argsort") {
        let got: Vec<i64> = stats::argsort(&c.f64("x"))
            .into_iter()
            .map(|v| v as i64)
            .collect();
        assert_eq!(got, c.i64("out"), "{} ({})", c.id, c.p("kind"));
        w.add(0.0, 0.0, &c.id);
    }
}

#[test]
fn stats_reductions() {
    let mut w_sum = Worst::new("sum/mean/std (bits)");
    let mut w = Worst::new("median/perc/ptp/arg*");
    let mut w_dt = Worst::new("detrend");
    for c in load("stats") {
        let x = c.f64("x");
        let id = &c.id;
        let bits = |a: f64, b: f64, what: &str| {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{id} {what}: {a:e} vs {b:e}"
            );
        };
        bits(stats::np_sum(&x), c.scalar("sum"), "sum");
        bits(stats::mean(&x), c.scalar("mean"), "mean");
        let sd = stats::std(&x);
        let e = if sd.is_nan() {
            0.0
        } else {
            (sd - c.scalar("std")).abs() / c.scalar("std").max(1e-300)
        };
        w_sum.add(e, 0.0, id);
        let exact = |a: f64, b: f64, what: &str| {
            assert!(
                a == b || (a.is_nan() && b.is_nan()),
                "{id} {what}: {a:e} vs {b:e}"
            );
        };
        exact(stats::median(&x), c.scalar("median"), "median");
        exact(stats::nanmedian(&x), c.scalar("nanmedian"), "nanmedian");
        let pq = stats::percentiles(&x, &c.f64("q")).unwrap();
        for (a, b) in pq.iter().zip(c.f64("perc")) {
            exact(*a, b, "percentile");
        }
        exact(stats::ptp(&x).unwrap(), c.scalar("ptp"), "ptp");
        assert_eq!(
            stats::argmax(&x).unwrap() as i64,
            c.i64("argmax")[0],
            "{id} argmax"
        );
        assert_eq!(
            stats::argmin(&x).unwrap() as i64,
            c.i64("argmin")[0],
            "{id} argmin"
        );
        w.add(0.0, 0.0, id);
        let dc = stats::detrend_constant(&x);
        let e = max_abs_diff(&dc, &c.f64("dcon"), id);
        let want = c.f64("dlin");
        if want.is_empty() {
            assert!(stats::detrend_linear(&x).is_err());
            w_dt.add(e, 0.0, id);
        } else {
            let got = stats::detrend_linear(&x).unwrap();
            let e2 = max_abs_diff(&got, &want, id) / rms(&x).max(1e-300);
            w_dt.add(e.max(e2), OUT_REL, id);
        }
    }
    for c in load("stats_nanmedian_allnan") {
        assert!(stats::nanmedian(&c.f64("x")).is_nan() && c.scalar("out").is_nan());
    }
}

#[test]
fn interp_numpy() {
    let mut w = Worst::new("np.interp");
    for c in load("interp") {
        let got = stats::interp(&c.f64("x"), &c.f64("xp"), &c.f64("fp")).unwrap();
        let fp = c.f64("fp");
        let e = max_abs_diff(&got, &c.f64("out"), &c.id) / max_abs(&fp).max(1e-300);
        w.add(e, 1e-15, &c.id);
    }
}

#[test]
fn corr_spearman() {
    let mut w = Worst::new("corrcoef/spearmanr");
    for c in load("corr") {
        let (a, b) = (c.f64("a"), c.f64("b"));
        let id = &c.id;
        let ra = stats::rankdata(&a);
        assert_eq!(max_abs_diff(&ra, &c.f64("ranka"), id), 0.0, "{id} rankdata");
        let mut e: f64 = 0.0;
        for (got, want) in [
            (stats::corrcoef(&a, &b).unwrap(), c.scalar("corrcoef")),
            (stats::spearmanr(&a, &b).unwrap(), c.scalar("spearman")),
        ] {
            assert_eq!(got.is_nan(), want.is_nan(), "{id}: {got} vs {want}");
            if !want.is_nan() {
                e = e.max((got - want).abs());
            }
        }
        w.add(e, 1e-12, id);
    }
}

#[test]
fn pchip() {
    let mut w = Worst::new("PchipInterpolator");
    for c in load("pchip") {
        let (x, y) = (c.f64("x"), c.f64("y"));
        let p = interp::Pchip::new(&x, &y).unwrap();
        // coefficients (design): shape (4, n-1)
        let cw = c.f64("c");
        let m = x.len() - 1;
        let got: Vec<f64> = p.coeffs().iter().flat_map(|r| r.iter().copied()).collect();
        assert_eq!(got.len(), 4 * m);
        let ec = max_abs_diff(&got, &cw, &c.id) / max_abs(&cw).max(1e-300);
        let out = p.eval(&c.f64("xi"));
        let eo = max_abs_diff(&out, &c.f64("out"), &c.id) / max_abs(&y).max(1e-300);
        w.add(ec.max(eo), DESIGN_REL, &c.id);
    }
    assert!(interp::Pchip::new(&[1.0], &[1.0]).is_err());
    assert!(interp::Pchip::new(&[1.0, 1.0], &[1.0, 2.0]).is_err());
    let _ = Complex64::new(0.0, 0.0);
}
