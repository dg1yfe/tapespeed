//! FIR / IIR / spline primitives against the stored reference vectors (tests/data/c).
//! Tolerances: designs 1e-12 relative to max |coef|, filter outputs 1e-10 max abs / rms(x).
#[path = "../../testdata.rs"]
mod testdata;
use rustfft::num_complex::Complex64;
use std::path::PathBuf;
use ts_dsp::{fir, iir, iir::PadType, spline};

fn path(case: &str, name: &str) -> PathBuf {
    testdata::dir("c").join(format!("{case}__{name}.npy"))
}
fn ld(case: &str, name: &str) -> Vec<f64> {
    let f = std::fs::File::open(path(case, name)).unwrap_or_else(|e| panic!("{case}/{name}: {e}"));
    let npy = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    match npy.dtype() {
        npyz::DType::Plain(t) if t.type_char() == npyz::TypeChar::Int => npy
            .into_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        _ => npy.into_vec::<f64>().unwrap(),
    }
}
fn ldc(case: &str, name: &str) -> Vec<Complex64> {
    let f = std::fs::File::open(path(case, name)).unwrap();
    npyz::NpyFile::new(std::io::BufReader::new(f))
        .unwrap()
        .into_vec::<num_complex::Complex<f64>>()
        .unwrap()
        .into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect()
}
fn sc(case: &str, name: &str) -> f64 {
    ld(case, name)[0]
}
fn exists(case: &str) -> bool {
    path(case, "x").exists()
        || path(case, "m").exists()
        || path(case, "n").exists()
        || path(case, "N").exists()
}
/// max |a-b| / max |b|
fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let s = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / s
}
/// max |a-b| / rms(x)
fn rel_rms(a: &[f64], b: &[f64], x: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let r = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
        / r
}
fn rel_rms_c(a: &[Complex64], b: &[Complex64], x: &[Complex64]) -> f64 {
    let r = (x.iter().map(|v| v.norm_sqr()).sum::<f64>() / x.len() as f64).sqrt();
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).norm())
        .fold(0.0, f64::max)
        / r
}

#[test]
fn i0_sinc() {
    let (x, y) = (ld("i0", "x"), ld("i0", "y"));
    let r: Vec<f64> = x.iter().map(|&v| fir::i0(v)).collect();
    for (a, b) in r.iter().zip(&y) {
        assert!(((a - b) / b).abs() < 1e-15, "{a} {b}");
    }
    let (x, y) = (ld("sinc", "x"), ld("sinc", "y"));
    let r: Vec<f64> = x.iter().map(|&v| fir::sinc(v)).collect();
    assert!(rel(&r, &y) < 1e-15);
}

#[test]
fn kaiser_kaiserord_firwin() {
    for i in 0..6 {
        let c = format!("kaiser{i}");
        let w = fir::kaiser(sc(&c, "m") as usize, sc(&c, "beta"));
        assert!(rel(&w, &ld(&c, "w")) < 1e-14, "{c}");
    }
    let t = ld("kaiserord", "tab");
    for r in t.chunks(4) {
        let (n, beta) = fir::kaiserord(r[0], r[1]).unwrap();
        assert_eq!(n as f64, r[2]);
        assert!((beta - r[3]).abs() <= 1e-15 * r[3].abs());
    }
    assert!(fir::kaiserord(7.0, 0.1).is_err());
    for c in ["firwin0", "firwin1", "firwin2", "firwin3", "firwin_even"] {
        let h = fir::firwin_kaiser(
            sc(c, "n") as usize,
            sc(c, "cutoff"),
            sc(c, "beta"),
            sc(c, "fs"),
        )
        .unwrap();
        let e = rel(&h, &ld(c, "h"));
        println!("{c}: n={} rel {e:.2e}", h.len());
        assert!(e < 1e-12, "{c} {e}");
    }
}

#[test]
fn fftconvolve_same() {
    for i in 0..4 {
        let c = format!("fconv{i}");
        let (x, h) = (ld(&c, "x"), ld(&c, "h"));
        let y = fir::fftconvolve_same(&x, &h);
        let e = rel_rms(&y, &ld(&c, "y"), &x);
        assert!(e < 1e-10, "{c} {e}");
    }
}

#[test]
fn notch_lfilter_filtfilt() {
    for i in 0..4 {
        let c = format!("notch{i}");
        let (b, a) = iir::iirnotch(sc(&c, "f0"), sc(&c, "q"), sc(&c, "fs")).unwrap();
        assert!(
            rel(&b, &ld(&c, "b")) < 1e-15 && rel(&a, &ld(&c, "a")) < 1e-15,
            "{c}"
        );
        let zi = iir::lfilter_zi(&b, &a).unwrap();
        assert!(rel(&zi, &ld(&c, "zi")) < 1e-12, "{c} zi");
        let x = ld(&c, "x");
        let z0: Vec<f64> = zi.iter().map(|v| v * 0.3).collect();
        let (y, zf) = iir::lfilter(&b, &a, &x, Some(&z0)).unwrap();
        assert!(rel_rms(&y, &ld(&c, "y"), &x) < 1e-10);
        assert!(rel(&zf, &ld(&c, "zf")) < 1e-10);
        let ff = iir::filtfilt(&b, &a, &x, PadType::Odd, None).unwrap();
        let e = rel_rms(&ff, &ld(&c, "ff"), &x);
        assert!(e < 1e-10, "{c} ff {e}");
        let ff10 = iir::filtfilt(&b, &a, &x[..10], PadType::Odd, None).unwrap();
        assert!(rel_rms(&ff10, &ld(&c, "ff10"), &x[..10]) < 1e-10);
        let err = iir::filtfilt(&b, &a, &x[..9], PadType::Odd, None).unwrap_err();
        assert!(err.0.contains("greater than padlen, which is 9"));
    }
    let c = "lfilt_gen";
    let (b, a, x) = (ld(c, "b"), ld(c, "a"), ld(c, "x"));
    assert!(rel(&iir::lfilter_zi(&b, &a).unwrap(), &ld(c, "zi")) < 1e-12);
    assert!(rel_rms(&iir::lfilter(&b, &a, &x, None).unwrap().0, &ld(c, "y"), &x) < 1e-10);
    assert!(
        rel_rms(
            &iir::filtfilt(&b, &a, &x, PadType::Odd, None).unwrap(),
            &ld(c, "ff"),
            &x
        ) < 1e-10
    );
    assert!(
        rel_rms(
            &iir::filtfilt(&b, &a, &x, PadType::Even, Some(20)).unwrap(),
            &ld(c, "ffe"),
            &x
        ) < 1e-10
    );
}

#[test]
fn butter_sosfiltfilt() {
    for i in 0..8 {
        let c = format!("butter{i}");
        if !exists(&c) {
            continue;
        }
        let sos =
            iir::butter_lowpass_sos(sc(&c, "N") as usize, sc(&c, "fc"), sc(&c, "fs")).unwrap();
        let rs = ld(&c, "sos");
        let flat: Vec<f64> = sos.iter().flatten().copied().collect();
        let e = rel(&flat, &rs);
        println!("{c}: N={} sos rel {e:.2e}", sc(&c, "N"));
        assert!(e < 1e-12, "{c} sos {e}");
        let zi: Vec<f64> = iir::sosfilt_zi(&sos)
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect();
        assert!(rel(&zi, &ld(&c, "zi")) < 1e-12, "{c} zi");
        let x = ld(&c, "x");
        let xc = ldc(&c, "xc");
        let (y, _) = iir::sosfilt(&sos, &x, None).unwrap();
        assert!(rel_rms(&y, &ld(&c, "y"), &x) < 1e-10);
        let ff = iir::sosfiltfilt(&sos, &x, PadType::Odd, None).unwrap();
        assert!(rel_rms(&ff, &ld(&c, "ff"), &x) < 1e-10, "{c} ff");
        let ffe = iir::sosfiltfilt(&sos, &x, PadType::Even, Some(37)).unwrap();
        assert!(rel_rms(&ffe, &ld(&c, "ffe"), &x) < 1e-10, "{c} ffe");
        let ffc = iir::sosfiltfilt(&sos, &xc, PadType::Odd, None).unwrap();
        assert!(rel_rms_c(&ffc, &ldc(&c, "ffc"), &xc) < 1e-10, "{c} ffc");
        let ffec = iir::sosfiltfilt(&sos, &xc, PadType::Even, Some(299)).unwrap();
        assert!(
            rel_rms_c(&ffec, &ldc(&c, "ffe_c"), &xc) < 1e-10,
            "{c} ffe_c"
        );
        assert!(iir::sosfiltfilt(&sos, &xc, PadType::Even, Some(300)).is_err());
    }
    let c = "lpcurve";
    let (t, r) = (ld(c, "t"), ld(c, "r"));
    let dt = t[1] - t[0];
    let sos = iir::butter_lowpass_sos(4, sc(c, "fc"), 1.0 / dt).unwrap();
    let m = r.iter().sum::<f64>() / r.len() as f64; // np.mean ~ pairwise; compare loosely
    assert!((m - sc(c, "m")).abs() < 1e-14);
    let m = sc(c, "m");
    let pl = sc(c, "padlen") as usize;
    let d: Vec<f64> = r.iter().map(|v| v - m).collect();
    let y: Vec<f64> = iir::sosfiltfilt(&sos, &d, PadType::Even, Some(pl))
        .unwrap()
        .iter()
        .map(|v| m + v)
        .collect();
    assert!(rel_rms(&y, &ld(c, "y"), &d) < 1e-10);
}

#[test]
fn cubic_spline_ppoly() {
    for c in [
        "spline0",
        "spline1",
        "spline2",
        "spline3",
        "spline4",
        "spline5",
        "spline_tm",
    ] {
        let (x, y) = (ld(c, "x"), ld(c, "y"));
        let sp = spline::cubic_spline(&x, &y).unwrap();
        let ad = sp.antiderivative();
        let cs: Vec<f64> = sp.coeffs().concat();
        let ca: Vec<f64> = ad.coeffs().concat();
        let e1 = rel(&cs, &ld(c, "c"));
        let e2 = rel(&ca, &ld(c, "ca"));
        let xe = ld(c, "xe");
        let ye: Vec<f64> = xe.iter().map(|&v| sp.eval(v)).collect();
        let yae: Vec<f64> = xe.iter().map(|&v| ad.eval(v)).collect();
        let e3 = rel(&ye, &ld(c, "ye"));
        let e4 = rel(&yae, &ld(c, "yae"));
        println!(
            "{c}: n={} coef {e1:.1e} anti {e2:.1e} eval {e3:.1e} anti-eval {e4:.1e}",
            x.len()
        );
        assert!(e1 < 1e-12 && e2 < 1e-12 && e3 < 1e-12 && e4 < 1e-12, "{c}");
    }
}
