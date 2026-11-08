//! lstsq / roots / poly, BSpline design matrix + eval and the bounded minimizer against the
//! stored reference vectors (tests/data/g).
#[path = "../../testdata.rs"]
mod testdata;

use num_complex::Complex64;
use ts_dsp::{bspline, linalg, optimize};

fn ld(case: &str, name: &str) -> Vec<f64> {
    let p = testdata::dir("g").join(format!("{case}__{name}.npy"));
    let f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let npy = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    if npy.dtype().descr().contains("i8") {
        npy.into_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        npy.into_vec::<f64>().unwrap()
    }
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let s = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / s
}

#[test]
fn lstsq_matches_reference() {
    for (case, tol) in [
        ("tall", 1e-12),
        ("square", 1e-11),
        ("wide", 1e-12),
        ("rankdef", 1e-11),
        ("law", 1e-12),
        ("ar", 1e-6),
    ] {
        let c = format!("lstsq_{case}");
        let sh = ld(&c, "shape");
        let (m, n) = (sh[0] as usize, sh[1] as usize);
        let r = linalg::lstsq(&ld(&c, "a"), m, n, &ld(&c, "b"), None).unwrap();
        assert_eq!(r.rank, ld(&c, "rank")[0] as usize, "{case} rank");
        let sv = max_rel(&r.sv, &ld(&c, "sv"));
        assert!(sv < 1e-13, "{case} sv {sv:e}");
        let e = max_rel(&r.x, &ld(&c, "x"));
        // the "ar" case is a Hankel matrix of a sinusoid (rank 2 + 1e-9 noise): the solution
        // is set by singular values ~1e-9 → only the prediction is well defined
        if case == "ar" {
            let a = ld(&c, "a");
            let xp = ld(&c, "x");
            let pr = |x: &[f64]| -> Vec<f64> {
                (0..m)
                    .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
                    .collect()
            };
            let pe = max_rel(&pr(&r.x), &pr(&xp));
            println!("lstsq {case}: x rel {e:.2e}, prediction rel {pe:.2e}");
            assert!(pe < 1e-9, "{case} prediction {pe:e}");
        } else {
            println!("lstsq {case}: x rel {e:.2e}");
            assert!(e < tol, "{case} x {e:e}");
        }
    }
}

#[test]
fn roots_poly_matches_reference() {
    for i in 0..4 {
        let c = format!("roots{i}");
        let mut z = linalg::roots(&ld(&c, "p")).unwrap();
        // np.sort_complex: by real, then imaginary part
        z.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());
        let (re, im) = (ld(&c, "re"), ld(&c, "im"));
        assert_eq!(z.len(), re.len());
        for (k, v) in z.iter().enumerate() {
            let d = (v - Complex64::new(re[k], im[k])).norm();
            assert!(
                d < 1e-12 * (1.0 + v.norm()),
                "{c}[{k}]: {v} vs {} {}",
                re[k],
                im[k]
            );
        }
        let p: Vec<f64> = linalg::poly(&z).iter().map(|v| v.re).collect();
        let e = max_rel(&p, &ld(&c, "poly"));
        assert!(e < 1e-13, "{c} poly {e:e}");
    }
}

#[test]
fn bspline_matches_reference() {
    for i in 0..3 {
        let c = format!("bspl{i}");
        let (kn, x) = (ld(&c, "knots"), ld(&c, "x"));
        let (off, data) = bspline::design_matrix(&x, &kn, 3).unwrap();
        let (indptr, indices, d) = (ld(&c, "indptr"), ld(&c, "indices"), ld(&c, "data"));
        for r in 0..x.len() {
            let (a, b) = (indptr[r] as usize, indptr[r + 1] as usize);
            assert_eq!(b - a, 4);
            for q in 0..4 {
                assert_eq!(indices[a + q] as usize, off[r] + q, "{c} row {r}");
                assert_eq!(
                    data[a + q].to_bits(),
                    d[a + q].to_bits(),
                    "{c} row {r} col {q}"
                );
            }
        }
        let y = bspline::eval(&kn, &ld(&c, "c"), 3, &ld(&c, "xe"));
        for (k, (a, b)) in y.iter().zip(ld(&c, "ye")).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{c} eval {k}: {a} {b}"
            );
        }
    }
}

#[test]
fn fminbound_matches_reference() {
    type F = fn(f64) -> f64;
    let fs: [(&str, F); 5] = [
        ("quad", |x| (x - 0.31) * (x - 0.31) + 1.0),
        ("cos", f64::cos),
        ("edge", |x| x),
        ("flat", |_| 1.0),
        ("abs", |x| (x - 1e-4).abs()),
    ];
    for (name, f) in fs {
        let c = format!("fmin_{name}");
        let b = ld(&c, "bounds");
        let r = optimize::minimize_scalar_bounded(f, (b[0], b[1]), 1e-12, 500).unwrap();
        assert_eq!(r.nfev, ld(&c, "nfev")[0] as usize, "{name} nfev");
        assert_eq!(r.status as f64, ld(&c, "status")[0], "{name} status");
        assert_eq!(
            r.x.to_bits(),
            ld(&c, "x")[0].to_bits(),
            "{name} x {} vs {}",
            r.x,
            ld(&c, "x")[0]
        );
    }
}
