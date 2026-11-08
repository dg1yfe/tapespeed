//! Shared helpers for the correct() tests: .npy loading (tests/data/c) and error metrics.
#![allow(dead_code)]
#[path = "../../../testdata.rs"]
pub mod testdata;
use std::path::PathBuf;

pub fn path(case: &str, name: &str) -> PathBuf {
    testdata::dir("c").join(format!("{case}__{name}.npy"))
}

/// (shape, data as f64) for f8/f4/i8 arrays.
pub fn ld(case: &str, name: &str) -> (Vec<usize>, Vec<f64>) {
    let f = std::fs::File::open(path(case, name)).unwrap_or_else(|e| panic!("{case}/{name}: {e}"));
    let npy = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    let shape: Vec<usize> = npy.shape().iter().map(|&s| s as usize).collect();
    let descr = npy.dtype().descr();
    let v = if descr.contains("f4") {
        npy.into_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    } else if descr.contains("i8") {
        npy.into_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        npy.into_vec::<f64>().unwrap()
    };
    (shape, v)
}

pub fn ld_f32(case: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
    let f = std::fs::File::open(path(case, name)).unwrap();
    let npy = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    let shape: Vec<usize> = npy.shape().iter().map(|&s| s as usize).collect();
    (shape, npy.into_vec::<f32>().unwrap())
}

/// (RMS diff re RMS(reference) in dB, max abs diff)
pub fn cmp(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let (mut e2, mut r2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (&p, &q) in a.iter().zip(b) {
        let d = p as f64 - q as f64;
        e2 += d * d;
        r2 += (q as f64) * (q as f64);
        mx = mx.max(d.abs());
    }
    (10.0 * (e2 / r2.max(1e-300)).log10(), mx)
}

pub fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| (*p as f64 - *q as f64).abs())
        .fold(0.0, f64::max)
}
