//! Shared helpers for the model tests: npy / JSON loading from tests/data/i and error metrics.
#![allow(dead_code)]
#[path = "../../../testdata.rs"]
pub mod testdata;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub fn data(name: &str) -> PathBuf {
    testdata::dir("i").join(name)
}

fn open_npy(p: &Path) -> npyz::NpyFile<std::io::BufReader<std::fs::File>> {
    let f = std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap()
}

/// Any real / bool / int npy as f64.
pub fn npy_f64(p: &Path) -> Vec<f64> {
    let npy = open_npy(p);
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
    } else if d.contains("u8") {
        npy.into_vec::<u64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else {
        npy.into_vec::<f64>().unwrap()
    }
}

pub fn npy_bool(p: &Path) -> Vec<bool> {
    npy_f64(p).into_iter().map(|v| v != 0.0).collect()
}

pub fn npy_i64(p: &Path) -> Vec<i64> {
    open_npy(p).into_vec::<i64>().unwrap()
}

pub fn npy_u64(p: &Path) -> Vec<u64> {
    open_npy(p).into_vec::<u64>().unwrap()
}

pub fn npy_u32(p: &Path) -> Vec<u32> {
    open_npy(p).into_vec::<u32>().unwrap()
}

pub fn json(p: &Path) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())),
    )
    .unwrap()
}

pub fn f(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) if s == "nan" => f64::NAN,
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        _ => panic!("not a number: {v}"),
    }
}

pub fn rms(a: &[f64]) -> f64 {
    (a.iter().map(|v| v * v).sum::<f64>() / a.len().max(1) as f64).sqrt()
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

pub fn mismatches(a: &[bool], b: &[bool]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}
