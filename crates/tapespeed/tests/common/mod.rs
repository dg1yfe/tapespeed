//! Shared test helpers: the CLI binary, per-test temp directories, curve metadata.
#![allow(dead_code)]
use std::path::{Path, PathBuf};
use ts_core::curve::Curve;

/// A fresh scratch directory under the cargo target tmp dir.
pub fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run the CLI; returns (status, stdout, stderr).
pub fn cli(args: &[&str]) -> (i32, String, String) {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_tapespeed"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

pub fn meta_str(c: &Curve, k: &str) -> String {
    c.meta
        .get(k)
        .map(|v| v.to_text())
        .unwrap_or_else(|| panic!("meta {k} missing"))
}

pub fn mean_b(v: &[bool]) -> f64 {
    v.iter().filter(|&&b| b).count() as f64 / v.len() as f64
}
