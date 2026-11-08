//! Test helpers for the analyze tests: minimal .npy reader (f8/c16/b1/i8/f4, little endian,
//! C order).
#![allow(dead_code)]
use num_complex::Complex64;
use std::path::Path;

pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub fn read_npy(p: &Path) -> Npy {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}", p.display());
    let (hl, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let h = std::str::from_utf8(&b[off..off + hl]).unwrap();
    let descr = h
        .split("'descr':")
        .nth(1)
        .unwrap()
        .split('\'')
        .nth(1)
        .unwrap()
        .to_string();
    assert!(h.contains("'fortran_order': False"), "{}", p.display());
    let sh = h.split("'shape':").nth(1).unwrap();
    let sh = &sh[sh.find('(').unwrap() + 1..sh.find(')').unwrap()];
    let shape = sh
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Npy {
        descr,
        shape,
        data: b[off + hl..].to_vec(),
    }
}

fn f8s(d: &[u8]) -> Vec<f64> {
    d.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn ld_f64(p: &Path) -> Vec<f64> {
    let n = read_npy(p);
    match n.descr.as_str() {
        "<f8" => f8s(&n.data),
        "<f4" => n
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        "<i8" => n
            .data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        "|b1" => n.data.iter().map(|&v| v as f64).collect(),
        d => panic!("{}: dtype {d}", p.display()),
    }
}

pub fn ld_c128(p: &Path) -> Vec<Complex64> {
    let n = read_npy(p);
    assert_eq!(n.descr, "<c16", "{}", p.display());
    f8s(&n.data)
        .chunks_exact(2)
        .map(|c| Complex64::new(c[0], c[1]))
        .collect()
}
