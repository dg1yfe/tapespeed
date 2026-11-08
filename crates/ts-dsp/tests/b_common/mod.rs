//! Test support for the primitive reference tests: reads the stored reference vectors (`.bpk`
//! packs with entries `<case>/<name>.npy` + `<case>/params.txt`, from `$TS_PRIM_B` or
//! tests/data/b) and compares at the tolerances below.
#![allow(dead_code)]
#[path = "../../../testdata.rs"]
pub mod testdata;

use std::collections::BTreeMap;
use std::path::PathBuf;

use num_complex::Complex64;

#[derive(Debug, Clone)]
pub enum Arr {
    F64(Vec<f64>),
    F32(Vec<f32>),
    C128(Vec<Complex64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Arr,
}

fn parse_npy(b: &[u8]) -> Npy {
    assert_eq!(&b[..6], b"\x93NUMPY", "npy magic");
    let major = b[6];
    let (hlen, off) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[off..off + hlen]).unwrap();
    let data = &b[off + hlen..];
    let descr = header.split("'descr':").nth(1).unwrap();
    let descr = descr.split('\'').nth(1).unwrap();
    assert!(header.contains("'fortran_order': False"));
    let shp = header.split("'shape':").nth(1).unwrap();
    let shp = &shp[shp.find('(').unwrap() + 1..shp.find(')').unwrap()];
    let shape: Vec<usize> = shp
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let n: usize = shape.iter().product();
    let f8 = |i: usize| f64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap());
    let arr = match descr {
        "<f8" => Arr::F64((0..n).map(f8).collect()),
        "<f4" => Arr::F32(
            (0..n)
                .map(|i| f32::from_le_bytes(data[4 * i..4 * i + 4].try_into().unwrap()))
                .collect(),
        ),
        "<c16" => Arr::C128(
            (0..n)
                .map(|i| Complex64::new(f8(2 * i), f8(2 * i + 1)))
                .collect(),
        ),
        "<i8" => Arr::I64(
            (0..n)
                .map(|i| i64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap()))
                .collect(),
        ),
        "|b1" => Arr::Bool(data[..n].iter().map(|&v| v != 0).collect()),
        d => panic!("unsupported dtype {d}"),
    };
    Npy { shape, data: arr }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub id: String,
    pub arrays: BTreeMap<String, Npy>,
    pub params: BTreeMap<String, String>,
}

impl Case {
    pub fn arr(&self, name: &str) -> &Npy {
        self.arrays
            .get(name)
            .unwrap_or_else(|| panic!("case {} has no array {name}", self.id))
    }
    pub fn f64(&self, name: &str) -> Vec<f64> {
        match &self.arr(name).data {
            Arr::F64(v) => v.clone(),
            Arr::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Arr::F32(v) => v.iter().map(|&x| x as f64).collect(),
            Arr::Bool(v) => v.iter().map(|&x| x as u8 as f64).collect(),
            Arr::C128(_) => panic!("{name} is complex"),
        }
    }
    pub fn scalar(&self, name: &str) -> f64 {
        self.f64(name)[0]
    }
    pub fn c128(&self, name: &str) -> Vec<Complex64> {
        match &self.arr(name).data {
            Arr::C128(v) => v.clone(),
            Arr::F64(v) => v.iter().map(|&x| Complex64::new(x, 0.0)).collect(),
            _ => panic!("{name} not complex"),
        }
    }
    pub fn i64(&self, name: &str) -> Vec<i64> {
        match &self.arr(name).data {
            Arr::I64(v) => v.clone(),
            _ => panic!("{name} not int"),
        }
    }
    pub fn bools(&self, name: &str) -> Vec<bool> {
        match &self.arr(name).data {
            Arr::Bool(v) => v.clone(),
            _ => panic!("{name} not bool"),
        }
    }
    pub fn p(&self, key: &str) -> &str {
        self.params
            .get(key)
            .unwrap_or_else(|| panic!("case {} has no param {key}", self.id))
    }
    pub fn pf(&self, key: &str) -> f64 {
        self.p(key).parse().unwrap()
    }
    pub fn pu(&self, key: &str) -> usize {
        self.p(key).parse().unwrap()
    }
}

pub fn data_dir() -> PathBuf {
    std::env::var_os("TS_PRIM_B")
        .map(PathBuf::from)
        .unwrap_or_else(|| testdata::dir("b"))
}

/// All cases of one function pack, in case order.
pub fn load(func: &str) -> Vec<Case> {
    let path = data_dir().join(format!("{func}.bpk"));
    let b = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {e} (stored reference vectors missing)", path.display()));
    let mut cases: BTreeMap<String, Case> = BTreeMap::new();
    let mut i = 0;
    while i < b.len() {
        let nl = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        let name = std::str::from_utf8(&b[i + 4..i + 4 + nl])
            .unwrap()
            .to_string();
        i += 4 + nl;
        let sz = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize;
        i += 8;
        let body = &b[i..i + sz];
        i += sz;
        let (cid, entry) = name.split_once('/').unwrap();
        let c = cases.entry(cid.to_string()).or_insert_with(|| Case {
            id: format!("{func}/{cid}"),
            arrays: BTreeMap::new(),
            params: BTreeMap::new(),
        });
        if entry == "params.txt" {
            for line in std::str::from_utf8(body).unwrap().lines() {
                if let Some((k, v)) = line.split_once(' ') {
                    c.params.insert(k.to_string(), v.to_string());
                }
            }
        } else {
            let key = entry.strip_suffix(".npy").unwrap().to_string();
            c.arrays.insert(key, parse_npy(body));
        }
    }
    assert!(!cases.is_empty(), "{func}: no cases");
    cases.into_values().collect()
}

// ---------------------------------------------------------------- comparisons --------

/// Max |a - b| (NaN must match NaN position; ±inf must match exactly).
pub fn max_abs_diff(a: &[f64], b: &[f64], ctx: &str) -> f64 {
    assert_eq!(a.len(), b.len(), "{ctx}: length {} vs {}", a.len(), b.len());
    let mut m = 0.0f64;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if y.is_nan() || x.is_nan() {
            assert!(x.is_nan() && y.is_nan(), "{ctx}[{i}]: got {x}, want {y}");
        } else if y.is_infinite() || x.is_infinite() {
            assert_eq!(x, y, "{ctx}[{i}]");
        } else {
            m = m.max((x - y).abs());
        }
    }
    m
}

pub fn max_abs(v: &[f64]) -> f64 {
    v.iter()
        .filter(|x| x.is_finite())
        .fold(0.0, |a, &b| a.max(b.abs()))
}

pub fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

pub fn rms_c(v: &[Complex64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x.norm_sqr()).sum::<f64>() / v.len() as f64).sqrt()
}

pub fn reim(v: &[Complex64]) -> Vec<f64> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

/// Design tolerance: max |a-b| / max |b| (1e-12).
pub const DESIGN_REL: f64 = 1e-12;
/// Output tolerance: max |a-b| / rms(input) (1e-10).
pub const OUT_REL: f64 = 1e-10;

/// Tracks the worst error per function and prints it (run with `--nocapture`).
pub struct Worst {
    pub name: &'static str,
    pub err: f64,
    pub n: usize,
}

impl Worst {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            err: 0.0,
            n: 0,
        }
    }
    pub fn add(&mut self, e: f64, tol: f64, ctx: &str) {
        assert!(e <= tol, "{ctx}: error {e:e} > {tol:e}");
        self.err = self.err.max(e);
        self.n += 1;
    }
}

impl Drop for Worst {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            eprintln!(
                "[ref] {:<22} cases {:>4}  max err {:.3e}",
                self.name, self.n, self.err
            );
        }
    }
}
