//! curve: speed-curve container and CSV I/O (`# tapespeed-curve v1`).
use indexmap::IndexMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use ts_io::{fmt_fixed, repr_shortest};

/// A header (meta) value; [`MetaValue::to_text`] gives its text form in the CSV header.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl MetaValue {
    /// Text form: integers without `.0`, floats as shortest round-trip repr, `True`/`False`.
    pub fn to_text(&self) -> String {
        match self {
            MetaValue::Str(s) => s.clone(),
            MetaValue::Int(i) => i.to_string(),
            MetaValue::Float(x) => repr_shortest(*x),
            MetaValue::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
        }
    }
}

impl From<&str> for MetaValue {
    fn from(v: &str) -> Self {
        MetaValue::Str(v.into())
    }
}
impl From<String> for MetaValue {
    fn from(v: String) -> Self {
        MetaValue::Str(v)
    }
}
impl From<i64> for MetaValue {
    fn from(v: i64) -> Self {
        MetaValue::Int(v)
    }
}
impl From<usize> for MetaValue {
    fn from(v: usize) -> Self {
        MetaValue::Int(v as i64)
    }
}
impl From<f64> for MetaValue {
    fn from(v: f64) -> Self {
        MetaValue::Float(v)
    }
}
impl From<bool> for MetaValue {
    fn from(v: bool) -> Self {
        MetaValue::Bool(v)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Curve {
    /// [s], uniform grid
    pub t: Vec<f64>,
    /// apparent / nominal
    pub ratio: Vec<f64>,
    /// 0..1
    pub confidence: Vec<f64>,
    pub valid: Vec<bool>,
    /// insertion-ordered header
    pub meta: IndexMap<String, MetaValue>,
}

impl Curve {
    /// `np.interp(t, self.t, self.ratio)` (clamped at both ends).
    pub fn ratio_at(&self, t: f64) -> f64 {
        let (xs, ys) = (&self.t, &self.ratio);
        let n = xs.len();
        if n == 0 {
            return f64::NAN;
        }
        if t <= xs[0] {
            return ys[0];
        }
        if t >= xs[n - 1] {
            return ys[n - 1];
        }
        let j = xs.partition_point(|&x| x <= t) - 1;
        let slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
        slope * (t - xs[j]) + ys[j]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CurveError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("curve CSV line {0}: {1}")]
    Parse(usize, String),
}

/// The curve as CSV text (the exact file format).
pub fn curve_to_string(c: &Curve) -> String {
    let mut s = String::with_capacity(64 * c.meta.len() + 36 * c.t.len());
    s.push_str("# tapespeed-curve v1\n");
    for (k, v) in &c.meta {
        let _ = writeln!(s, "# {k}={}", v.to_text());
    }
    s.push_str("t_s,ratio,confidence,valid\n");
    for i in 0..c.t.len() {
        let _ = writeln!(
            s,
            "{},{},{},{}",
            fmt_fixed(c.t[i], 6),
            fmt_fixed(c.ratio[i], 9),
            fmt_fixed(c.confidence[i], 4),
            c.valid[i] as u8
        );
    }
    s
}

pub fn write_curve(path: impl AsRef<Path>, c: &Curve) -> Result<(), CurveError> {
    assert!(
        c.ratio.len() == c.t.len() && c.confidence.len() == c.t.len() && c.valid.len() == c.t.len()
    );
    std::fs::File::create(path)?.write_all(curve_to_string(c).as_bytes())?;
    Ok(())
}

/// Read a curve CSV: meta values stay text.
pub fn read_curve(path: impl AsRef<Path>) -> Result<Curve, CurveError> {
    parse_curve(&std::fs::read_to_string(path)?)
}

pub fn parse_curve(text: &str) -> Result<Curve, CurveError> {
    let mut c = Curve::default();
    let mut lines = text.lines().enumerate();
    for (_, line) in lines.by_ref() {
        if !line.starts_with('#') {
            break; // column-name line
        }
        if let Some((k, v)) = line[1..].trim().split_once('=') {
            c.meta
                .insert(k.trim().to_string(), MetaValue::Str(v.trim().to_string()));
        }
    }
    for (i, line) in lines {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue; // numpy.loadtxt skips blank / comment lines
        }
        let mut f = line.split(',').map(|v| v.trim().parse::<f64>());
        let mut next = || -> Result<f64, CurveError> {
            f.next()
                .ok_or_else(|| CurveError::Parse(i + 1, "expected 4 columns".into()))?
                .map_err(|e| CurveError::Parse(i + 1, e.to_string()))
        };
        let (t, r, q, v) = (next()?, next()?, next()?, next()?);
        if f.next().is_some() {
            return Err(CurveError::Parse(i + 1, "expected 4 columns".into()));
        }
        c.t.push(t);
        c.ratio.push(r);
        c.confidence.push(q);
        c.valid.push(v != 0.0); // numpy astype(bool): NaN -> True
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_str_and_format() {
        let mut c = Curve::default();
        c.meta.insert("a".into(), 9000.0.into());
        c.meta.insert("b".into(), 1e-5.into());
        c.meta.insert("c".into(), true.into());
        c.meta.insert("d".into(), 3i64.into());
        c.meta.insert("e".into(), "none".into());
        c.t = vec![0.0, 0.005];
        c.ratio = vec![0.9973291605, 1.0];
        c.confidence = vec![0.00005, 0.99995];
        c.valid = vec![false, true];
        assert_eq!(
            curve_to_string(&c),
            "# tapespeed-curve v1\n# a=9000.0\n# b=1e-05\n# c=True\n# d=3\n# e=none\nt_s,ratio,confidence,valid\n\
             0.000000,0.997329160,0.0001,0\n0.005000,1.000000000,1.0000,1\n"
        );
        let r = parse_curve(&curve_to_string(&c)).unwrap();
        assert_eq!(r.meta["b"], MetaValue::Str("1e-05".into()));
        assert_eq!(r.valid, vec![false, true]);
        assert!((c.ratio_at(0.0025) - 0.99866458).abs() < 1e-8);
    }
}
