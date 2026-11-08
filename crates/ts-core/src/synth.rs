//! synth: synthetic drifting test signals with a known speed curve.
//!
//! `x(t_in) = s(T(t_in))`, `T = ∫ r dt` (analytic). The reference tone is generated
//! analytically from `T`; program material is generated on the uniform τ grid and warped by
//! band-limited (Kaiser-windowed sinc) interpolation.
//!
//! Randomness goes through [`RandomSource`] streams (one per random generator: speed,
//! envelope, program, hiss, mech). [`PcgSource`] (rand_pcg) is the default and only
//! statistically equivalent to NumPy's generators; [`Recorded`] replays draws recorded from
//! NumPy generators, and then truth curve and audio follow a fixed operation order
//! (curve ≤ 1e-9, audio ≤ −120 dB against the stored reference vectors). Parallel parts
//! (T(t), warp) split the output into fixed chunks evaluated independently → identical for
//! any rayon thread count.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::path::Path;
use std::rc::Rc;

use indexmap::IndexMap;
use rand::RngExt;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use ts_dsp::spline::{PPoly, cubic_spline};
use ts_dsp::stats::{interp, np_sum};
use ts_dsp::window::{arange, i0};
use ts_dsp::{ValueError, conv, fir, ndimage};

use crate::curve::{Curve, CurveError, MetaValue, write_curve};

/// Truth grid [s].
pub const DT: f64 = 0.005;

/// Errors of the synth API.
#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    #[error("{0}")]
    Value(#[from] ValueError),
    #[error("{0}")]
    Param(String),
    #[error(transparent)]
    Io(#[from] ts_io::Error),
    #[error(transparent)]
    Curve(#[from] CurveError),
}

type Result<T> = std::result::Result<T, SynthError>;

fn perr<T>(msg: impl Into<String>) -> Result<T> {
    Err(SynthError::Param(msg.into()))
}

// ============================================================== parameters ============

/// Field kinds of [`Params`] (string / int / float).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Str,
    Float,
    Int,
}

/// Dataclass field order (meta order) and kinds.
const FIELDS: [(&str, Kind); 29] = [
    ("preset", Kind::Str),
    ("duration", Kind::Float),
    ("sr", Kind::Int),
    ("channels", Kind::Int),
    ("seed", Kind::Int),
    ("ref_hz", Kind::Float),
    ("offset", Kind::Float),
    ("drift", Kind::Float),
    ("drift_sin", Kind::Float),
    ("drift_period", Kind::Float),
    ("wow", Kind::Str),
    ("ref_db", Kind::Float),
    ("fade_db", Kind::Float),
    ("dropouts", Kind::Int),
    ("dropout_len", Kind::Float),
    ("program", Kind::Str),
    ("prog_db", Kind::Float),
    ("prog_lp", Kind::Float),
    ("snr_db", Kind::Float),
    ("mech", Kind::Str),
    ("reels", Kind::Str),
    ("mech_var", Kind::Float),
    ("mech_phase", Kind::Float),
    ("mech_tc", Kind::Float),
    ("mech_fdrift", Kind::Float),
    ("partials", Kind::Str),
    ("pair_hz", Kind::Float),
    ("pair_db", Kind::Float),
    ("lines", Kind::Str),
];

/// Synthesis parameters (see `Default` for the defaults).
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub preset: String,
    /// input file length [s]
    pub duration: f64,
    pub sr: usize,
    pub channels: usize,
    pub seed: u64,
    pub ref_hz: f64,
    /// speed curve [%]: constant offset
    pub offset: f64,
    /// linear change over the file [%]
    pub drift: f64,
    /// very-low-frequency sinusoid amplitude [%]
    pub drift_sin: f64,
    /// [s]; 0 → 1.5 · duration
    pub drift_period: f64,
    /// "hz:pct,..." (random phases)
    pub wow: String,
    /// tone rms rel. program rms [dB]
    pub ref_db: f64,
    /// slow fading depth (peak-to-peak) [dB]
    pub fade_db: f64,
    pub dropouts: usize,
    /// mean dropout length [s]
    pub dropout_len: f64,
    /// synth | none | audio file path
    pub program: String,
    pub prog_db: f64,
    /// program bandwidth [Hz]; 0 → 0.45 · sr
    pub prog_lp: f64,
    /// hiss rel. program rms (tone rms if none); inf → off
    pub snr_db: f64,
    /// "fspec:a1/a2/a3,..." (fspec = Hz | T | S | <Hz>-T; a_h in %); "" → off
    pub mech: String,
    /// v [m/s] : tape h [m] : R take-up0 : R supply0 [m]
    pub reels: String,
    pub mech_var: f64,
    pub mech_phase: f64,
    pub mech_tc: f64,
    pub mech_fdrift: f64,
    /// "f0[~f1]:dB rel. ref tone:t0:t1,..." on-tape interfering partials
    pub partials: String,
    /// AM beat pair channel spacing [Hz]; 0 → off
    pub pair_hz: f64,
    pub pair_db: f64,
    /// "f:dB[:post],..." extra steady lines
    pub lines: String,
    /// Float fields currently holding an integer literal (preset literals such as
    /// `ref_hz=19000`); an integer literal stays an integer in the meta header (no ".0").
    pub int_valued: BTreeSet<String>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            preset: "custom".into(),
            duration: 10.0,
            sr: 48000,
            channels: 2,
            seed: 0,
            ref_hz: 9000.0,
            offset: 0.0,
            drift: -0.2,
            drift_sin: 0.1,
            drift_period: 0.0,
            wow: "0.55:0.12,1.7:0.06,4.3:0.03".into(),
            ref_db: -20.0,
            fade_db: 0.0,
            dropouts: 0,
            dropout_len: 2.0,
            program: "synth".into(),
            prog_db: -20.0,
            prog_lp: 0.0,
            snr_db: 60.0,
            mech: String::new(),
            reels: "0.1905:31.5e-6:0.0288:0.0815".into(),
            mech_var: 0.0,
            mech_phase: 0.0,
            mech_tc: 60.0,
            mech_fdrift: 0.0,
            partials: String::new(),
            pair_hz: 0.0,
            pair_db: -10.0,
            lines: String::new(),
            int_valued: BTreeSet::new(),
        }
    }
}

/// Lenient float parse (accepts inf / nan spellings).
fn parse_float(v: &str) -> Result<f64> {
    let t = v.trim();
    match t.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Ok(f64::INFINITY),
        "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Ok(f64::NAN),
        _ => t
            .replace('_', "")
            .parse()
            .or_else(|_| perr(format!("could not convert string to float: {v:?}"))),
    }
}

fn parse_int(v: &str) -> Result<i64> {
    v.trim()
        .replace('_', "")
        .parse()
        .or_else(|_| perr(format!("invalid literal for int(): {v:?}")))
}

impl Params {
    fn kind(name: &str) -> Option<Kind> {
        FIELDS.iter().find(|(n, _)| *n == name).map(|f| f.1)
    }

    fn float_mut(&mut self, name: &str) -> Option<&mut f64> {
        Some(match name {
            "duration" => &mut self.duration,
            "ref_hz" => &mut self.ref_hz,
            "offset" => &mut self.offset,
            "drift" => &mut self.drift,
            "drift_sin" => &mut self.drift_sin,
            "drift_period" => &mut self.drift_period,
            "ref_db" => &mut self.ref_db,
            "fade_db" => &mut self.fade_db,
            "dropout_len" => &mut self.dropout_len,
            "prog_db" => &mut self.prog_db,
            "prog_lp" => &mut self.prog_lp,
            "snr_db" => &mut self.snr_db,
            "mech_var" => &mut self.mech_var,
            "mech_phase" => &mut self.mech_phase,
            "mech_tc" => &mut self.mech_tc,
            "mech_fdrift" => &mut self.mech_fdrift,
            "pair_hz" => &mut self.pair_hz,
            "pair_db" => &mut self.pair_db,
            _ => return None,
        })
    }

    fn str_mut(&mut self, name: &str) -> Option<&mut String> {
        Some(match name {
            "preset" => &mut self.preset,
            "wow" => &mut self.wow,
            "program" => &mut self.program,
            "mech" => &mut self.mech,
            "reels" => &mut self.reels,
            "partials" => &mut self.partials,
            "lines" => &mut self.lines,
            _ => return None,
        })
    }

    fn set_int_field(&mut self, name: &str, v: i64) -> Result<()> {
        let u = |v: i64| -> Result<u64> {
            u64::try_from(v).or_else(|_| perr(format!("{name} must be >= 0")))
        };
        match name {
            "sr" => self.sr = u(v)? as usize,
            "channels" => self.channels = u(v)? as usize,
            "seed" => self.seed = u(v)?,
            "dropouts" => self.dropouts = u(v)? as usize,
            _ => return perr(format!("unknown int field {name}")),
        }
        Ok(())
    }

    /// Set a field from a literal (`make_params` override semantics): for float fields an
    /// integer literal ("9000") stays an integer in the meta header (meta prints "9000").
    pub fn set(&mut self, name: &str, value: &str) -> Result<()> {
        match Self::kind(name) {
            None => perr(format!("unknown parameter {name:?}")),
            Some(Kind::Str) => {
                *self.str_mut(name).unwrap() = value.to_string();
                Ok(())
            }
            Some(Kind::Int) => self.set_int_field(name, parse_int(value)?),
            Some(Kind::Float) => {
                let is_int = parse_int(value).is_ok();
                *self.float_mut(name).unwrap() = parse_float(value)?;
                if is_int {
                    self.int_valued.insert(name.to_string());
                } else {
                    self.int_valued.remove(name);
                }
                Ok(())
            }
        }
    }

    /// Set a field as the CLI does (typed by the field's default: floats stay floats).
    pub fn set_cli(&mut self, name: &str, value: &str) -> Result<()> {
        if Self::kind(name) == Some(Kind::Float) {
            *self.float_mut(name).unwrap() = parse_float(value)?;
            self.int_valued.remove(name);
            Ok(())
        } else {
            self.set(name, value)
        }
    }

    /// Set a float field from an `f64` (printed as a float).
    pub fn set_f(&mut self, name: &str, v: f64) -> Result<()> {
        match self.float_mut(name) {
            Some(f) => {
                *f = v;
                self.int_valued.remove(name);
                Ok(())
            }
            None => perr(format!("{name} is not a float field")),
        }
    }

    /// Curve meta in its fixed order: `ref_hz, sr, source, <all other fields>`.
    pub fn meta(&self) -> IndexMap<String, MetaValue> {
        let mut m = IndexMap::new();
        m.insert("ref_hz".into(), self.meta_value("ref_hz"));
        m.insert("sr".into(), MetaValue::Int(self.sr as i64));
        m.insert("source".into(), MetaValue::Str("synth".into()));
        for (name, _) in FIELDS.iter() {
            if *name != "ref_hz" && *name != "sr" {
                m.insert(name.to_string(), self.meta_value(name));
            }
        }
        m
    }

    fn meta_value(&self, name: &str) -> MetaValue {
        match Self::kind(name) {
            Some(Kind::Str) => MetaValue::Str(self.clone().str_mut(name).unwrap().clone()),
            Some(Kind::Int) => MetaValue::Int(match name {
                "sr" => self.sr as i64,
                "channels" => self.channels as i64,
                "seed" => self.seed as i64,
                _ => self.dropouts as i64,
            }),
            _ => {
                let v = *self.clone().float_mut(name).unwrap();
                if self.int_valued.contains(name) {
                    MetaValue::Int(v as i64)
                } else {
                    MetaValue::Float(v)
                }
            }
        }
    }
}

/// Preset names (excluding "custom").
pub const PRESET_NAMES: [&str; 6] = ["fm19", "am9w", "am9", "reel", "clean", "mech"];

/// Preset `name` as literals (int literals stay ints, see [`Params::set`]).
fn preset(name: &str) -> Option<Vec<(&'static str, &'static str)>> {
    Some(match name {
        "fm19" => vec![
            ("ref_hz", "19000"),
            ("ref_db", "-20"),
            ("offset", "0.8"),
            ("snr_db", "50"),
        ],
        "am9w" => vec![
            ("ref_hz", "9000"),
            ("ref_db", "-30"),
            ("offset", "-1.5"),
            ("fade_db", "10"),
            ("dropouts", "2"),
            ("snr_db", "40"),
        ],
        "am9" => vec![
            ("ref_hz", "9000"),
            ("ref_db", "-24"),
            ("offset", "1.5"),
            ("fade_db", "6"),
            ("dropouts", "1"),
            ("snr_db", "45"),
        ],
        "reel" => vec![
            ("ref_hz", "9000"),
            ("ref_db", "-20"),
            ("offset", "-1.13"),
            ("sr", "44100"),
            ("fade_db", "4"),
            ("wow", "0.63:0.04,3.52:0.06,5.03:0.06,7.47:0.3,14.94:0.1"),
            ("snr_db", "45"),
        ],
        "clean" => vec![
            ("ref_hz", "9000"),
            ("ref_db", "0"),
            ("offset", "1.2"),
            ("program", "none"),
            ("snr_db", "inf"),
        ],
        "mech" => vec![
            ("ref_hz", "9000"),
            ("sr", "44100"),
            ("channels", "1"),
            ("duration", "720.0"),
            ("ref_db", "-20"),
            ("offset", "-1.0"),
            ("drift", "0.4"),
            ("drift_sin", "0.05"),
            ("wow", ""),
            ("fade_db", "6"),
            ("dropouts", "4"),
            ("dropout_len", "2.0"),
            ("snr_db", "45"),
            (
                "mech",
                "7.46:0.30/0.025/0.009,1.484:0.035/0.012,4.15-T:0.13/0.09/0.02,T:0.04/0.013,S:0.022,4.98:0.03",
            ),
            ("mech_var", "0.25"),
            ("mech_phase", "0.8"),
            ("mech_tc", "90.0"),
            ("mech_fdrift", "2e-4"),
            (
                "partials",
                "9030:3:150:160,9120~8930:6:400:406,8990:0:600:612,9040:-3:452:466",
            ),
        ],
        _ => return None,
    })
}

/// `make_params(preset, kw)`: defaults, then the preset, then `preset=name`, then `kw`
/// (literal values, see [`Params::set`]).
pub fn make_params(preset_name: &str, kw: &[(&str, &str)]) -> Result<Params> {
    let pre = if preset_name == "custom" {
        vec![]
    } else {
        match preset(preset_name) {
            Some(p) => p,
            None => return perr(format!("unknown preset {preset_name:?}")),
        }
    };
    let mut p = Params::default();
    for (k, v) in pre {
        p.set(k, v)?;
    }
    p.preset = preset_name.to_string();
    for (k, v) in kw {
        p.set(k, v)?;
    }
    Ok(p)
}

// ============================================================== randomness ============

/// One random stream (NumPy `Generator` API subset).
pub trait RandomSource {
    /// `rng.random()`: uniform double in [0, 1).
    fn random(&mut self) -> f64;
    /// `rng.uniform(lo, hi)` = `lo + (hi - lo) * random()` (numpy's formula).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random()
    }
    /// `rng.standard_normal(n)` (C order for multi-dimensional shapes).
    fn standard_normal(&mut self, n: usize) -> Vec<f64>;
}

/// Streams in creation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// `rs` (SeedSequence(seed).spawn(4)[0]): speed curve phases
    Speed = 0,
    /// `re`: reference envelope
    Envelope = 1,
    /// `rp`: program
    Program = 2,
    /// `rh`: hiss
    Hiss = 3,
    /// `default_rng([seed, 0x6D656368])`: mechanical components
    Mech = 4,
}

/// Creates the random streams of one synth run.
pub trait RandomFactory {
    fn stream(&self, s: Stream) -> Box<dyn RandomSource>;
}

/// Default source: PCG64 (rand_pcg) per (seed, stream); normals from rand_distr.
/// Statistically equivalent to numpy, not bit-compatible.
pub struct PcgSource(Pcg64);

impl RandomSource for PcgSource {
    fn random(&mut self) -> f64 {
        self.0.random::<f64>()
    }
    fn standard_normal(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.0.sample(StandardNormal)).collect()
    }
}

/// Factory of [`PcgSource`] streams for a seed.
#[derive(Debug, Clone, Copy)]
pub struct PcgFactory {
    pub seed: u64,
}

impl RandomFactory for PcgFactory {
    fn stream(&self, s: Stream) -> Box<dyn RandomSource> {
        let state = (self.seed as u128) << 64 | 0x7461_7065_7370_6565; // "tapespee"
        Box::new(PcgSource(Pcg64::new(state, 2 * s as u128 + 1)))
    }
}

/// Draws of one stream recorded from a NumPy generator (test vectors).
#[derive(Debug, Clone, Default)]
pub struct RecordedDraws {
    /// doubles behind `random()` / `uniform()` in call order
    pub u: Vec<f64>,
    /// all `standard_normal` outputs in call order
    pub n: Vec<f64>,
    iu: usize,
    in_: usize,
}

impl RecordedDraws {
    pub fn new(u: Vec<f64>, n: Vec<f64>) -> Self {
        Self {
            u,
            n,
            iu: 0,
            in_: 0,
        }
    }
    /// (uniforms used, normals used)
    pub fn used(&self) -> (usize, usize) {
        (self.iu, self.in_)
    }
}

struct RecordedSource(Rc<RefCell<RecordedDraws>>);

impl RandomSource for RecordedSource {
    fn random(&mut self) -> f64 {
        let mut d = self.0.borrow_mut();
        let i = d.iu;
        assert!(
            i < d.u.len(),
            "recorded uniforms exhausted (draw order differs)"
        );
        d.iu += 1;
        d.u[i]
    }
    fn standard_normal(&mut self, n: usize) -> Vec<f64> {
        let mut d = self.0.borrow_mut();
        let i = d.in_;
        assert!(
            i + n <= d.n.len(),
            "recorded normals exhausted (draw order differs)"
        );
        d.in_ += n;
        d.n[i..i + n].to_vec()
    }
}

/// Replays recorded NumPy draws per stream.
#[derive(Clone, Default)]
pub struct Recorded {
    streams: [Rc<RefCell<RecordedDraws>>; 5],
}

impl Recorded {
    /// `draws[s]` for `Stream as usize` = 0..5.
    pub fn new(draws: [RecordedDraws; 5]) -> Self {
        Self {
            streams: draws.map(|d| Rc::new(RefCell::new(d))),
        }
    }
    /// True if every recorded draw was consumed (same number and kinds of calls as when recorded).
    pub fn all_consumed(&self) -> bool {
        self.streams.iter().all(|s| {
            let d = s.borrow();
            d.iu == d.u.len() && d.in_ == d.n.len()
        })
    }
    pub fn used(&self, s: Stream) -> (usize, usize) {
        self.streams[s as usize].borrow().used()
    }
}

impl RandomFactory for Recorded {
    fn stream(&self, s: Stream) -> Box<dyn RandomSource> {
        Box::new(RecordedSource(self.streams[s as usize].clone()))
    }
}

// ============================================================== helpers ===============

/// numpy `np.mod(x, 1.0)` (floor-mod; 0 → +0).
#[inline]
fn np_mod1(x: f64) -> f64 {
    let m = x % 1.0;
    if m != 0.0 {
        if m < 0.0 { m + 1.0 } else { m }
    } else {
        0.0
    }
}

fn mean_sq(v: &[f64]) -> f64 {
    let sq: Vec<f64> = v.iter().map(|x| x * x).collect();
    np_sum(&sq) / v.len() as f64
}

fn split_nonempty(s: &str) -> impl Iterator<Item = &str> {
    s.split(',').filter(|c| !c.is_empty())
}

/// `smooth_noise(rng, t, tc)`: unit-rms Gaussian-filtered white noise (σ = tc/2) on a 1 s
/// grid, interpolated to `t`.
fn smooth_noise(rng: &mut dyn RandomSource, t: &[f64], tc: f64) -> Result<Vec<f64>> {
    let g = arange(t[0] - 4.0 * tc, t[t.len() - 1] + 4.0 * tc + 1.0, 1.0);
    let z = rng.standard_normal(g.len());
    let mut w = ndimage::gaussian_filter1d(&z, (tc / 2.0).max(1e-3), ndimage::Mode::Nearest)?;
    let d = mean_sq(&w).sqrt() + 1e-30;
    for v in &mut w {
        *v /= d;
    }
    Ok(interp(t, &g, &w)?)
}

// ============================================================== Mech ==================

/// Periodic mechanical components with harmonics and slowly varying parameters
/// (reel frequencies `f = v / (2π R)`, `R² = R0² ± v h t / π`); `r_mech` on a 1 kHz grid,
/// `T_mech` = exact antiderivative of its cubic spline.
pub struct Mech {
    pub tg: Vec<f64>,
    /// (fspec, frequency track on `tg`)
    pub comps: Vec<(String, Vec<f64>)>,
    r: PPoly,
    t_int: PPoly,
}

impl Mech {
    pub fn new(p: &Params, tmax: f64, rng: &mut dyn RandomSource) -> Result<Self> {
        let reels: Vec<f64> = p.reels.split(':').map(parse_float).collect::<Result<_>>()?;
        if reels.len() != 4 {
            return perr("reels must be v:h:R_takeup0:R_supply0");
        }
        let (v, h, rt0, rs0) = (reels[0], reels[1], reels[2], reels[3]);
        let tg = arange(0.0, tmax + 2.0, 1e-3);
        let k = v * h / PI;
        let ft: Vec<f64> = tg
            .iter()
            .map(|&t| v / (2.0 * PI * (rt0 * rt0 + k * t).sqrt()))
            .collect();
        let fsup: Vec<f64> = tg
            .iter()
            .map(|&t| v / (2.0 * PI * (rs0 * rs0 - k * t).max(1e-6).sqrt()))
            .collect();
        let mut y = vec![0.0; tg.len()];
        let mut comps = Vec::new();
        for spec in split_nonempty(&p.mech) {
            let Some((fs_, amps)) = spec.split_once(':') else {
                return perr(format!("bad mech component {spec:?}"));
            };
            let f: Vec<f64> = if fs_ == "T" {
                ft.clone()
            } else if fs_ == "S" {
                fsup.clone()
            } else if let Some(base) = fs_.strip_suffix("-T") {
                let b = parse_float(base)?;
                ft.iter().map(|&x| b - x).collect()
            } else {
                let b = parse_float(fs_)?;
                let sn = smooth_noise(rng, &tg, p.mech_tc)?;
                sn.iter().map(|&s| b * (1.0 + p.mech_fdrift * s)).collect()
            };
            // ph = 2π · concat([0], cumsum((f[1:] + f[:-1]) / 2) · 1e-3)
            let mut ph = Vec::with_capacity(f.len());
            ph.push(2.0 * PI * 0.0);
            let mut cs = 0.0;
            for i in 1..f.len() {
                cs += (f[i] + f[i - 1]) / 2.0;
                ph.push(2.0 * PI * (cs * 1e-3));
            }
            let ph0 = rng.uniform(0.0, 2.0 * PI);
            for (hh, a) in amps.split('/').enumerate() {
                let a = parse_float(a)? / 100.0;
                let hh = (hh + 1) as f64;
                let sv = smooth_noise(rng, &tg, p.mech_tc)?;
                let u0 = rng.uniform(0.0, 2.0 * PI);
                let sp = smooth_noise(rng, &tg, p.mech_tc)?;
                for i in 0..tg.len() {
                    let am = a * (1.0 + p.mech_var * sv[i]).max(0.0);
                    let ps = u0 + p.mech_phase * sp[i];
                    y[i] += am * (hh * (ph[i] + ph0) + ps).sin();
                }
            }
            comps.push((fs_.to_string(), f));
        }
        let r = cubic_spline(&tg, &y)?;
        let t_int = r.antiderivative();
        Ok(Self {
            tg,
            comps,
            r,
            t_int,
        })
    }

    /// `r_mech(t)` (cubic spline).
    pub fn r(&self, t: f64) -> f64 {
        self.r.eval(t)
    }

    /// `T_mech(t)` (antiderivative).
    pub fn t_int(&self, t: f64) -> f64 {
        self.t_int.eval(t)
    }
}

// ============================================================== Speed =================

/// `r(t) = 1 + offset + drift·t/D + a_s sin(2π t/P + ps) + Σ a_i sin(2π f_i t + p_i) [+ mech]`
/// and its analytic integral `T(t)`.
pub struct Speed {
    o: f64,
    d: f64,
    dur: f64,
    a_s: f64,
    period: f64,
    p_s: f64,
    /// (f, a, phase)
    pub wow: Vec<(f64, f64, f64)>,
    pub mech: Option<Mech>,
}

impl Speed {
    /// `Speed(p, rng)`; the mechanical part draws from `mech_rng` (its own generator).
    pub fn new(
        p: &Params,
        rng: &mut dyn RandomSource,
        mech_rng: Option<&mut dyn RandomSource>,
    ) -> Result<Self> {
        let p_s = rng.uniform(0.0, 2.0 * PI);
        let mut wow = Vec::new();
        for w in split_nonempty(&p.wow) {
            let Some((f, a)) = w.split_once(':') else {
                return perr(format!("bad wow component {w:?}"));
            };
            wow.push((
                parse_float(f)?,
                parse_float(a)? / 100.0,
                rng.uniform(0.0, 2.0 * PI),
            ));
        }
        let mech = if p.mech.is_empty() {
            None
        } else {
            match mech_rng {
                Some(r) => Some(Mech::new(p, p.duration, r)?),
                None => return perr("mech components need a mech random stream"),
            }
        };
        Ok(Self {
            o: p.offset / 100.0,
            d: p.drift / 100.0,
            dur: p.duration,
            a_s: p.drift_sin / 100.0,
            period: if p.drift_period != 0.0 {
                p.drift_period
            } else {
                1.5 * p.duration
            },
            p_s,
            wow,
            mech,
        })
    }

    /// `r(t)` at one point.
    pub fn r(&self, t: f64) -> f64 {
        let mut y = 1.0
            + self.o
            + self.d * t / self.dur
            + self.a_s * (2.0 * PI * t / self.period + self.p_s).sin();
        for &(f, a, ph) in &self.wow {
            y += a * (2.0 * PI * f * t + ph).sin();
        }
        if let Some(m) = &self.mech {
            y += m.r(t);
        }
        y
    }

    /// `T(t) = ∫_0^t r` at one point (analytic).
    pub fn t_int(&self, t: f64) -> f64 {
        let w = 2.0 * PI / self.period;
        let mut y = (1.0 + self.o) * t
            + self.d * (t * t) / (2.0 * self.dur)
            + self.a_s / w * (self.p_s.cos() - (w * t + self.p_s).cos());
        for &(f, a, ph) in &self.wow {
            let w = 2.0 * PI * f;
            y += a / w * (ph.cos() - (w * t + ph).cos());
        }
        if let Some(m) = &self.mech {
            y += m.t_int(t);
        }
        y
    }

    /// `r` on many points (parallel, fixed chunks).
    pub fn r_vec(&self, t: &[f64]) -> Vec<f64> {
        par_map(t, |v| self.r(v))
    }

    /// `T` on many points (parallel, fixed chunks).
    pub fn t_vec(&self, t: &[f64]) -> Vec<f64> {
        par_map(t, |v| self.t_int(v))
    }
}

fn par_map<F: Fn(f64) -> f64 + Sync>(t: &[f64], f: F) -> Vec<f64> {
    let mut out = vec![0.0; t.len()];
    out.par_chunks_mut(1 << 14)
        .zip(t.par_chunks(1 << 14))
        .for_each(|(o, ti)| {
            for (a, &b) in o.iter_mut().zip(ti) {
                *a = f(b);
            }
        });
    out
}

// ============================================================== warp ==================

/// Kernel table of [`warp`]: `(nph + 1) × 2·half` row-major,
/// `tab[p][jj] = cutoff·sinc(cutoff·x)·i0(β·sqrt(max(1 − (x/half)², 0)))/i0(β)`,
/// `x = j − p/nph`, `j = jj − half + 1`.
pub fn warp_table(half: usize, cutoff: f64, beta: f64, nph: usize) -> Vec<f64> {
    let nt = 2 * half;
    let den = i0(beta);
    let mut tab = Vec::with_capacity((nph + 1) * nt);
    for p in 0..=nph {
        let fr = p as f64 / nph as f64;
        for jj in 0..nt {
            let j = jj as f64 - half as f64 + 1.0;
            let x = j - fr;
            let q = x / half as f64;
            let arg = beta * (1.0 - q * q).max(0.0).sqrt();
            tab.push(cutoff * fir::sinc(cutoff * x) * i0(arg) / den);
        }
    }
    tab
}

/// Frames per parallel work item of the streamed synthesis.
const BLK: usize = 1 << 15;
/// Frames per sequential super-block (hiss draws are taken in this granularity).
const SUPER: usize = 1 << 20;

/// Kernel lookup of [`warp`] for one output frame.
struct Warper {
    tab: Vec<f64>,
    nt: usize,
    nph: usize,
}

impl Warper {
    fn new(half: usize, cutoff: f64, beta: f64, nph: usize) -> Self {
        Warper {
            tab: warp_table(half, cutoff, beta, nph),
            nt: 2 * half,
            nph,
        }
    }

    /// One frame at position `uu` from zero-padded channels `sp` (data at offset `half`,
    /// `half + 1` zeros after): `out[c] = Σ h·sp[c][clip(i + j + half)]`. `h`, `g`: scratch.
    #[inline]
    fn frame(&self, sp: &[Vec<f64>], uu: f64, h: &mut [f64], g: &mut [f64], out: &mut [f64]) {
        let (nt, nph) = (self.nt, self.nph);
        let len = sp.first().map_or(0, |v| v.len()) as i64;
        let i = uu.floor();
        let qf = (uu - i) * nph as f64;
        let k = (qf as i64).min(nph as i64 - 1) as usize;
        let w = qf - k as f64;
        let (t0, t1) = (
            &self.tab[k * nt..(k + 1) * nt],
            &self.tab[(k + 1) * nt..(k + 2) * nt],
        );
        for ((hv, &a), &bb) in h.iter_mut().zip(t0).zip(t1) {
            *hv = (1.0 - w) * a + w * bb;
        }
        // idx = clip(i + j + half, 0, len-1), j = -half+1 ..= half
        let lo = i as i64 + 1;
        let hi = lo + nt as i64 - 1;
        for (c, o) in out.iter_mut().enumerate() {
            let src = &sp[c];
            *o = if lo >= 0 && hi < len {
                dot(h, &src[lo as usize..=hi as usize])
            } else {
                for (jj, gv) in g.iter_mut().enumerate() {
                    *gv = src[(lo + jj as i64).clamp(0, len - 1) as usize];
                }
                dot(h, g)
            };
        }
    }
}

/// `warp(s, u)`: band-limited interpolation of the channels `s[c]` at fractional sample
/// positions `u` (Kaiser-windowed sinc, `2·half` taps, lowpass at `cutoff`·Nyquist, kernel
/// table with `nph` phases linearly interpolated). Returns one output vector per channel.
pub fn warp(
    s: &[Vec<f64>],
    u: &[f64],
    half: usize,
    cutoff: f64,
    beta: f64,
    nph: usize,
) -> Vec<Vec<f64>> {
    let wp = Warper::new(half, cutoff, beta, nph);
    let nt = 2 * half;
    let sp: Vec<Vec<f64>> = s
        .iter()
        .map(|c| {
            let mut v = vec![0.0; half];
            v.extend_from_slice(c);
            v.resize(v.len() + half + 1, 0.0);
            v
        })
        .collect();
    let ch = s.len().max(1);
    let mut out = vec![0.0; u.len() * ch]; // frame-major
    out.par_chunks_mut(4096 * ch)
        .enumerate()
        .for_each(|(b, dst)| {
            let (mut h, mut g) = (vec![0.0; nt], vec![0.0; nt]);
            for (q, frame) in dst.chunks_mut(ch).enumerate() {
                wp.frame(&sp, u[b * 4096 + q], &mut h, &mut g, frame);
            }
        });
    (0..s.len())
        .map(|c| out.iter().skip(c).step_by(ch).copied().collect())
        .collect()
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let ca = a.chunks_exact(4);
    let cb = b.chunks_exact(4);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0f64; 4];
    for (x, y) in ca.zip(cb) {
        for k in 0..4 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Σ v² over fixed blocks of `f(block range)` partial sums, merged in block order
/// (deterministic for any thread count; differs from numpy's pairwise sum only in rounding).
fn sum_sq_blocks<F: Fn(usize, usize) -> f64 + Sync>(n: usize, f: F) -> f64 {
    let parts: Vec<f64> = (0..n.div_ceil(BLK))
        .into_par_iter()
        .map(|b| f(b * BLK, ((b + 1) * BLK).min(n)))
        .collect();
    parts.iter().sum()
}

/// Mutable per-block views of every column's data range `[off, off + n)`:
/// `out[b][c]` = frames `b·BLK ..` of column `c`.
fn col_blocks(cols: &mut [Vec<f64>], off: usize, n: usize) -> Vec<Vec<&mut [f64]>> {
    let mut per: Vec<Vec<&mut [f64]>> = (0..n.div_ceil(BLK)).map(|_| Vec::new()).collect();
    for col in cols.iter_mut() {
        for (b, s) in col[off..off + n].chunks_mut(BLK).enumerate() {
            per[b].push(s);
        }
    }
    per
}

// ============================================================== program ===============

/// One note of [`synth_program`] (drawn first, rendered block by block afterwards).
struct Note {
    i0: usize,
    m: usize,
    d: f64,
    f0: f64,
    vr: f64,
    vd: f64,
    amp: Vec<f64>,
    tau: f64,
    g: Vec<f64>,
}

impl Note {
    /// Sample `q` (0-based within the note), before the channel gains.
    #[inline]
    fn value(&self, q: usize, srf: f64) -> f64 {
        let tt = q as f64 / srf;
        let (f0, vd, vr) = (self.f0, self.vd, self.vr);
        let ph = 2.0 * PI * f0 * (tt + vd * (1.0 - (2.0 * PI * vr * tt).cos()) / (2.0 * PI * vr));
        let mut note = 0.0;
        for (k, &a) in self.amp.iter().enumerate() {
            note += a * ((k + 1) as f64 * ph).sin();
        }
        let mut env = (1.0 - (-tt / 0.005).exp()) * (-tt / self.tau).exp();
        env *= ((self.d - tt) / 0.01).clamp(0.0, 1.0);
        note * env
    }
}

/// Notes summed into frame-major `y` (frames `a..a + y.len()/ch`), in draw order.
fn render_notes(notes: &[Note], a: usize, ch: usize, srf: f64, y: &mut [f64]) {
    let b = a + y.len() / ch;
    for nt in notes {
        let (lo, hi) = (nt.i0.max(a), (nt.i0 + nt.m).min(b));
        for i in lo..hi {
            let v = nt.value(i - nt.i0, srf);
            for (c, &gc) in nt.g.iter().enumerate() {
                y[(i - a) * ch + c] += v * gc;
            }
        }
    }
}

/// `synth_program(n, sr, ch, lp, rng)`: harmonic notes (random pitch, vibrato, decay, pan) +
/// pink noise. Returns one column per channel, `pad.0` zeros + `n` frames + `pad.1` zeros.
/// Memory ~ the columns + one FFT spectrum: the notes are rendered per block twice (rms,
/// then output) instead of being stored.
pub fn synth_program(
    n: usize,
    sr: usize,
    ch: usize,
    lp: f64,
    rng: &mut dyn RandomSource,
    pad: (usize, usize),
) -> Vec<Vec<f64>> {
    let voices = 3;
    let srf = sr as f64;
    let dur = n as f64 / srf;
    let mut notes = Vec::new();
    for _ in 0..voices {
        let mut t0 = rng.uniform(0.0, 0.5);
        while t0 < dur {
            let d = rng.uniform(0.15, 1.2);
            let i0 = (t0 * srf) as usize;
            let i1 = n.min(((t0 + d) * srf) as usize);
            t0 += d;
            if rng.random() < 0.2 || i1 <= i0 {
                continue; // rest
            }
            let f0 = 60.0 * 2f64.powf(rng.uniform(0.0, 4.4));
            let vr = rng.uniform(4.0, 6.5);
            let vd = rng.uniform(0.0, 0.006);
            let kk = 1.max(24.min((lp / (f0 * (1.0 + vd))) as i64)) as usize;
            let ex = -rng.uniform(0.6, 1.4);
            let mut amp: Vec<f64> = (1..=kk).map(|k| (k as f64).powf(ex)).collect();
            let s = np_sum(&amp.iter().map(|a| a * a).collect::<Vec<_>>()).sqrt();
            for a in &mut amp {
                *a /= s;
            }
            let tau = rng.uniform(0.2, 1.5);
            let pan = if ch == 2 { rng.uniform(0.2, 0.8) } else { 0.5 };
            let gu = rng.uniform(0.4, 1.0);
            let g: Vec<f64> = if ch == 2 {
                vec![gu * (1.0 - pan).sqrt(), gu * pan.sqrt()]
            } else {
                vec![gu * 1.0; ch]
            };
            notes.push(Note {
                i0,
                m: i1 - i0,
                d,
                f0,
                vr,
                vd,
                amp,
                tau,
                g,
            });
        }
    }
    let render = |a: usize, b: usize| {
        let mut y = vec![0.0; (b - a) * ch];
        render_notes(&notes, a, ch, srf, &mut y);
        y
    };
    let sy = sum_sq_blocks(n, |a, b| render(a, b).iter().map(|v| v * v).sum());
    let dy = (sy / (n * ch) as f64).sqrt() + 1e-12;
    // pink noise, partly correlated between channels, slowly modulated, bandlimited to lp:
    // w1 = normal(n·ch) (frame-major), w2 = normal(n), drawn block by block in that order
    let off = pad.0;
    let mut cols: Vec<Vec<f64>> = (0..ch).map(|_| vec![0.0; off + n + pad.1]).collect();
    for a in (0..n).step_by(SUPER) {
        let b = (a + SUPER).min(n);
        let w1 = rng.standard_normal((b - a) * ch);
        for (c, col) in cols.iter_mut().enumerate() {
            for (i, v) in col[off + a..off + b].iter_mut().enumerate() {
                *v = w1[i * ch + c] * 0.6;
            }
        }
    }
    for a in (0..n).step_by(SUPER) {
        let b = (a + SUPER).min(n);
        let w2 = rng.standard_normal(b - a);
        for col in cols.iter_mut() {
            for (v, w) in col[off + a..off + b].iter_mut().zip(&w2) {
                *v += w * 0.8;
            }
        }
    }
    for col in cols.iter_mut() {
        let x = &mut col[off..off + n];
        if n <= FULL_FFT_MAX {
            pink_full(x, lp, srf);
        } else {
            let y = pink_blocks(x, lp, srf);
            x.copy_from_slice(&y);
        }
    }
    let ua = rng.uniform(-6.0, 0.0);
    let ub = rng.uniform(5.0, 15.0);
    let gm =
        |i: usize| 10f64.powf(ua / 20.0 * (1.0 - (2.0 * PI * i as f64 / srf / ub).cos()) / 2.0);
    let sn = {
        let mut blocks = col_blocks(&mut cols, off, n);
        let parts: Vec<f64> = blocks
            .par_iter_mut()
            .enumerate()
            .map(|(bi, bc)| {
                let mut s = 0.0;
                for q in 0..bc[0].len() {
                    let g = gm(bi * BLK + q);
                    for col in bc.iter_mut() {
                        col[q] *= g;
                        s += col[q] * col[q];
                    }
                }
                s
            })
            .collect();
        parts.iter().sum::<f64>()
    };
    let dn = (sn / (n * ch) as f64).sqrt() + 1e-12;
    let mut blocks = col_blocks(&mut cols, off, n);
    blocks.par_iter_mut().enumerate().for_each(|(bi, bc)| {
        let a = bi * BLK;
        let y = render(a, a + bc[0].len());
        for (c, col) in bc.iter_mut().enumerate() {
            for (q, v) in col.iter_mut().enumerate() {
                *v = y[q * ch + c] / dy + 0.3 * *v / dn;
            }
        }
    });
    cols
}

/// Longest program (frames) whose pink noise is shaped by one full-length FFT (the defining
/// construction); longer programs use [`pink_blocks`] (an exact full-length DFT of an arbitrary
/// size needs Bluestein buffers of ~4× the column, ~20 GB at 2 h).
const FULL_FFT_MAX: usize = 1 << 24;

/// Pink mask of `synth_program` at frequency `fv`.
#[inline]
fn pink_mask(fv: f64, lp: f64) -> f64 {
    (if fv < lp {
        1.0 / fv.max(40.0).sqrt()
    } else {
        0.0
    }) * 40f64.sqrt()
}

/// Reference construction: `irfft(rfft(x) * mask)` over the whole column, in place.
fn pink_full(x: &mut [f64], lp: f64, srf: f64) {
    let n = x.len();
    let val = 1.0 / (n as f64 * (1.0 / srf)); // rfftfreq(n, 1/sr) step
    let mut planner = RealFftPlanner::<f64>::new();
    let mut spec = {
        let fwd = planner.plan_fft_forward(n);
        let mut spec = fwd.make_output_vec();
        fwd.process(x, &mut spec).expect("fft length");
        spec
    };
    for (k, v) in spec.iter_mut().enumerate() {
        *v *= pink_mask(k as f64 * val, lp);
    }
    spec[0].im = 0.0;
    if n.is_multiple_of(2) {
        let l = spec.len() - 1;
        spec[l].im = 0.0;
    }
    planner
        .plan_fft_inverse(n)
        .process(&mut spec, x)
        .expect("fft length");
    let s = 1.0 / n as f64;
    x.par_iter_mut().for_each(|v| *v *= s);
}

/// Long programs: the same mask as a zero-phase FIR (`K` taps, Hann-windowed impulse response
/// of the mask sampled on a `K`-point grid), applied by overlap-save in fixed blocks
/// (parallel, deterministic). Same spectral shape except a ~sr/K smoothing of the edges;
/// the absolute scale does not matter (normalised to unit rms afterwards).
fn pink_blocks(x: &[f64], lp: f64, srf: f64) -> Vec<f64> {
    const K: usize = 1 << 14;
    const L: usize = 1 << 16;
    const B: usize = L - K;
    let n = x.len();
    let mut planner = RealFftPlanner::<f64>::new();
    let (fk, ik) = (planner.plan_fft_forward(K), planner.plan_fft_inverse(K));
    let mut mk = fk.make_output_vec();
    for (k, v) in mk.iter_mut().enumerate() {
        *v = num_complex::Complex64::new(pink_mask(k as f64 * srf / K as f64, lp), 0.0);
    }
    let mut h0 = ik.make_output_vec();
    ik.process(&mut mk, &mut h0).expect("fft length");
    let (fl, il) = (planner.plan_fft_forward(L), planner.plan_fft_inverse(L));
    let mut hc = fl.make_input_vec();
    for (j, v) in hc[..K].iter_mut().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * j as f64 / K as f64).cos();
        *v = h0[(j + K - K / 2) % K] / K as f64 * w;
    }
    let mut hf = fl.make_output_vec();
    fl.process(&mut hc, &mut hf).expect("fft length");
    let mut y = vec![0.0; n];
    y.par_chunks_mut(B).enumerate().for_each(|(bi, dst)| {
        let a = (bi * B) as i64;
        let mut buf = fl.make_input_vec();
        for (t, v) in buf.iter_mut().enumerate() {
            let i = a - (K / 2) as i64 + t as i64;
            *v = if i >= 0 && (i as usize) < n {
                x[i as usize]
            } else {
                0.0
            };
        }
        let mut sp = fl.make_output_vec();
        fl.process(&mut buf, &mut sp).expect("fft length");
        for (v, h) in sp.iter_mut().zip(&hf) {
            *v *= h;
        }
        sp[0].im = 0.0;
        let l = sp.len() - 1;
        sp[l].im = 0.0;
        il.process(&mut sp, &mut buf).expect("fft length");
        for (q, o) in dst.iter_mut().enumerate() {
            *o = buf[q + K] / L as f64;
        }
    });
    y
}

/// `load_program(path, n, sr, ch)`: audio file, resampled to `sr` (resample_poly), channel
/// count matched (mono mix), DC removed, looped to `n` frames. Columns padded as
/// [`synth_program`].
pub fn load_program(
    path: &str,
    n: usize,
    sr: usize,
    ch: usize,
    pad: (usize, usize),
) -> Result<Vec<Vec<f64>>> {
    let a = ts_io::load(path, None, None)?;
    let ch0 = a.channels.max(1);
    let mut cols: Vec<Vec<f64>> = (0..ch0)
        .map(|c| {
            a.data
                .iter()
                .skip(c)
                .step_by(ch0)
                .map(|&v| v as f64)
                .collect()
        })
        .collect();
    let sr0 = a.sr as usize;
    drop(a);
    if sr0 != sr {
        let g = gcd(sr, sr0);
        for c in &mut cols {
            *c = conv::resample_poly(c, sr / g, sr0 / g, None)?;
        }
    }
    if ch0 != ch {
        let len = cols[0].len();
        let m: Vec<f64> = (0..len)
            .map(|i| np_sum(&cols.iter().map(|c| c[i]).collect::<Vec<_>>()) / ch0 as f64)
            .collect();
        cols = vec![m; ch];
    }
    for c in &mut cols {
        let mu = np_sum(c) / c.len() as f64;
        for v in c.iter_mut() {
            *v -= mu;
        }
    }
    let len = cols[0].len();
    if len == 0 {
        return perr("program file is empty");
    }
    Ok(cols
        .iter()
        .map(|c| {
            let mut v = vec![0.0; pad.0 + n + pad.1];
            for (i, o) in v[pad.0..pad.0 + n].iter_mut().enumerate() {
                *o = c[i % len];
            }
            v
        })
        .collect())
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

// ============================================================== envelope / partials ===

/// `ref_envelope(t, p, rng)`: slow fading (dB) times hard dropouts with 50 ms raised-cosine
/// edges. All draws happen in [`Envelope::new`]; [`Envelope::at`] evaluates one sample.
struct Envelope {
    fade_db: f64,
    /// (fr, ph) of the three fading sines
    fades: Vec<(f64, f64)>,
    mn: f64,
    ptp: f64,
    /// (c, l) per dropout
    drops: Vec<(f64, f64)>,
}

impl Envelope {
    fn new(n: usize, srf: f64, p: &Params, rng: &mut dyn RandomSource) -> Self {
        let mut fades = vec![];
        if p.fade_db > 0.0 {
            for _ in 0..3 {
                let fr = rng.uniform(0.02, 0.25);
                let ph = rng.uniform(0.0, 2.0 * PI);
                fades.push((fr, ph));
            }
        }
        let dur = (n - 1) as f64 / srf;
        let mut drops = vec![];
        for k in 0..p.dropouts {
            let l = p.dropout_len * rng.uniform(0.6, 1.4);
            let c = dur * (k as f64 + 0.5 + rng.uniform(-0.25, 0.25)) / p.dropouts as f64;
            drops.push((c, l));
        }
        let mut e = Envelope {
            fade_db: p.fade_db,
            fades,
            mn: 0.0,
            ptp: 0.0,
            drops,
        };
        if !e.fades.is_empty() {
            let (mn, mx) = (0..n.div_ceil(BLK))
                .into_par_iter()
                .map(|b| {
                    (b * BLK..((b + 1) * BLK).min(n))
                        .map(|i| e.fade_sum(i as f64 / srf))
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                            (lo.min(v), hi.max(v))
                        })
                })
                .reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |a, b| (a.0.min(b.0), a.1.max(b.1)),
                );
            e.mn = mn;
            e.ptp = mx - mn;
        }
        e
    }

    #[inline]
    fn fade_sum(&self, tv: f64) -> f64 {
        let mut ev = 0.0;
        for &(fr, ph) in &self.fades {
            ev += (2.0 * PI * fr * tv + ph).sin();
        }
        ev
    }

    #[inline]
    fn at(&self, tv: f64) -> f64 {
        let mut g = 1.0;
        if !self.fades.is_empty() {
            let en = (self.fade_sum(tv) - self.mn) / (self.ptp + 1e-12);
            g *= 10f64.powf(-self.fade_db * en / 20.0);
        }
        for &(c, l) in &self.drops {
            let m = ((c - l / 2.0 - tv).max(tv - c - l / 2.0) / 0.05).clamp(0.0, 1.0);
            g *= 0.5 - 0.5 * (PI * m).cos();
        }
        g
    }
}

/// One `partials` entry: steady `f0` or sweep `f0~f1` in original time over frames
/// `i0..i1`, 20 ms raised-cosine edges.
struct Partial {
    f0: f64,
    f1: f64,
    t0: f64,
    t1: f64,
    i0: usize,
    i1: usize,
    tt0: f64,
    ulast: f64,
    amp: f64,
}

/// `partials(p, t, T, a)` setup: frame ranges `t[i0..i1]` (`t[i] = i/sr`) and `T` at the ends.
fn partial_specs(p: &Params, n: usize, srf: f64, sp: &Speed, a: f64) -> Result<Vec<Partial>> {
    // t.partition_point(|v| v < x) on t[i] = i / sr
    let pp = |x: f64| {
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (mid as f64 / srf) < x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let mut out = vec![];
    for spec in split_nonempty(&p.partials) {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 4 {
            return perr(format!("bad partial {spec:?}"));
        }
        let (f0, f1) = match parts[0].split_once('~') {
            Some((a0, a1)) => (parse_float(a0)?, parse_float(a1)?),
            None => (parse_float(parts[0])?, parse_float(parts[0])?),
        };
        let db = parse_float(parts[1])?;
        let (t0, t1) = (parse_float(parts[2])?, parse_float(parts[3])?);
        let (i0, i1) = (pp(t0), pp(t1));
        if i1 <= i0 {
            continue;
        }
        let tt0 = sp.t_int(i0 as f64 / srf);
        let ulast = sp.t_int((i1 - 1) as f64 / srf) - tt0;
        out.push(Partial {
            f0,
            f1,
            t0,
            t1,
            i0,
            i1,
            tt0,
            ulast,
            amp: a * 10f64.powf(db / 20.0),
        });
    }
    Ok(out)
}

/// Sum of the partials at frame `i` (time `ti`, original time `tti`).
#[inline]
fn partials_at(ps: &[Partial], i: usize, ti: f64, tti: f64) -> f64 {
    let mut y = 0.0;
    for q in ps {
        if i < q.i0 || i >= q.i1 {
            continue;
        }
        let u = tti - q.tt0;
        let ph = q.f0 * u + (q.f1 - q.f0) * (u * u) / (2.0 * q.ulast.max(1e-9));
        let mut env = ((ti - q.t0).min(q.t1 - ti) / 0.02).clamp(0.0, 1.0);
        env = 0.5 - 0.5 * (PI * env).cos();
        y += q.amp * env * (2.0 * PI * np_mod1(ph)).sin();
    }
    y
}

// ============================================================== synth =================

/// Result of [`run`]: audio (interleaved f32) and the truth curve on the [`DT`] grid.
#[derive(Debug, Clone)]
pub struct Synth {
    pub audio: ts_io::Audio,
    pub truth: Curve,
}

/// `synth(p)` with the default (PCG) random streams.
pub fn run(p: &Params) -> Result<Synth> {
    run_with(p, &PcgFactory { seed: p.seed })
}

/// `synth(p)` with the given random streams (e.g. [`Recorded`] numpy draws).
///
/// Streamed: the output is computed frame by frame in fixed blocks (time, `T`, envelope,
/// partials, lines, warped program, hiss) straight into the f32 result; the only
/// full-length intermediates are the program columns (f64) and, while they are made, one
/// FFT spectrum. Peak ≈ 2.5 × the f32 output for stereo program material.
pub fn run_with(p: &Params, rf: &dyn RandomFactory) -> Result<Synth> {
    let mut rs = rf.stream(Stream::Speed);
    let mut re = rf.stream(Stream::Envelope);
    let mut rp = rf.stream(Stream::Program);
    let mut rh = rf.stream(Stream::Hiss);
    let (sr, ch) = (p.sr, p.channels);
    if sr == 0 || ch == 0 {
        return perr("sr and channels must be positive");
    }
    let srf = sr as f64;
    let n = (p.duration * srf).round_ties_even() as usize;
    if n == 0 {
        return perr("duration too short (no samples)");
    }
    let mut rm = if p.mech.is_empty() {
        None
    } else {
        Some(rf.stream(Stream::Mech))
    };
    let sp = Speed::new(
        p,
        rs.as_mut(),
        rm.as_mut().map(|b| &mut **b as &mut dyn RandomSource),
    )?;
    let a = 2f64.sqrt() * 10f64.powf((p.prog_db + p.ref_db) / 20.0);
    let env = Envelope::new(n, srf, p, re.as_mut());
    let parts = partial_specs(p, n, srf, &sp, a)?;
    let has_partials = !p.partials.is_empty();
    let mut extra: Vec<(f64, f64, bool)> = Vec::new();
    if p.pair_hz != 0.0 {
        extra.push((2.0 * p.pair_hz - p.ref_hz, p.pair_db, false));
    }
    for w in split_nonempty(&p.lines) {
        let v: Vec<&str> = w.split(':').collect();
        if v.len() < 2 {
            return perr(format!("bad line {w:?}"));
        }
        extra.push((
            parse_float(v[0])?,
            parse_float(v[1])?,
            v.len() > 2 && v[2] == "post",
        ));
    }
    let extra: Vec<(f64, f64, bool)> = extra
        .into_iter()
        .map(|(f, db, post)| (f, a * 10f64.powf(db / 20.0), post))
        .collect();
    let mut ref_rms = a / 2f64.sqrt();
    let half = 64;
    let prog: Option<Vec<Vec<f64>>> = if p.program != "none" {
        let m = (sp.t_int((n - 1) as f64 / srf) * srf).ceil() as usize + half + 2;
        let lp = if p.prog_lp != 0.0 {
            p.prog_lp
        } else {
            0.45 * srf
        };
        let pad = (half, half + 1);
        let mut cols = if p.program == "synth" {
            synth_program(m, sr, ch, lp, rp.as_mut(), pad)
        } else {
            load_program(&p.program, m, sr, ch, pad)?
        };
        let ss = sum_sq_blocks(m, |a0, b0| {
            cols.iter()
                .map(|c| c[half + a0..half + b0].iter().map(|v| v * v).sum::<f64>())
                .sum()
        });
        let fac = 10f64.powf(p.prog_db / 20.0) / (ss / (m * ch) as f64).sqrt();
        for c in &mut cols {
            c[half..half + m].par_iter_mut().for_each(|v| *v *= fac);
        }
        ref_rms = 10f64.powf(p.prog_db / 20.0);
        Some(cols)
    } else {
        None
    };
    let wp = Warper::new(half, 0.9, 9.0, 2048);
    let hk = p.snr_db.is_finite().then(|| 10f64.powf(-p.snr_db / 20.0));
    let mut data = vec![0f32; n * ch];
    let mut pk = 0.0f64;
    for (sb, out) in data.chunks_mut(SUPER * ch).enumerate() {
        let a0 = sb * SUPER;
        // hiss: rh.standard_normal(n·ch), drawn in super-blocks (same sequence)
        let z = hk.map(|_| rh.standard_normal(out.len()));
        let bpk = out
            .par_chunks_mut(BLK * ch)
            .enumerate()
            .map(|(bi, dst)| {
                let (mut h, mut g) = (vec![0.0; 2 * half], vec![0.0; 2 * half]);
                let mut wv = vec![0.0; ch];
                let mut x = vec![0.0; ch];
                let mut lpk = 0.0f64;
                for (q, fr) in dst.chunks_mut(ch).enumerate() {
                    let j = bi * BLK + q; // frame within the super-block
                    let i = a0 + j;
                    let ti = i as f64 / srf;
                    let tti = sp.t_int(ti);
                    let tone = a * env.at(ti) * (2.0 * PI * np_mod1(p.ref_hz * tti)).sin();
                    x.fill(tone);
                    if has_partials {
                        let pa = partials_at(&parts, i, ti, tti);
                        x.iter_mut().for_each(|v| *v += pa);
                    }
                    for &(f, amp, post) in &extra {
                        let arg = f * if post { ti } else { tti };
                        let v = amp * (2.0 * PI * np_mod1(arg)).sin();
                        x.iter_mut().for_each(|xv| *xv += v);
                    }
                    if let Some(cols) = &prog {
                        wp.frame(cols, tti * srf, &mut h, &mut g, &mut wv);
                        x.iter_mut().zip(&wv).for_each(|(xv, w)| *xv += w);
                    }
                    if let (Some(k), Some(z)) = (hk, &z) {
                        for (c, xv) in x.iter_mut().enumerate() {
                            *xv += z[j * ch + c] * ref_rms * k;
                        }
                    }
                    for (o, &v) in fr.iter_mut().zip(&x) {
                        lpk = lpk.max(v.abs());
                        *o = v as f32;
                    }
                }
                lpk
            })
            .reduce(|| 0.0, f64::max);
        pk = pk.max(bpk);
    }
    drop(prog);
    if pk > 0.99 {
        // scaling in f64 before the f32 cast would be exact; here after it (≤ 1 ulp f32)
        let f = 0.99 / pk;
        data.par_iter_mut()
            .for_each(|v| *v = (*v as f64 * f) as f32);
    }
    let ng = ((n - 1) as f64 / srf / DT).ceil() as usize + 1;
    let tg: Vec<f64> = (0..ng).map(|i| i as f64 * DT).collect();
    let ratio = sp.r_vec(&tg);
    let truth = Curve {
        confidence: vec![1.0; ng],
        valid: vec![true; ng],
        t: tg,
        ratio,
        meta: p.meta(),
    };
    Ok(Synth {
        audio: ts_io::Audio {
            data,
            channels: ch,
            sr: sr as u32,
        },
        truth,
    })
}

/// `tapespeed synth` entry for the CLI: `make_params(preset, kw)`
/// with CLI typing (`overrides` = (field, value) as given on the command line), then
/// write the audio and the truth CSV (default `<out without extension>_truth.csv`).
pub fn cli(
    preset_name: &str,
    overrides: &[(String, String)],
    out: &Path,
    truth: Option<&Path>,
) -> Result<Synth> {
    if preset_name != "custom" && !PRESET_NAMES.contains(&preset_name) {
        return perr(format!("unknown preset {preset_name:?}"));
    }
    let mut p = make_params(preset_name, &[])?;
    for (k, v) in overrides {
        if k == "preset" {
            return perr("preset is not an override");
        }
        p.set_cli(k, v)?;
    }
    let s = run(&p)?;
    ts_io::save(out, &s.audio.data, s.audio.channels, s.audio.sr, None)?;
    let tp = match truth {
        Some(t) => t.to_path_buf(),
        None => {
            let o = out.to_string_lossy();
            let stem = match o.rsplit_once('.') {
                Some((a, _)) => a.to_string(),
                None => o.to_string(),
            };
            Path::new(&format!("{stem}_truth.csv")).to_path_buf()
        }
    };
    write_curve(tp, &s.truth)?;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The long-program noise shaping (overlap-save FIR) matches the full-length FFT
    /// construction: correlation in the interior and band powers.
    #[test]
    fn pink_blocks_matches_full_fft() {
        let (n, srf, lp) = (1usize << 19, 44100.0, 0.45 * 44100.0);
        let mut src = PcgFactory { seed: 3 }.stream(Stream::Program);
        let x = src.standard_normal(n);
        let mut a = x.clone();
        pink_full(&mut a, lp, srf);
        let b = pink_blocks(&x, lp, srf);
        let (lo, hi) = (1 << 15, n - (1 << 15));
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for i in lo..hi {
            ab += a[i] * b[i];
            aa += a[i] * a[i];
            bb += b[i] * b[i];
        }
        let rho = ab / (aa * bb).sqrt();
        let gain_db = 10.0 * (bb / aa).log10();
        assert!(rho > 0.999, "correlation {rho}");
        assert!(gain_db.abs() < 0.05, "gain {gain_db} dB");
    }
}
