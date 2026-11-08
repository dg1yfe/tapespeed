//! Hierarchical mechanical speed model.
//!
//! - [`law`]: `Comp` frequency laws `f(t) = a + b / sqrt(1 + beta t)`, `fit_law` (BIC, fminbound).
//! - [`ident`]: `masked_spectrogram`, `ridges`, `link`, `identify`.
//! - [`fit`]: the L1/L2/L3 weighted least-squares fit (sparse normal equations, IRLS).
//! - this file: [`Model`] prediction, `level_outliers`, `fit_hier`, `explained`.
//!
//! Model II (gap filling, aided tracking, hold-out, `model_step`) is a later work package.

pub mod aided;
pub mod fit;
pub mod gaps;
pub mod ident;
pub mod law;
pub mod rng;
pub mod step;

use indexmap::IndexMap;
use num_complex::Complex64;
use ts_dsp::ValueError;
use ts_dsp::bspline;
use ts_dsp::ndimage::{binary_closing, binary_dilation};
use ts_dsp::stats::{mean, median, nanmedian};

pub use fit::{FitOpts, fit};
pub use ident::{Cand, Family, IdentInfo, IdentOpts, identify, masked_spectrogram};
pub use law::{Comp, Kind, Law, fit_law};

/// Curve grid step [s] (`model.DT`).
pub const DT: f64 = 0.005;

/// Amplitude of one (component, harmonic): complex scalar (L2) or values on the L3 knots
/// `span.0 + Ta * i`.
#[derive(Debug, Clone, PartialEq)]
pub enum Amp {
    L2(Complex64),
    L3(Vec<Complex64>),
}

/// `Model.stats` of a level ≥ 2 fit.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FitStats {
    pub n: usize,
    pub outlier_frac: f64,
}

/// Fitted speed model: level, base ratio, span, components, drift and amplitudes.
#[derive(Debug, Clone)]
pub struct Model {
    pub level: u8,
    pub r0: f64,
    pub span: (f64, f64),
    pub comps: Vec<Comp>,
    pub td: f64,
    pub drift_knots: Option<Vec<f64>>,
    pub drift_coef: Option<Vec<f64>>,
    pub ta: f64,
    /// `(k, h)` → amplitude, in insertion order (part of the report format).
    pub amp: IndexMap<(usize, u32), Amp>,
    pub stats: Option<FitStats>,
}

/// `hat(t, t0, Ta, M)`: knot index `m` and fraction `u` of the piecewise-linear basis.
#[inline]
pub(crate) fn hat(t: f64, t0: f64, ta: f64, m_knots: usize) -> (usize, f64) {
    let u = ((t - t0) / ta).clamp(0.0, m_knots as f64 - 1.0 - 1e-9);
    let m = u.floor();
    (m as usize, u - m)
}

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

impl Model {
    fn phase_of(&self, c: &Comp, t: f64) -> f64 {
        if self.level >= 3 {
            c.phase_at(t)
        } else {
            c.f_mean * t
        }
    }

    /// `Model.drift(t)`: B-spline (extrapolate=False) of `clip(t, t0, t1)`; zeros without
    /// drift coefficients.
    pub fn drift(&self, t: &[f64]) -> Vec<f64> {
        match (&self.drift_knots, &self.drift_coef) {
            (Some(kn), Some(c)) => {
                let (t0, t1) = self.span;
                let tc: Vec<f64> = t.iter().map(|&v| v.clamp(t0, t1)).collect();
                bspline::eval(kn, c, 3, &tc)
            }
            _ => vec![0.0; t.len()],
        }
    }

    fn component_at(&self, k: usize, t: f64, h: Option<u32>) -> f64 {
        let c = &self.comps[k];
        let ph = TWO_PI * self.phase_of(c, t);
        let mut y = 0.0;
        let hs: &[u32] = match &h {
            Some(h) => std::slice::from_ref(h),
            None => &c.harmonics,
        };
        for &hh in hs {
            let Some(a) = self.amp.get(&(k, hh)) else {
                continue;
            };
            let cv = match a {
                Amp::L2(z) => *z,
                Amp::L3(v) => {
                    let (m, u) = hat(t, self.span.0, self.ta, v.len());
                    v[m] * (1.0 - u) + v[m + 1] * u
                }
            };
            let x = hh as f64 * ph;
            y += cv.re * x.cos() - cv.im * x.sin();
        }
        y
    }

    /// `Model.component(k, t, h)`.
    pub fn component(&self, k: usize, t: &[f64], h: Option<u32>) -> Vec<f64> {
        t.iter().map(|&v| self.component_at(k, v, h)).collect()
    }

    /// `Model.predict(t)`: `r0 (+ drift + Σ components for level ≥ 2)`.
    pub fn predict(&self, t: &[f64]) -> Vec<f64> {
        let d = if self.level >= 2 {
            Some(self.drift(t))
        } else {
            None
        };
        t.iter()
            .enumerate()
            .map(|(i, &tv)| {
                let mut y = self.r0;
                if let Some(d) = &d {
                    y += d[i];
                    for k in 0..self.comps.len() {
                        y += self.component_at(k, tv, None);
                    }
                }
                y
            })
            .collect()
    }
}

// ---------------------------------------------------------------- level_outliers -----

/// `nanmed(a, n)` of `level_outliers`: NaN-padded sliding `np.nanmedian` (odd window).
fn nanmed_window(a: &[f64], n: f64) -> Vec<f64> {
    let n = (n as i64).max(1) as usize | 1;
    let h = n / 2;
    let len = a.len();
    let mut buf = Vec::with_capacity(n);
    (0..len)
        .map(|i| {
            buf.clear();
            let lo = i.saturating_sub(h);
            let hi = (i + h).min(len - 1);
            buf.extend(a[lo..=hi].iter().copied().filter(|v| !v.is_nan()));
            nanmedian(&buf)
        })
        .collect()
}

/// Parameters of [`level_outliers`] (with their defaults).
#[derive(Debug, Clone, Copy)]
pub struct OutlierOpts {
    pub thr: f64,
    pub blk: f64,
    pub short: f64,
    pub long: f64,
    pub dil: f64,
    pub close: f64,
}

impl Default for OutlierOpts {
    fn default() -> Self {
        OutlierOpts {
            thr: 1e-3,
            blk: 0.25,
            short: 1.0,
            long: 30.0,
            dil: 1.0,
            close: 6.0,
        }
    }
}

/// `level_outliers(t, r, w)`: samples whose 1 s block-median level deviates from the 30 s
/// level by more than `max(thr, 6 MAD)` (reference false locks); blocks closer than `close`
/// joined, dilated by `dil`.
pub fn level_outliers(
    t: &[f64],
    r: &[f64],
    w: &[f64],
    o: &OutlierOpts,
) -> Result<Vec<bool>, ValueError> {
    let n = t.len();
    let mut out = vec![false; n];
    if n < 2 {
        return Ok(out);
    }
    let dt = t[1] - t[0];
    let k = ((o.blk / dt) as i64).max(1) as usize;
    let nb = n / k;
    if nb == 0 {
        return Ok(out);
    }
    let bm: Vec<f64> = (0..nb)
        .map(|b| {
            let rs = &r[b * k..(b + 1) * k];
            let ws = &w[b * k..(b + 1) * k];
            let cnt = ws.iter().filter(|&&v| v > 0.0).count();
            if cnt as f64 / k as f64 > 0.5 {
                let v: Vec<f64> = rs
                    .iter()
                    .zip(ws)
                    .filter(|&(_, &wv)| wv > 0.0)
                    .map(|(&x, _)| x)
                    .collect();
                median(&v)
            } else {
                f64::NAN
            }
        })
        .collect();
    let s_ = nanmed_window(&bm, o.short / o.blk);
    let l_ = nanmed_window(&bm, o.long / o.blk);
    let d: Vec<f64> = s_.iter().zip(&l_).map(|(a, b)| a - b).collect();
    let dok: Vec<f64> = d.iter().copied().filter(|v| v.is_finite()).collect();
    if dok.len() < 10 {
        return Ok(out);
    }
    let md = median(&dok);
    let dev: Vec<f64> = dok.iter().map(|v| (v - md).abs()).collect();
    let mad = 1.4826 * median(&dev);
    // max(thr, 6 * mad): first argument wins ties
    let thr = if 6.0 * mad > o.thr { 6.0 * mad } else { o.thr };
    let bad: Vec<bool> = d.iter().map(|v| v.is_finite() && v.abs() > thr).collect();
    let kc = (o.close / o.blk) as usize | 1;
    let mut padded = vec![false; nb + 2 * kc];
    padded[kc..kc + nb].copy_from_slice(&bad);
    let closed = binary_closing(&padded, kc)?;
    let bad = binary_dilation(&closed[kc..kc + nb], 2 * (o.dil / o.blk) as usize + 1)?;
    for (b, &v) in bad.iter().enumerate() {
        if v {
            out[b * k..(b + 1) * k].iter_mut().for_each(|x| *x = true);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- fit_hier -----------

/// Options of [`fit_hier`] (`fit_kw` carries `lam`, `irls`, `dec`, `lam_d`, forwarded to the
/// m2/m3 fits).
#[derive(Debug, Clone)]
pub struct HierOpts {
    pub fmax: f64,
    pub td: f64,
    pub ta: f64,
    pub comps: Option<Vec<Comp>>,
    pub ident: IdentOpts,
    pub reject: bool,
    pub fit_kw: FitOpts,
}

impl Default for HierOpts {
    fn default() -> Self {
        HierOpts {
            fmax: 22.0,
            td: 8.0,
            ta: 20.0,
            comps: None,
            ident: IdentOpts::default(),
            reject: true,
            fit_kw: FitOpts::default(),
        }
    }
}

/// Result of [`fit_hier`].
#[derive(Debug, Clone)]
pub struct Hier {
    pub m1: Model,
    pub m2: Model,
    pub m3: Model,
    pub comps: Vec<Comp>,
    pub info: Option<IdentInfo>,
    pub rejected: Vec<bool>,
}

/// `fit_hier(t, r, w, fmax, Td, Ta, comps, ident_kw, reject, fit_kw)`: L1 → identify on
/// `r - r0 - drift` → L2 → L3.
pub fn fit_hier(t: &[f64], r: &[f64], w: &[f64], o: &HierOpts) -> Result<Hier, ValueError> {
    let rej: Vec<bool> = if o.reject {
        let lo = level_outliers(t, r, w, &OutlierOpts::default())?;
        lo.iter().zip(w).map(|(&a, &wv)| a && wv > 0.0).collect()
    } else {
        vec![false; t.len()]
    };
    let w: Vec<f64> = w
        .iter()
        .zip(&rej)
        .map(|(&wv, &j)| if j { 0.0 } else { wv })
        .collect();
    let l1 = FitOpts {
        level: 1,
        ..FitOpts::default()
    };
    let m1 = fit(t, r, &w, &[], &l1)?;
    let mut info = None;
    let comps = match &o.comps {
        Some(c) => c.clone(),
        None => {
            let od = FitOpts {
                level: 2,
                td: o.td,
                r0: Some(m1.r0),
                ..FitOpts::default()
            };
            let md = fit(t, r, &w, &[], &od)?;
            let d = md.drift(t);
            let y: Vec<f64> = r
                .iter()
                .zip(&d)
                .map(|(&rv, &dv)| (rv - m1.r0) - dv)
                .collect();
            let io = IdentOpts {
                fmax: o.fmax,
                ..o.ident.clone()
            };
            let (c, inf) = identify(t, &y, &w, &io)?;
            info = Some(inf);
            c
        }
    };
    let o2 = FitOpts {
        level: 2,
        td: o.td,
        r0: Some(m1.r0),
        ..o.fit_kw
    };
    let m2 = fit(t, r, &w, &comps, &o2)?;
    let o3 = FitOpts {
        level: 3,
        td: o.td,
        ta: o.ta,
        r0: Some(m1.r0),
        ..o.fit_kw
    };
    let m3 = fit(t, r, &w, &comps, &o3)?;
    Ok(Hier {
        m1,
        m2,
        m3,
        comps,
        info,
        rejected: rej,
    })
}

/// `explained(models, t, r, w)`: `ms_raw`, `ms_<L>` (mean square of r − prediction on
/// w > 0) and `expl_<L> = 1 − ms_L / ms_m1` (keys in report order).
pub fn explained(
    models: &[(&str, &Model)],
    t: &[f64],
    r: &[f64],
    w: &[f64],
) -> IndexMap<String, f64> {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| w[i] > 0.0).collect();
    let tv: Vec<f64> = idx.iter().map(|&i| t[i]).collect();
    let rv: Vec<f64> = idx.iter().map(|&i| r[i]).collect();
    let mut out = IndexMap::new();
    let sq: Vec<f64> = rv.iter().map(|v| (v - 1.0) * (v - 1.0)).collect();
    out.insert("ms_raw".to_string(), mean(&sq));
    let mut base = f64::NAN;
    for (name, m) in models {
        let p = m.predict(&tv);
        let sq: Vec<f64> = rv.iter().zip(&p).map(|(a, b)| (a - b) * (a - b)).collect();
        let ms = mean(&sq);
        out.insert(format!("ms_{name}"), ms);
        if *name == "m1" {
            base = ms;
        }
    }
    for (name, _) in models {
        let v = if *name != "m1" {
            1.0 - out[&format!("ms_{name}")] / base
        } else {
            0.0
        };
        out.insert(format!("expl_{name}"), v);
    }
    out
}
