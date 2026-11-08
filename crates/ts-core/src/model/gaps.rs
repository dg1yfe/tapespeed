//! Gap filling and curve composition:
//! `spans`, `pchip_fill`, `model_fill`, `long_gaps`, `fill_long_gaps`, `time_integral`,
//! `compose`.

use ts_dsp::ValueError;
use ts_dsp::interp::Pchip;
use ts_dsp::ndimage::{Mode, uniform_filter1d};
use ts_dsp::spline::{PPoly, cubic_spline};
use ts_dsp::stats::{interp_one_by, mean, median};

use super::{FitOpts, Model, fit};
use crate::curve::{Curve, MetaValue};

type Res<T> = Result<T, ValueError>;

/// `spans(mask)`: index spans `[a, b)` of consecutive true samples.
pub fn spans(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &m) in mask.iter().enumerate() {
        match (m, start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                out.push((a, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(a) = start {
        out.push((a, mask.len()));
    }
    out
}

/// `pchip_fill(t, r, valid)`: invalid samples from a PCHIP through the valid ones (held
/// constant outside the valid range); a copy of `r` with fewer than 2 valid samples.
pub fn pchip_fill(t: &[f64], r: &[f64], valid: &[bool]) -> Res<Vec<f64>> {
    let (tv, rv): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(r)
        .zip(valid)
        .filter(|&(_, &v)| v)
        .map(|((&a, &b), _)| (a, b))
        .unzip();
    if tv.len() < 2 {
        return Ok(r.to_vec());
    }
    let p = Pchip::new(&tv, &rv)?;
    let (lo, hi) = (tv[0], tv[tv.len() - 1]);
    Ok((0..t.len())
        .map(|i| {
            if valid[i] {
                r[i]
            } else {
                p.eval_one(t[i].clamp(lo, hi))
            }
        })
        .collect())
}

/// `model_fill(t, r, valid, model, edge)`: invalid stretches replaced by the model prediction,
/// blended linearly so the fill meets the measured offsets (mean over `edge` s each side).
pub fn model_fill(t: &[f64], r: &[f64], valid: &[bool], model: &Model, edge: f64) -> Vec<f64> {
    let pm = model.predict(t);
    model_fill_pred(t, r, valid, &pm, edge)
}

/// [`model_fill`] with the prediction `pm = model.predict(t)` given.
pub fn model_fill_pred(t: &[f64], r: &[f64], valid: &[bool], pm: &[f64], edge: f64) -> Vec<f64> {
    let n = t.len();
    let mut out = r.to_vec();
    if n < 2 {
        return out;
    }
    let d: Vec<f64> = r.iter().zip(pm).map(|(a, b)| a - b).collect();
    let k = ((edge / (t[1] - t[0])) as i64).max(1) as usize;
    let inv: Vec<bool> = valid.iter().map(|v| !v).collect();
    for (s, e) in spans(&inv) {
        let dl = (s > 0).then(|| mean(&d[s.saturating_sub(k)..s]));
        let dr = (e < n).then(|| mean(&d[e..(e + k).min(n)]));
        let (dl, dr) = match (dl, dr) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => (a, a),
            (None, Some(b)) => (b, b),
            (None, None) => (0.0, 0.0),
        };
        let len = (e - s).max(1) as f64;
        for i in s..e {
            let u = ((i - s) as f64 + 0.5) / len;
            out[i] = pm[i] + dl + (dr - dl) * u;
        }
    }
    out
}

/// No reference this long [s] → slow fallback (`LONG_GAP_S`).
pub const LONG_GAP_S: f64 = 20.0;

/// `long_gaps(t, valid, min_s, win, dens)`: spans `[a, b)` whose valid density over `win` s
/// stays below `dens` for at least `min_s` s.
pub fn long_gaps(
    t: &[f64],
    valid: &[bool],
    min_s: f64,
    win: f64,
    dens: f64,
) -> Res<Vec<(usize, usize)>> {
    if t.len() < 2 {
        return Ok(vec![]);
    }
    let dt = t[1] - t[0];
    let vf: Vec<f64> = valid.iter().map(|&v| v as u8 as f64).collect();
    let size = ((win / dt) as i64).max(1) as usize;
    let d = uniform_filter1d(&vf, size, Mode::Constant(0.0))?;
    let low: Vec<bool> = d.iter().map(|&v| v < dens).collect();
    Ok(spans(&low)
        .into_iter()
        .filter(|&(a, b)| (b - a) as f64 * dt >= min_s)
        .collect())
}

/// [`long_gaps`] with the defaults (20 s, 10 s window, 2 % density).
pub fn long_gaps_default(t: &[f64], valid: &[bool]) -> Res<Vec<(usize, usize)>> {
    long_gaps(t, valid, LONG_GAP_S, 10.0, 0.02)
}

/// `fill_long_gaps(t, r, valid, gaps, edge, ramp)`: inside long gaps (valid → false) linear
/// between the edge levels (median of valid samples within `edge` s), at the file ends a
/// cosine ramp from the edge level to L1 over `ramp` s. Returns `(ratio, valid, L1)`.
pub fn fill_long_gaps(
    t: &[f64],
    r: &[f64],
    valid: &[bool],
    gaps: &[(usize, usize)],
    edge: f64,
    ramp: f64,
) -> Res<(Vec<f64>, Vec<bool>, f64)> {
    let dt = t[1] - t[0];
    let mut r = r.to_vec();
    let mut v = valid.to_vec();
    for &(a, b) in gaps {
        v[a..b].iter_mut().for_each(|x| *x = false);
    }
    if !v.iter().any(|&x| x) {
        let l1 = median(&r);
        return Ok((r, v, l1));
    }
    let w: Vec<f64> = v.iter().map(|&x| x as u8 as f64).collect();
    let l1 = fit(
        t,
        &r,
        &w,
        &[],
        &FitOpts {
            level: 1,
            ..FitOpts::default()
        },
    )?
    .r0;
    let k = (edge / dt) as usize;
    let n = r.len();
    let sel = |r: &[f64], lo: usize, hi: usize| -> Vec<f64> {
        (lo..hi).filter(|&i| v[i]).map(|i| r[i]).collect()
    };
    for &(a, b) in gaps {
        let lo = sel(&r, a.saturating_sub(k), a);
        let hi = sel(&r, b, (b + k).min(n));
        let e0 = (!lo.is_empty()).then(|| median(&lo));
        let e1 = (!hi.is_empty()).then(|| median(&hi));
        let len = (b - a) as f64;
        match (e0, e1) {
            (Some(e0), Some(e1)) => {
                for (j, ri) in r[a..b].iter_mut().enumerate() {
                    let u = (j as f64 + 0.5) / len;
                    *ri = e0 + (e1 - e0) * u;
                }
            }
            _ => {
                let lead = e0.is_some();
                let e = e0.or(e1).unwrap_or(l1);
                for (j, ri) in r[a..b].iter_mut().enumerate() {
                    let d = if lead {
                        j as f64 * dt
                    } else {
                        (b - 1 - (a + j)) as f64 * dt
                    };
                    let g = 0.5 + 0.5 * (std::f64::consts::PI * (d / ramp).clamp(0.0, 1.0)).cos();
                    *ri = l1 + (e - l1) * g;
                }
            }
        }
    }
    Ok((r, v, l1))
}

/// [`fill_long_gaps`] with the defaults (edge 10 s, ramp 30 s).
pub fn fill_long_gaps_default(
    t: &[f64],
    r: &[f64],
    valid: &[bool],
    gaps: &[(usize, usize)],
) -> Res<(Vec<f64>, Vec<bool>, f64)> {
    fill_long_gaps(t, r, valid, gaps, 10.0, 30.0)
}

/// `time_integral(t, r)`: `T(t) = ∫_0^t r` (not-a-knot cubic spline antiderivative).
pub fn time_integral(t: &[f64], r: &[f64]) -> Res<PPoly> {
    Ok(cubic_spline(t, r)?.antiderivative())
}

/// `compose(c1, c2)`: total speed curve for resampling once from the original,
/// `r_tot(t) = r1(t) · r2(T1(t))`, `valid = valid1 & valid2(T1(t))`; `c2` lives on `c1`'s
/// output time.
pub fn compose(c1: &Curve, c2: &Curve) -> Res<Curve> {
    let t1 = time_integral(&c1.t, &c1.ratio)?;
    let mut h = 0;
    let tau: Vec<f64> = c1.t.iter().map(|&x| t1.eval_hint(x, &mut h)).collect();
    let s2 = cubic_spline(&c2.t, &c2.ratio)?;
    let (lo, hi) = (c2.t[0], c2.t[c2.t.len() - 1]);
    let v2f: Vec<f64> = c2.valid.iter().map(|&v| v as u8 as f64).collect();
    let (mut h1, mut h2, mut h3) = (0, 0, 0);
    let n = c1.t.len();
    let mut ratio = Vec::with_capacity(n);
    let mut conf = Vec::with_capacity(n);
    let mut valid = Vec::with_capacity(n);
    for (i, &x) in tau.iter().enumerate() {
        let r2 = s2.eval_hint(x.clamp(lo, hi), &mut h1);
        let v2 = interp_one_by(x, &c2.t, |j| v2f[j], &mut h2) > 0.5;
        let q2 = interp_one_by(x, &c2.t, |j| c2.confidence[j], &mut h3);
        ratio.push(c1.ratio[i] * r2);
        // np.minimum: NaN propagates
        let q1 = c1.confidence[i];
        conf.push(if q1.is_nan() || q2.is_nan() {
            f64::NAN
        } else {
            q1.min(q2)
        });
        valid.push(c1.valid[i] && v2);
    }
    let mut meta = c1.meta.clone();
    meta.insert("composed".into(), MetaValue::Str("r1*r2(T1)".into()));
    Ok(Curve {
        t: c1.t.clone(),
        ratio,
        confidence: conf,
        valid,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_edges() {
        assert_eq!(spans(&[]), vec![]);
        assert_eq!(spans(&[true, true, false, true]), vec![(0, 2), (3, 4)]);
        assert_eq!(spans(&[false, true, true, false]), vec![(1, 3)]);
    }

    #[test]
    fn pchip_fill_constant_ends() {
        let t = [0.0, 1.0, 2.0, 3.0, 4.0];
        let r = [9.0, 1.0, 2.0, 3.0, 9.0];
        let v = [false, true, true, true, false];
        let p = pchip_fill(&t, &r, &v).unwrap();
        assert_eq!(p, vec![1.0, 1.0, 2.0, 3.0, 3.0]);
        let one = [false, true, false, false, false];
        assert_eq!(pchip_fill(&t, &r, &one).unwrap(), r.to_vec());
    }
}
