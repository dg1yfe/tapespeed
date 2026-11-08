//! Stage 6a: track → 200 Hz grid and gap filling ([`stable`], [`ar_fill`], [`consistent`],
//! [`to_grid`], [`gap_fill`]).
//!
//! Numerical notes:
//! - `lstsq` is an SVD least-squares solve (gelsd algorithm, `ts_dsp::linalg`, faer), `roots` /
//!   `poly` via faer eigenvalues, the AR recursion a plain dot product. Other LAPACK builds agree
//!   to rounding on well-conditioned fits. Ill-conditioned fits ([`GapInfo::ill`]) can differ
//!   up to the clip bound inside the (invalid) gap; valid samples are unaffected.
//! - The [`ar_fill`] context selection `c[len(c) - n:]` with
//!   `n = len(c) - argmin(kc[::-1])` keeps `c[p:]` (`p` = trailing kept run), which is not
//!   necessarily one contiguous kept run; this selection is intentional and part of the
//!   defined behaviour.
//! - Scalar `min`/`max` (NaN-propagating, first argument wins) via [`max2`] / [`min2`].

use num_complex::Complex64;
use ts_dsp::ValueError;
use ts_dsp::interp::Pchip;
use ts_dsp::linalg::{lstsq, poly, roots};
use ts_dsp::ndimage::{Mode, binary_dilation, median_filter, uniform_filter1d};
use ts_dsp::stats::{median, std};

/// Output grid step [s] (`analyze.DT`).
pub const DT: f64 = 0.005;
/// Minimum confidence to keep a soft (invalid) measurement (`analyze.SOFT_CONF`).
pub const SOFT_CONF: f64 = 0.25;

type Res<T> = Result<T, ValueError>;

/// `max(a, b)` = `b if b > a else a` (NaN in `a` propagates; first argument wins ties).
#[inline]
pub(crate) fn max2(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// `min(a, b)` = `b if b < a else a`.
#[inline]
pub(crate) fn min2(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// The track output fields stage 6 reads (borrowed from a `TrackOut`, or from injected
/// reference arrays in the stage tests).
#[derive(Debug, Clone, Copy)]
pub struct TrackGridIn<'a> {
    /// `d["tp"]` = `arange(n) / fs_p`
    pub tp: &'a [f64],
    /// `d["fc"]` coarse track on the fs_p grid [Hz offset]
    pub fc: &'a [f64],
    /// `d["f"]` refined residual [Hz]
    pub f: &'a [f64],
    pub conf: &'a [f64],
    pub valid: &'a [bool],
    pub soft: &'a [bool],
    pub fs_p: f64,
}

/// [`to_grid`] result plus the intermediates the regression tests compare.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// gap-filled ratio on `t`
    pub ratio: Vec<f64>,
    /// confidence (interpolated, not rounded)
    pub cg: Vec<f64>,
    /// valid after [`consistent`]
    pub vg: Vec<bool>,
    /// mask passed to `gap_fill` (kept samples)
    pub keep: Vec<bool>,
    /// ratio before `gap_fill` (interp of `1 + (fc + f) / ref_hz`)
    pub ratio_raw: Vec<f64>,
    /// per-gap AR diagnostics of `gap_fill` (empty when the constant fallback was used)
    pub gaps: Vec<GapInfo>,
}

/// `np.interp(t, xp, fp)` (ts-dsp; numpy's arm64 FMA form of the interpolation).
fn interp_mono(t: &[f64], xp: &[f64], fp: &[f64]) -> Res<Vec<f64>> {
    ts_dsp::stats::interp(t, xp, fp)
}

/// `np.linalg.lstsq(X, y, rcond=None)` for the AR design `X[i, j] = c[i + j]`,
/// `y[i] = c[i + order]` (`sliding_window_view(c, order + 1)` split) → (x, rank).
fn lstsq_ar(c: &[f64], order: usize) -> Res<(Vec<f64>, usize)> {
    let m = c.len() - order;
    let mut xm = Vec::with_capacity(m * order);
    for i in 0..m {
        xm.extend_from_slice(&c[i..i + order]);
    }
    let r = lstsq(&xm, m, order, &c[order..], None)?;
    Ok((r.x, r.rank))
}

// ----------------------------------------------------------------- stable ------------

/// `stable(coef, rmax)`: AR coefficients (oldest first) with poles shrunk to `|z| <= rmax`.
pub fn stable(coef: &[f64], rmax: f64) -> Res<Vec<f64>> {
    // np.r_[1.0, -coef[::-1]]
    let mut p = Vec::with_capacity(coef.len() + 1);
    p.push(1.0);
    p.extend(coef.iter().rev().map(|&c| -c));
    let z = roots(&p)?;
    // np.max(np.abs(z)): NaN propagates
    let mx = z.iter().map(|v| v.norm()).fold(f64::NEG_INFINITY, |m, a| {
        if m.is_nan() || a.is_nan() {
            f64::NAN
        } else {
            m.max(a)
        }
    });
    if z.is_empty() || mx <= rmax {
        return Ok(coef.to_vec());
    }
    let zs: Vec<Complex64> = z
        .iter()
        .map(|&v| {
            let a = v.norm();
            if a > rmax { v / a.max(1e-30) * rmax } else { v }
        })
        .collect();
    // -np.real(np.poly(z))[1:][::-1]
    Ok(poly(&zs)[1..].iter().rev().map(|c| -c.re).collect())
}

// ----------------------------------------------------------------- ar_fill -----------

/// Per-gap diagnostics of [`ar_fill_info`]: `ill` = an AR side was rank-deficient
/// (`lstsq` rank < order) or needed pole reflection in [`stable`]. Such fits are
/// ill-conditioned (the context selection can feed mostly-zero context), so their
/// prediction is reproducible across LAPACK builds only at prediction level, bounded by `lim`
/// (max of the sides' `4 std + 1e-12` clip).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapInfo {
    /// gap = `[a, b)`
    pub a: usize,
    pub b: usize,
    pub ill: bool,
    pub lim: f64,
}

/// One side's AR prediction (`preds`, `ws` entries of [`ar_fill`]), in gap order for the
/// left side and reversed (= gap order) for the right side.
#[allow(clippy::too_many_arguments)]
fn ar_side(
    y: &[f64],
    keep: &[bool],
    a: usize,
    b: usize,
    side: u8,
    ctx: usize,
    order: usize,
    reach: usize,
) -> Res<(Vec<f64>, Vec<f64>, bool, f64)> {
    let l = b - a;
    // c, kc: context in "towards the gap" order; lo..hi = its memory range in y
    let (lo, hi) = if side == 0 {
        (a.saturating_sub(ctx), a)
    } else {
        (b, (b + ctx).min(y.len()))
    };
    let at = |i: usize| -> usize { if side == 0 { lo + i } else { hi - 1 - i } };
    let len = hi - lo;
    let all = (0..len).all(|i| keep[at(i)]);
    let n = if all {
        len
    } else {
        // argmin(kc[::-1]) = number of trailing True of kc
        let p = (0..len).rev().take_while(|&i| keep[at(i)]).count();
        len - p
    };
    if n < 3 * order {
        return Ok((vec![0.0; l], vec![0.0; l], false, 0.0));
    }
    let off = len - n; // c = c[len(c) - n:]
    let c: Vec<f64> = (off..len).map(|i| y[at(i)]).collect();
    // sliding_window_view(c, order + 1) split into X[:, :-1], X[:, -1]
    let (x, rank) = lstsq_ar(&c, order)?;
    let coef = stable(&x, 0.999)?;
    let ill = rank < order || coef != x;
    // np.std of the (reversed for side 1) view reduces in logical order (checked: equals
    // np.std of a contiguous copy, not of the memory-order slice)
    let lim = 4.0 * std(&c) + 1e-12;
    let steps = l.min(2 * reach);
    let mut buf = Vec::with_capacity(order + steps);
    buf.extend_from_slice(&c[n - order..]);
    for _ in 0..steps {
        let w = &buf[buf.len() - order..];
        let d: f64 = coef.iter().zip(w).map(|(a, b)| a * b).sum();
        buf.push(min2(max2(d, -lim), lim));
    }
    let mut pr: Vec<f64> = buf[order..].to_vec();
    pr.resize(l, 0.0);
    let rf = reach as f64;
    let mut wt: Vec<f64> = (1..=l)
        .map(|k| (2.0 - k as f64 / rf).clamp(0.0, 1.0))
        .collect();
    if side == 1 {
        pr.reverse();
        wt.reverse();
    }
    Ok((pr, wt, ill, lim))
}

/// `ar_fill(y, keep, ctx, order, reach)` → (prediction, weight). Defaults:
/// `ctx=400, order=24, reach=100` (see [`ar_fill_default`]).
pub fn ar_fill(
    y: &[f64],
    keep: &[bool],
    ctx: usize,
    order: usize,
    reach: usize,
) -> Res<(Vec<f64>, Vec<f64>)> {
    ar_fill_info(y, keep, ctx, order, reach).map(|(p, w, _)| (p, w))
}

/// [`ar_fill`] plus per-gap diagnostics.
pub fn ar_fill_info(
    y: &[f64],
    keep: &[bool],
    ctx: usize,
    order: usize,
    reach: usize,
) -> Res<(Vec<f64>, Vec<f64>, Vec<GapInfo>)> {
    let mut info = Vec::new();
    let n = y.len();
    let mut out = vec![0.0; n];
    let mut wout = vec![0.0; n];
    let mut i = 0;
    while i < n {
        if keep[i] {
            i += 1;
            continue;
        }
        let a = i;
        while i < n && !keep[i] {
            i += 1;
        }
        let b = i;
        let l = b - a;
        let (p0, w0, ill0, lim0) = ar_side(y, keep, a, b, 0, ctx, order, reach)?;
        let (p1, w1, ill1, lim1) = ar_side(y, keep, a, b, 1, ctx, order, reach)?;
        info.push(GapInfo {
            a,
            b,
            ill: ill0 || ill1,
            lim: lim0.max(lim1),
        });
        let lf = l as f64;
        for j in 0..l {
            let x = (j as f64 + 0.5) / lf;
            let ang = std::f64::consts::PI / 2.0 * x;
            let (c, s) = (ang.cos(), ang.sin());
            let cl = c * c * ((w0[j] > 0.0) as u8 as f64);
            let cr = s * s * ((w1[j] > 0.0) as u8 as f64);
            let tot = cl + cr;
            if tot > 0.0 {
                let d = tot.max(1e-12);
                out[a + j] = (cl * p0[j] + cr * p1[j]) / d;
                wout[a + j] = (cl * w0[j] + cr * w1[j]) / d;
            }
        }
    }
    Ok((out, wout, info))
}

/// [`ar_fill`] with the defaults `ctx=400, order=24, reach=100`.
pub fn ar_fill_default(y: &[f64], keep: &[bool]) -> Res<(Vec<f64>, Vec<f64>)> {
    ar_fill(y, keep, 400, 24, 100)
}

// ---------------------------------------------------------------- gap_fill -----------

/// `gap_fill(t, ratio, keep)`: smooth (0.5 s) trend through kept samples, PCHIP across
/// gaps, plus AR prediction of the residual. Needs `keep.sum() >= 2` (the PCHIP
/// interpolation errors otherwise). Public: also used by model II.
pub fn gap_fill(t: &[f64], ratio: &[f64], keep: &[bool]) -> Res<Vec<f64>> {
    gap_fill_info(t, ratio, keep).map(|(r, _)| r)
}

/// [`gap_fill`] plus the per-gap AR diagnostics ([`GapInfo`]).
pub fn gap_fill_info(t: &[f64], ratio: &[f64], keep: &[bool]) -> Res<(Vec<f64>, Vec<GapInfo>)> {
    let n = t.len();
    if ratio.len() != n || keep.len() != n {
        return Err(ValueError("gap_fill: length mismatch".into()));
    }
    let k: Vec<f64> = keep.iter().map(|&b| b as u8 as f64).collect();
    let w = 1usize.max((0.5 / DT) as usize);
    let rk: Vec<f64> = ratio.iter().zip(&k).map(|(r, k)| r * k).collect();
    let num = uniform_filter1d(&rk, w, Mode::Reflect)?;
    let den = uniform_filter1d(&k, w, Mode::Reflect)?;
    let (mut tv, mut trv) = (Vec::new(), Vec::new());
    for i in 0..n {
        if keep[i] {
            tv.push(t[i]);
            trv.push(num[i] / den[i].max(1e-9));
        }
    }
    if tv.len() < 2 {
        return Err(ValueError(
            "gap_fill: at least 2 kept samples are needed".into(),
        ));
    }
    let pc = Pchip::new(&tv, &trv)?;
    let (t0, t1) = (tv[0], tv[tv.len() - 1]);
    let base: Vec<f64> = t.iter().map(|&v| pc.eval_one(v.clamp(t0, t1))).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| if keep[i] { ratio[i] - base[i] } else { 0.0 })
        .collect();
    let (pr, wt, info) = ar_fill_info(&y, keep, 400, 24, 100)?;
    let r = (0..n)
        .map(|i| {
            if keep[i] {
                ratio[i]
            } else {
                base[i] + wt[i] * pr[i]
            }
        })
        .collect();
    Ok((r, info))
}

// ----------------------------------------------------------------- consistent --------

/// `consistent(ratio, mask, ref_mask, win=11.0, k=4.0, floor=1.5e-3)`. `ref_mask = None`
/// means the reference mask is `mask` itself.
pub fn consistent(ratio: &[f64], mask: &[bool], ref_mask: Option<&[bool]>) -> Res<Vec<bool>> {
    consistent_with(ratio, mask, ref_mask, 11.0, 4.0, 1.5e-3)
}

/// [`consistent`] with explicit parameters.
pub fn consistent_with(
    ratio: &[f64],
    mask: &[bool],
    ref_mask: Option<&[bool]>,
    win: f64,
    k: f64,
    floor: f64,
) -> Res<Vec<bool>> {
    let len = ratio.len();
    let mut out = mask.to_vec();
    let nb = 1usize.max((1.0 / DT) as usize);
    let nblk = len.div_ceil(nb);
    let h = (win as i64 / 2) as usize;
    let mf = 1usize.max((0.5 / DT) as usize) | 1;
    let dil = (0.3 / DT) as usize | 1;
    for _ in 0..2 {
        let m: Vec<bool> = match ref_mask {
            Some(r) => r.iter().zip(&out).map(|(a, b)| *a && *b).collect(),
            None => out.clone(),
        };
        let mut bm = vec![f64::NAN; nblk];
        let mut seg = Vec::with_capacity(nb);
        for (i, b) in bm.iter_mut().enumerate() {
            seg.clear();
            for j in i * nb..((i + 1) * nb).min(len) {
                if m[j] {
                    seg.push(ratio[j]);
                }
            }
            if seg.len() > nb / 4 {
                *b = median(&seg);
            }
        }
        let ok: Vec<bool> = bm.iter().map(|v| v.is_finite()).collect();
        if ok.iter().filter(|&&b| b).count() < 3 {
            return Ok(out);
        }
        let tb: Vec<f64> = (0..nblk)
            .map(|i| {
                let lo = i.saturating_sub(h);
                let hi = (i + h + 1).min(nblk);
                let v: Vec<f64> = (lo..hi).filter(|&j| ok[j]).map(|j| bm[j]).collect();
                if v.is_empty() { f64::NAN } else { median(&v) }
            })
            .collect();
        let (xi, yi): (Vec<f64>, Vec<f64>) = (0..nblk)
            .filter(|&i| tb[i].is_finite())
            .map(|i| (i as f64, tb[i]))
            .unzip();
        let ar: Vec<f64> = (0..nblk).map(|i| i as f64).collect();
        let tb = interp_mono(&ar, &xi, &yi)?;
        let xc: Vec<f64> = (0..nblk).map(|i| (i as f64 + 0.5) * nb as f64).collect();
        let tl: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let trend = interp_mono(&tl, &xc, &tb)?;
        let dev: Vec<f64> = ratio.iter().zip(&trend).map(|(r, t)| r - t).collect();
        let ad: Vec<f64> = (0..len).filter(|&i| m[i]).map(|i| dev[i].abs()).collect();
        let tol = if ad.is_empty() {
            floor
        } else {
            max2(k * 1.4826 * median(&ad), floor)
        };
        let ds = median_filter(&dev, mf, Mode::Nearest)?;
        let big: Vec<bool> = ds.iter().map(|v| v.abs() > tol).collect();
        let rej = binary_dilation(&big, dil)?;
        for (o, r) in out.iter_mut().zip(&rej) {
            *o &= !r;
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------- to_grid -----------

/// `to_grid(d, t, ref_hz)` → ratio, cg, vg (+ keep and the raw ratio for regression tests).
pub fn to_grid(d: &TrackGridIn, t: &[f64], ref_hz: f64) -> Res<Grid> {
    let np_ = d.tp.len();
    if np_ == 0 {
        return Err(ValueError("_to_grid: empty track".into()));
    }
    let r: Vec<f64> =
        d.fc.iter()
            .zip(d.f)
            .map(|(fc, f)| 1.0 + (fc + f) / ref_hz)
            .collect();
    let ratio_raw = interp_mono(t, d.tp, &r)?;
    let cg = interp_mono(t, d.tp, d.conf)?;
    let ix: Vec<usize> = t
        .iter()
        .map(|&v| {
            let r = (v * d.fs_p).round_ties_even();
            // clip(round(.).astype(int), 0, n - 1)
            if r.is_nan() || r <= 0.0 {
                0
            } else {
                (r as usize).min(np_ - 1)
            }
        })
        .collect();
    let v0: Vec<bool> = ix.iter().map(|&i| d.valid[i]).collect();
    let c1 = consistent(&ratio_raw, &v0, None)?;
    let vg: Vec<bool> = v0.iter().zip(&c1).map(|(a, b)| *a && *b).collect();
    let sg: Vec<bool> = ix.iter().map(|&i| d.soft[i]).collect();
    let vs: Vec<bool> = vg.iter().zip(&sg).map(|(a, b)| *a || *b).collect();
    let c2 = consistent(&ratio_raw, &vs, Some(&vg))?;
    let keep: Vec<bool> = (0..t.len())
        .map(|i| vg[i] || (sg[i] && cg[i] >= SOFT_CONF && c2[i]))
        .collect();
    let (ratio, gaps) = if keep.iter().filter(|&&b| b).count() >= 2 {
        gap_fill_info(t, &ratio_raw, &keep)?
    } else {
        (vec![1.0 + median(d.fc) / ref_hz; t.len()], vec![])
    };
    Ok(Grid {
        ratio,
        cg,
        vg,
        keep,
        ratio_raw,
        gaps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An exploding AR context stays finite; pole shrinking clips to `rmax`.
    #[test]
    fn ar_fill_finite() {
        let mut y = vec![0.0; 6000];
        let mut keep = vec![false; 6000];
        for i in 0..100 {
            y[i] = 1e-4 * 1.2f64.powi(i as i32) / 1.2f64.powi(100);
            keep[i] = true;
        }
        let (out, w) = ar_fill_default(&y, &keep).unwrap();
        assert!(out.iter().chain(&w).all(|v| v.is_finite()));
        let mut c = vec![0.0; 24];
        c[23] = 1.2;
        let co = stable(&c, 0.999).unwrap();
        let mut p = vec![1.0];
        p.extend(co.iter().rev().map(|v| -v));
        let mx = roots(&p)
            .unwrap()
            .iter()
            .map(|z| z.norm())
            .fold(0.0, f64::max);
        assert!(mx <= 0.999 + 1e-9, "{mx}");
        // expected: [-0.0] * 23 + [0.999]
        assert!((co[23] - 0.999).abs() < 1e-12);
        assert!(co[..23].iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn stable_passthrough() {
        let c = [0.1, -0.2, 0.3];
        assert_eq!(stable(&c, 0.999).unwrap(), c.to_vec());
        assert_eq!(stable(&[], 0.999).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn minmax_nan() {
        assert!(max2(f64::NAN, 1.0).is_nan());
        assert_eq!(max2(1.0, f64::NAN), 1.0);
        assert!(min2(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn consistent_few_blocks_returns_mask() {
        let r = vec![1.0; 300];
        let m = vec![true; 300];
        assert_eq!(consistent(&r, &m, None).unwrap(), m);
    }
}
