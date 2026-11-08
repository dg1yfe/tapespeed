//! The speed plot written next to every correction.
use crate::PlotError;
use crate::draw::*;
use crate::numbers::{SpeedNumbers, meta_f64, meta_str, pct2, speed_plot_numbers};
use plotters::style::text_anchor::{HPos, VPos};
use std::path::Path;
use ts_core::curve::Curve;

pub(crate) fn range_pad(lo: f64, hi: f64, frac: f64) -> (f64, f64) {
    let p = if hi > lo { (hi - lo) * frac } else { 1e-3 };
    (lo - p, hi + p)
}

pub(crate) fn finite_minmax(a: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in a.iter().filter(|x| x.is_finite()) {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if lo > hi { (-1.0, 1.0) } else { (lo, hi) }
}

/// Confidence strip as a filled area from 0 to the per-pixel-column mean confidence
/// (a thin 0.8 px line at 200 samples/s over minutes would turn every dropout into a dark
/// block of vertical strokes): dropouts read as empty, sure parts as full.
fn conf_strip(cv: &Canvas, a: &Ax, t: &[f64], conf: &[f64]) {
    let w = (a.r - a.l).ceil() as usize + 1;
    let xs: Vec<f64> = t.iter().map(|&ti| a.px(ti) - a.l).collect();
    let y0 = a.py(0.0);
    for (i, m) in column_means(&xs, conf, w).into_iter().enumerate() {
        if let Some(m) = m {
            let x = a.l + i as f64;
            cv.rect(x, a.py(m), x + 1.0, y0, Col(BLUE, 0.55));
        }
    }
}

/// Mean of `v` (clamped to 0..1) per pixel column `floor(x)`, `w` columns; None: no sample.
fn column_means(x: &[f64], v: &[f64], w: usize) -> Vec<Option<f64>> {
    let (mut sum, mut cnt) = (vec![0.0f64; w], vec![0u32; w]);
    for (&xi, &vi) in x.iter().zip(v) {
        if !(vi.is_finite() && xi >= 0.0) || xi as usize >= w {
            continue;
        }
        sum[xi as usize] += vi.clamp(0.0, 1.0);
        cnt[xi as usize] += 1;
    }
    sum.iter()
        .zip(&cnt)
        .map(|(&s, &n)| (n > 0).then(|| s / n as f64))
        .collect()
}

/// Draw the speed plot; returns its numbers (wow, flutter, L1).
///
/// `c1`: pass-1 curve, `c`: final curve, `cm`: model-step curve, `f_means`: mean frequency [Hz]
/// of each model component (dotted lines in the spectrum).
pub fn plot_speed(
    c1: &Curve,
    c: &Curve,
    cm: &Curve,
    f_means: &[f64],
    path: &Path,
    title: &str,
) -> Result<SpeedNumbers, PlotError> {
    let nm = speed_plot_numbers(c1, c, cm)?;
    let t = &c.t;
    let n = nm.raw.len();
    let valid = &c.valid;
    let (w, h) = (1430u32, 975u32);
    let cv = Canvas::new(path, w, h)?;
    let (l, r) = (88.0, 1410.0);

    // ---- timeline -------------------------------------------------------------------------
    let tspan = t[t.len() - 1] - t[0];
    let xr = (t[0] - 0.02 * tspan, t[t.len() - 1] + 0.02 * tspan);
    let mut all = nm.dev.clone();
    all.extend(nm.raw.iter().filter(|x| x.is_finite()));
    let (lo, hi) = pct2(&all, 0.2, 99.8);
    let yr = range_pad(lo, hi, 0.15);
    let a1 = Ax::new(l, 62.0, r, 458.0, xr, yr);
    let a2 = Ax::new(l, 466.0, r, 548.0, xr, (-0.05, 1.05));
    let xt = nice_ticks(xr.0, xr.1, 10);
    let yt = nice_ticks(yr.0, yr.1, 8);
    for a in [&a1, &a2] {
        a.fill_rect(&cv, SURF);
    }
    a1.frame(&cv, &xt, &yt);
    let ct = [(0.0, "0".to_string()), (1.0, "1".to_string())];
    a2.frame(&cv, &xt, &ct);
    for &(a, b) in &nm.invalid_spans {
        a1.span_x(&cv, a, b, SHADE);
        a2.span_x(&cv, a, b, SHADE);
    }
    for &(a, b) in &nm.gaps {
        a1.span_x(&cv, a, b, GAP);
        a2.span_x(&cv, a, b, GAP);
    }
    let l1 = nm.l1_pct;
    a1.line(&cv, &c1.t[..n], &nm.raw, Pen::solid(ORANGE, 0.6, 1));
    a1.line(&cv, t, &nm.dev, Pen::solid(BLUE, 0.95, 1));
    a1.line(&cv, t, &nm.trend, Pen::solid(BLUE_DK, 1.0, 2));
    a1.hline(&cv, l1, Pen::dashed(INK2, 1, 6.0, 4.0));
    conf_strip(&cv, &a2, t, &c.confidence);
    a1.spines_and_labels(&cv, &xt, &yt, false, 13.0);
    a2.spines_and_labels(&cv, &xt, &ct, true, 13.0);
    cv.text_up(
        "tape speed vs nominal [%]",
        20.0,
        (a1.t + a1.b) / 2.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    cv.text_up(
        "conf.",
        38.0,
        (a2.t + a2.b) / 2.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    cv.text(
        "time [s]",
        (l + r) / 2.0,
        a2.b + 30.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    let ttl = if title.is_empty() {
        "Estimated tape speed".to_string()
    } else {
        format!("Estimated tape speed \u{2014} {title}")
    };
    cv.text(&ttl, l, 10.0, 16.0, INK, HPos::Left, VPos::Top);
    let mode = meta_str(c, "abs_mode", "?");
    legend_row(
        &cv,
        &[
            LegendItem {
                label: "measured (pass 1, L1-aligned)".into(),
                pen: Pen::solid(ORANGE, 0.8, 1),
            },
            LegendItem {
                label: "correction curve".into(),
                pen: Pen::solid(BLUE, 1.0, 2),
            },
            LegendItem {
                label: "slow trend (2 s mean)".into(),
                pen: Pen::solid(BLUE_DK, 1.0, 2),
            },
            LegendItem {
                label: fmt_minus(&format!("L1 {l1:+.3} % ({mode})")),
                pen: Pen::dashed(INK2, 1, 6.0, 4.0),
            },
        ],
        r,
        46.0,
        12.5,
    );

    // ---- detail ---------------------------------------------------------------------------
    let (t0, wd) = (nm.detail_t0, nm.detail_w);
    let di: Vec<usize> = (0..t.len())
        .filter(|&i| t[i] >= t0 && t[i] <= t0 + wd)
        .collect();
    let ri: Vec<usize> = (0..n)
        .filter(|&i| c1.t[i] >= t0 && c1.t[i] <= t0 + wd)
        .collect();
    let td: Vec<f64> = di.iter().map(|&i| t[i]).collect();
    let dd: Vec<f64> = di.iter().map(|&i| nm.dev[i]).collect();
    let vd: Vec<bool> = di.iter().map(|&i| valid[i]).collect();
    let tr_: Vec<f64> = ri.iter().map(|&i| c1.t[i]).collect();
    let rr: Vec<f64> = ri.iter().map(|&i| nm.raw[i]).collect();
    let mut ally = dd.clone();
    ally.extend(rr.iter().filter(|x| x.is_finite()));
    let (ylo, yhi) = finite_minmax(&ally);
    let yr3 = range_pad(ylo, yhi, 0.07);
    let xr3 = (t0 - 0.02 * wd, t0 + wd * 1.02);
    let a3 = Ax::new(l, 648.0, 640.0, 870.0, xr3, yr3);
    a3.fill_rect(&cv, SURF);
    let (xt3, yt3) = (nice_ticks(xr3.0, xr3.1, 6), nice_ticks(yr3.0, yr3.1, 6));
    a3.frame(&cv, &xt3, &yt3);
    let inv3: Vec<bool> = vd.iter().map(|b| !b).collect();
    for (a, b) in crate::numbers::spans(&td, &inv3) {
        a3.span_x(&cv, a, b, SHADE);
    }
    a3.line(&cv, &tr_, &rr, Pen::solid(ORANGE, 0.8, 1));
    a3.line(&cv, &td, &dd, Pen::solid(BLUE, 1.0, 2));
    a3.spines_and_labels(&cv, &xt3, &yt3, true, 13.0);
    cv.text(
        &format!("Detail {:.1}\u{2013}{:.1} s", t0, t0 + wd),
        l,
        a3.t - 10.0,
        15.0,
        INK,
        HPos::Left,
        VPos::Bottom,
    );
    cv.text(
        "time [s]",
        (a3.l + a3.r) / 2.0,
        a3.b + 30.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    cv.text_up(
        "tape speed vs nominal [%]",
        20.0,
        (a3.t + a3.b) / 2.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    legend_row(
        &cv,
        &[
            LegendItem {
                label: "measured".into(),
                pen: Pen::solid(ORANGE, 0.8, 1),
            },
            LegendItem {
                label: "correction curve".into(),
                pen: Pen::solid(BLUE, 1.0, 2),
            },
        ],
        a3.r,
        a3.t - 20.0,
        12.5,
    );

    // ---- spectrum -------------------------------------------------------------------------
    let sel: Vec<usize> = (0..nm.f.len())
        .filter(|&i| nm.f[i] > 0.05 && nm.f[i] <= 50.0)
        .collect();
    let sf: Vec<f64> = sel.iter().map(|&i| nm.f[i]).collect();
    let sp: Vec<f64> = sel.iter().map(|&i| nm.p[i].sqrt()).collect();
    let pos: Vec<f64> = sp.iter().cloned().filter(|&x| x > 0.0).collect();
    let (pmin, pmax) = if pos.is_empty() {
        (1e-6, 1.0)
    } else {
        finite_minmax(&pos)
    };
    let ytop = if sel.is_empty() { 1.0 } else { pmax * 1.6 };
    let (fmin, _) = if sf.is_empty() {
        (0.05, 50.0)
    } else {
        finite_minmax(&sf)
    };
    let mut a4 = Ax::new(
        770.0,
        648.0,
        r,
        870.0,
        (fmin * 0.85, 60.0),
        ((pmin * 0.6).max(pmax * 1e-9), ytop * 1.25),
    );
    a4.xlog = true;
    a4.ylog = true;
    a4.fill_rect(&cv, SURF);
    let (xt4, yt4) = (log_ticks(a4.xr.0, a4.xr.1), log_ticks(a4.yr.0, a4.yr.1));
    // shaded wow/flutter band under the grid
    a4.span_x(&cv, 0.5, 6.0, SHADE);
    a4.frame(&cv, &xt4, &yt4);
    let sp_plot: Vec<f64> = sp
        .iter()
        .map(|&x| if x > 0.0 { x } else { f64::NAN })
        .collect();
    a4.line(&cv, &sf, &sp_plot, Pen::solid(BLUE, 1.0, 2));
    let mut fcs: Vec<f64> = f_means.to_vec();
    fcs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut last = 0.0;
    for fc in fcs {
        if fc > 0.05 && fc <= 50.0 {
            a4.vline(&cv, fc, Pen::dashed(INK2, 1, 2.0, 3.0));
            if fc > last * 1.12 {
                cv.text_up(
                    &format!("{fc:.2}"),
                    a4.px(fc) - 3.0,
                    a4.py(ytop) + 2.0,
                    11.0,
                    INK2,
                    HPos::Right,
                    VPos::Top,
                );
                last = fc;
            }
        }
    }
    let fl = meta_str(c, "corr_max_hz", "none");
    let fl_v: Option<f64> = if matches!(fl.as_str(), "none" | "") {
        None
    } else {
        fl.parse().ok()
    };
    if let Some(fv) = fl_v {
        a4.vline(&cv, fv, Pen::dashed(ORANGE, 2, 7.0, 4.0));
        let txt = format!("band limit {} Hz", fmt_g(fv));
        let x = a4.px(fv);
        if x + 10.0 + cv.text_w(&txt, 11.0) as f64 > a4.r {
            cv.text(&txt, x - 5.0, a4.t + 6.0, 11.0, INK, HPos::Right, VPos::Top);
        } else {
            cv.text(&txt, x + 5.0, a4.t + 6.0, 11.0, INK, HPos::Left, VPos::Top);
        }
    }
    a4.spines_and_labels(&cv, &xt4, &yt4, true, 13.0);
    cv.text(
        "wow",
        a4.px(0.55),
        a4.b - 6.0,
        11.0,
        INK2,
        HPos::Left,
        VPos::Bottom,
    );
    cv.text(
        "flutter",
        a4.px(6.3),
        a4.b - 6.0,
        11.0,
        INK2,
        HPos::Left,
        VPos::Bottom,
    );
    cv.text(
        &format!(
            "Speed variation spectrum \u{2014} wow {:.3} %, flutter {:.3} % rms",
            nm.wow_pct, nm.flutter_pct
        ),
        a4.l,
        a4.t - 10.0,
        15.0,
        INK,
        HPos::Left,
        VPos::Bottom,
    );
    cv.text(
        "modulation frequency [Hz]  (dotted: model components)",
        (a4.l + a4.r) / 2.0,
        a4.b + 30.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    cv.text_up(
        "% rms / \u{221a}Hz",
        722.0,
        (a4.t + a4.b) / 2.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );

    // ---- footer ---------------------------------------------------------------------------
    let trusted = valid.iter().filter(|&&b| b).count() as f64 / valid.len() as f64 * 100.0;
    let band = match fl_v {
        None => "full".to_string(),
        Some(_) => format!("\u{2264} {fl} Hz"),
    };
    let mut info = format!(
        "ref {} Hz \u{2192} true {:.1} Hz   path {} (EV {})   trusted {:.1} %   band {} ({})   tape speed given {}, inferred {}",
        fmt_g(meta_f64(c, "ref_hz").unwrap_or(0.0)),
        meta_f64(c, "abs_ref_true_hz").unwrap_or(0.0),
        meta_str(c, "model_path", "?"),
        meta_str(c, "model_ev", "?"),
        trusted,
        band,
        meta_str(c, "corr_policy", ""),
        meta_str(c, "tape_speed_given", "none"),
        meta_str(c, "tape_speed_inferred", "none"),
    );
    if c.meta.contains_key("ref_coverage_pct") {
        info += &format!(
            "   reference coverage {} %",
            meta_str(c, "ref_coverage_pct", "")
        );
    }
    cv.text(&info, 10.0, 930.0, 12.0, INK2, HPos::Left, VPos::Top);
    let l2 = format!(
        "shading: light grey = no trusted measurement{}",
        if nm.gaps.is_empty() {
            ""
        } else {
            ", darker grey = no reference (slow fallback only)"
        }
    );
    cv.text(&l2, 10.0, 948.0, 12.0, INK2, HPos::Left, VPos::Top);
    cv.finish()?;
    Ok(nm)
}

#[cfg(test)]
mod tests {
    #[test]
    fn conf_columns_average_dropouts() {
        // 10 samples per column: column 0 sure, column 1 dropping out every other sample,
        // column 2 lost, column 3 empty
        let x: Vec<f64> = (0..30).map(|i| i as f64 / 10.0).collect();
        let v: Vec<f64> = (0..30)
            .map(|i| match i / 10 {
                0 => 1.0,
                1 => (i % 2) as f64,
                _ => 0.0,
            })
            .collect();
        assert_eq!(
            super::column_means(&x, &v, 4),
            vec![Some(1.0), Some(0.5), Some(0.0), None]
        );
    }
}
