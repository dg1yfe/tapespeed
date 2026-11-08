//! Model-step plot: pass 1 vs refined curve, model part,
//! component frequency laws.
use crate::PlotError;
use crate::draw::*;
use crate::numbers::{meta_str, spans};
use crate::speed::{finite_minmax, range_pad};
use plotters::style::text_anchor::{HPos, VPos};
use std::path::Path;
use ts_core::curve::Curve;

/// One model component's frequency law evaluated on every sample of `c_in.t`.
pub struct CompTrack {
    /// chirp components are drawn orange, the rest blue
    pub chirp: bool,
    pub harmonics: Vec<f64>,
    /// `c.freq(t)` for each sample of `c_in.t` (only every 100th is drawn)
    pub freq: Vec<f64>,
}

/// The fitted model on the grid of `c_in.t`, without depending on the model's types.
pub struct ModelFit {
    /// `m.predict(t)` (ratio)
    pub predict: Vec<f64>,
    /// `m.r0`
    pub r0: f64,
    pub comps: Vec<CompTrack>,
}

fn panel(cv: &Canvas, a: &Ax, xt: &[(f64, String)], yt: &[(f64, String)]) {
    a.fill_rect(cv, SURF);
    a.frame(cv, xt, yt);
}

pub fn plot_model(
    c_in: &Curve,
    c_out: &Curve,
    fit: Option<&ModelFit>,
    path: &Path,
) -> Result<(), PlotError> {
    let t = &c_in.t;
    if t.len() < 2 || c_out.ratio.len() != t.len() || c_out.valid.len() != t.len() {
        return Err(PlotError::Input(
            "plot_model: curves too short or of different length".into(),
        ));
    }
    let cv = Canvas::new(path, 1320, 880)?;
    let (l, r) = (88.0, 1300.0);
    let span = t[t.len() - 1] - t[0];
    let xr = (t[0] - 0.02 * span, t[t.len() - 1] + 0.02 * span);
    let xt = nice_ticks(xr.0, xr.1, 10);
    let d_in: Vec<f64> = c_in.ratio.iter().map(|x| (x - 1.0) * 100.0).collect();
    let d_out: Vec<f64> = c_out.ratio.iter().map(|x| (x - 1.0) * 100.0).collect();
    let rows = [(50.0, 360.0), (374.0, 580.0), (594.0, 800.0)];
    let mut ax: Vec<Ax> = vec![];

    // panel 0
    let mut all = d_in.clone();
    all.extend(&d_out);
    let (lo, hi) = finite_minmax(&all);
    let y0 = range_pad(lo, hi, 0.05);
    let a0 = Ax::new(l, rows[0].0, r, rows[0].1, xr, y0);
    let yt0 = nice_ticks(y0.0, y0.1, 7);
    panel(&cv, &a0, &xt, &yt0);
    let inv: Vec<bool> = c_out.valid.iter().map(|b| !b).collect();
    for (a, b) in spans(t, &inv) {
        a0.span_x(&cv, a, b, SHADE);
    }
    a0.line(&cv, t, &d_in, Pen::solid(GREY_PASS1, 1.0, 1));
    a0.line(&cv, t, &d_out, Pen::solid(BLUE, 1.0, 1));
    a0.spines_and_labels(&cv, &xt, &yt0, false, 13.0);
    cv.text(
        &format!(
            "model step: EV {}, grey = invalid",
            meta_str(c_out, "model_ev", "None")
        ),
        l,
        a0.t - 10.0,
        15.0,
        INK,
        HPos::Left,
        VPos::Bottom,
    );
    cv.text_up(
        "speed dev. [%]",
        22.0,
        (a0.t + a0.b) / 2.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    legend_col(
        &cv,
        &[
            LegendItem {
                label: "pass 1 (analyze)".into(),
                pen: Pen::solid(GREY_PASS1, 1.0, 2),
            },
            LegendItem {
                label: format!("refined ({})", meta_str(c_out, "model_path", "None")),
                pen: Pen::solid(BLUE, 1.0, 2),
            },
        ],
        a0.r - 8.0,
        a0.t + 6.0,
        12.5,
    );
    ax.push(a0);

    // panel 1
    let mut mdl: Vec<f64> = vec![];
    let mut res: Vec<f64> = vec![];
    if let Some(m) = fit.filter(|m| m.predict.len() == t.len()) {
        mdl = m.predict.iter().map(|p| (p - m.r0) * 100.0).collect();
        res = c_out
            .ratio
            .iter()
            .zip(&m.predict)
            .map(|(o, p)| (o - p) * 100.0)
            .collect();
    }
    let mut all1 = mdl.clone();
    all1.extend(&res);
    let y1 = if all1.is_empty() {
        (-1.0, 1.0)
    } else {
        range_pad(finite_minmax(&all1).0, finite_minmax(&all1).1, 0.05)
    };
    let a1 = Ax::new(l, rows[1].0, r, rows[1].1, xr, y1);
    let yt1 = nice_ticks(y1.0, y1.1, 5);
    panel(&cv, &a1, &xt, &yt1);
    if fit.is_some() {
        a1.line(&cv, t, &mdl, Pen::solid(ORANGE, 1.0, 1));
        a1.line(&cv, t, &res, Pen::solid(INK2, 0.7, 1));
        legend_col(
            &cv,
            &[
                LegendItem {
                    label: "model L3 - r0".into(),
                    pen: Pen::solid(ORANGE, 1.0, 2),
                },
                LegendItem {
                    label: "residual".into(),
                    pen: Pen::solid(INK2, 0.7, 2),
                },
            ],
            a1.r - 8.0,
            a1.t + 6.0,
            12.5,
        );
    }
    a1.spines_and_labels(&cv, &xt, &yt1, false, 13.0);
    cv.text_up(
        "[%]",
        22.0,
        (a1.t + a1.b) / 2.0,
        14.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );
    ax.push(a1);

    // panel 2: component laws
    let mut tracks: Vec<(bool, f64, Vec<f64>, Vec<f64>)> = vec![];
    let mut allf: Vec<f64> = vec![];
    if let Some(m) = fit {
        for cp in &m.comps {
            if cp.freq.len() != t.len() {
                continue;
            }
            let ts: Vec<f64> = t.iter().step_by(100).cloned().collect();
            let fs: Vec<f64> = cp.freq.iter().step_by(100).cloned().collect();
            for &h in &cp.harmonics {
                let ys: Vec<f64> = fs.iter().map(|f| h * f).collect();
                allf.extend(&ys);
                tracks.push((cp.chirp, h, ts.clone(), ys));
            }
        }
    }
    let y2 = if allf.is_empty() {
        (0.0, 1.0)
    } else {
        range_pad(finite_minmax(&allf).0, finite_minmax(&allf).1, 0.05)
    };
    let a2 = Ax::new(l, rows[2].0, r, rows[2].1, xr, y2);
    let yt2 = nice_ticks(y2.0, y2.1, 5);
    panel(&cv, &a2, &xt, &yt2);
    for (chirp, h, ts, ys) in &tracks {
        a2.line(
            &cv,
            ts,
            ys,
            Pen::solid(
                if *chirp { ORANGE } else { BLUE },
                1.0,
                if *h == 1.0 { 2 } else { 1 },
            ),
        );
    }
    a2.spines_and_labels(&cv, &xt, &yt2, true, 13.0);
    cv.text_up(
        "component f [Hz]",
        22.0,
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
    cv.finish()
}
