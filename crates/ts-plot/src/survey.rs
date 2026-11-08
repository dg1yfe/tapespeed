//! Survey plot: spectrum overview, line-prominence map,
//! presence map. The input struct is ts-plot's own; the pipeline maps the real survey onto it.
use crate::PlotError;
use crate::draw::*;
use crate::speed::finite_minmax;
use plotters::style::RGBColor;
use plotters::style::text_anchor::{HPos, VPos};
use std::collections::BTreeMap;
use std::path::Path;
use ts_dsp::stats::{interp_one, median, percentiles};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurveyClass {
    FmPilot,
    AmBeat,
    TvLine,
    Unknown,
    Codec,
    Hum,
}

impl SurveyClass {
    fn short(self) -> &'static str {
        match self {
            Self::FmPilot => "FM",
            Self::AmBeat => "AM",
            Self::TvLine => "TV",
            Self::Hum => "hum",
            Self::Unknown => "?",
            Self::Codec => "codec",
        }
    }
    fn style(self) -> (RGBColor, Marker) {
        match self {
            Self::FmPilot | Self::AmBeat | Self::TvLine => (BLUE, Marker::TriDown),
            Self::Unknown => (INK2, Marker::Circle),
            Self::Codec => (ORANGE, Marker::Cross),
            Self::Hum => (AQUA, Marker::Square),
        }
    }
}

/// Detection of a candidate in one block.
#[derive(Debug, Clone, Copy)]
pub struct SurveyDet {
    /// detected frequency [Hz]
    pub f: f64,
    /// C/N0 [dB-Hz] on the best channel
    pub cn0: f64,
}

#[derive(Debug, Clone)]
pub struct SurveyCandidate {
    pub id: u32,
    pub f: f64,
    pub cls: SurveyClass,
    pub low: bool,
    pub primary: bool,
    /// channel label ("mid", "mono", ...)
    pub ch: String,
    /// block index -> detection
    pub blocks: BTreeMap<usize, SurveyDet>,
}

#[derive(Debug, Clone)]
pub struct SurveyPlotInput {
    /// input file name (title)
    pub file: String,
    /// log-spaced frequency grid [Hz]
    pub f: Vec<f64>,
    /// median density over blocks (`D`), on `f`
    pub d: Vec<f64>,
    /// density per block (`D_blocks`), blocks x f
    pub d_blocks: Vec<Vec<f64>>,
    /// line prominence [dB] per block (`prom`), blocks x f
    pub prom: Vec<Vec<f64>>,
    /// (start s, duration s) per block
    pub blocks: Vec<(f64, f64)>,
    /// ranked candidates, in ranking order
    pub candidates: Vec<SurveyCandidate>,
    /// log-frequency grid step (survey `STEP`)
    pub step: f64,
    /// `LOCK_DB`, `MARGIN_DB`
    pub lock_db: f64,
    pub margin_db: f64,
}

fn cmap(u: f64) -> RGBColor {
    let stops = [
        (0xfc, 0xfc, 0xfb),
        (0xcf, 0xe0, 0xf5),
        (0x2a, 0x78, 0xd6),
        (0x12, 0x3a, 0x6b),
    ];
    let u = u.clamp(0.0, 1.0) * 3.0;
    let i = (u.floor() as usize).min(2);
    let f = u - i as f64;
    let m = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * f).round() as u8;
    RGBColor(
        m(stops[i].0, stops[i + 1].0),
        m(stops[i].1, stops[i + 1].1),
        m(stops[i].2, stops[i + 1].2),
    )
}

fn khz(v: f64) -> String {
    format!("{}k", fmt_g(v / 1000.0))
}

pub fn plot_survey(s: &SurveyPlotInput, path: &Path) -> Result<(), PlotError> {
    let nb = s.blocks.len();
    let f = &s.f;
    if nb == 0 || f.len() < 2 || s.d_blocks.len() != nb || s.prom.len() != nb {
        return Err(PlotError::Input(
            "plot_survey: empty or inconsistent survey".into(),
        ));
    }
    let nf = f.len();
    let refd = median(
        &f.iter()
            .zip(&s.d)
            .filter(|(f, _)| **f > 1000.0 && **f < 2000.0)
            .map(|(_, d)| *d)
            .collect::<Vec<_>>(),
    ) + 1e-30;
    let db: Vec<Vec<f64>> = s
        .d_blocks
        .iter()
        .map(|r| {
            r.iter()
                .map(|d| 10.0 * (d / refd + 1e-30).log10())
                .collect()
        })
        .collect();
    let (mut lo_, mut med, mut hi_) = (vec![0.0; nf], vec![0.0; nf], vec![0.0; nf]);
    for j in 0..nf {
        let col: Vec<f64> = db.iter().map(|r| r[j]).collect();
        let q = percentiles(&col, &[10.0, 50.0, 90.0])?;
        (lo_[j], med[j], hi_[j]) = (q[0], q[1], q[2]);
    }
    let top: Vec<&SurveyCandidate> = s
        .candidates
        .iter()
        .filter(|c| !c.low && c.cls != SurveyClass::Codec)
        .take(8)
        .collect();
    let lab_c: Vec<&SurveyCandidate> = s.candidates.iter().filter(|c| !c.low).collect();
    let cod: Vec<&&SurveyCandidate> = lab_c
        .iter()
        .filter(|c| c.cls == SurveyClass::Codec)
        .collect();
    let mut named: Vec<&SurveyCandidate> = lab_c
        .iter()
        .filter(|c| c.cls != SurveyClass::Codec)
        .take(6)
        .cloned()
        .collect();
    if let Some(c) = cod.first() {
        named.push(**c);
    }
    named.sort_by(|a, b| a.f.partial_cmp(&b.f).unwrap());

    let rows = top.len().max(1);
    let (l, r) = (200.0, 1370.0);
    let (a1t, a1b) = (50.0, 390.0);
    let (a2t, a2b) = (a1b + 85.0, a1b + 85.0 + 265.0);
    let (a3t, a3b) = (a2b + 110.0, a2b + 110.0 + 44.0 * rows as f64);
    let (w, h) = (1430u32, (a3b + 70.0) as u32);
    let cv = Canvas::new(path, w, h)?;
    let fmax = f[nf - 1];
    let xt: Vec<(f64, String)> = [1000.0, 2000.0, 5000.0, 9000.0, 15625.0, 19000.0]
        .iter()
        .filter(|&&v| v < fmax)
        .map(|&v| (v, khz(v)))
        .collect();

    // ---- a1 spectrum ----------------------------------------------------------------------
    let hi1k: Vec<f64> = f
        .iter()
        .zip(&hi_)
        .filter(|(f, _)| **f > 1000.0)
        .map(|(_, v)| *v)
        .collect();
    let lo1k: Vec<f64> = f
        .iter()
        .zip(&lo_)
        .filter(|(f, _)| **f > 1000.0)
        .map(|(_, v)| *v)
        .collect();
    let ymax = finite_minmax(&hi1k).1 + 6.0;
    let ymin = percentiles(&lo1k, &[1.0]).map(|v| v[0]).unwrap_or(-100.0) - 5.0;
    let mut a1 = Ax::new(l, a1t, r, a1b, (1000.0, fmax), (ymin, ymax + 14.0));
    a1.xlog = true;
    a1.fill_rect(&cv, SURF);
    let yt1 = nice_ticks(a1.yr.0, a1.yr.1, 7);
    a1.frame(&cv, &xt, &yt1);
    // p10-p90 band as vertical strips per frequency
    for j in 0..nf {
        let x0 = a1.px(f[j].max(1000.0));
        if f[j] < 1000.0 || !(lo_[j].is_finite() && hi_[j].is_finite()) {
            continue;
        }
        let (ya, yb) = (a1.py(hi_[j]).max(a1.t), a1.py(lo_[j]).min(a1.b));
        if yb > ya {
            let xn = if j + 1 < nf {
                a1.px(f[j + 1])
            } else {
                x0 + 1.0
            };
            cv.rect(
                x0,
                ya,
                xn.max(x0 + 1.0).min(a1.r),
                yb,
                RGBColor(0xe4, 0xe3, 0xdf),
            );
        }
    }
    a1.line(&cv, f, &med, Pen::solid(INK2, 1.0, 1));
    for c in &lab_c {
        let (col, mk) = c.cls.style();
        let (x, y) = a1.pt(c.f, interp_one(c.f, f, &hi_) + 2.0);
        cv.marker(mk, x, y, 6.0, col, !matches!(mk, Marker::Cross), 2);
    }
    for (k, c) in named.iter().enumerate() {
        let txt = if c.cls != SurveyClass::Codec {
            format!("#{} {} {:.0}", c.id, c.cls.short(), c.f)
        } else {
            format!("codec x{}", cod.len())
        };
        let (x, y) = a1.pt(c.f, interp_one(c.f, f, &hi_) + 2.0);
        let off = (12 + 12 * (k % 4)) as f64 * 1.53;
        cv.path(&[(x, y - 8.0), (x, y - off)], c.cls.style().0, 1);
        cv.text(
            &txt,
            x,
            y - off - 2.0,
            12.0,
            INK,
            HPos::Center,
            VPos::Bottom,
        );
    }
    a1.spines_and_labels(&cv, &xt, &yt1, true, 13.0);
    cv.text_up(
        "density rel. 1-2 kHz [dB]",
        l - 66.0,
        (a1.t + a1.b) / 2.0,
        14.0,
        INK,
        HPos::Center,
        VPos::Top,
    );
    let hum: Vec<&SurveyCandidate> = s
        .candidates
        .iter()
        .filter(|c| c.cls == SurveyClass::Hum)
        .take(5)
        .collect();
    let mut t1 = format!(
        "{}: spectrum of {} blocks x {:.0} s",
        s.file, nb, s.blocks[0].1
    );
    if !hum.is_empty() {
        t1 += &format!(
            "   hum: {}",
            hum.iter()
                .map(|c| format!("{:.2} Hz", c.f))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    cv.text_bold(&t1, l, a1.t - 28.0, 15.0, INK);
    // legend (lower left)
    let mut ly = a1.b - 20.0 * 5.0 - 2.0;
    let lx = a1.l + 12.0;
    let mut entry = |draw: &dyn Fn(f64, f64), label: &str| {
        draw(lx + 12.0, ly + 8.0);
        cv.text(
            label,
            lx + 34.0,
            ly + 8.0,
            12.0,
            INK,
            HPos::Left,
            VPos::Center,
        );
        ly += 19.0;
    };
    entry(
        &|x, y| {
            cv.rect(
                x - 12.0,
                y - 4.0,
                x + 12.0,
                y + 4.0,
                RGBColor(0xe4, 0xe3, 0xdf),
            )
        },
        "p10-p90 over blocks",
    );
    entry(
        &|x, y| cv.path(&[(x - 12.0, y), (x + 12.0, y)], INK2, 1),
        "median density (mid/mono)",
    );
    entry(
        &|x, y| cv.marker(Marker::TriDown, x, y, 5.0, BLUE, true, 2),
        "broadcast ref (FM/AM/TV)",
    );
    entry(
        &|x, y| cv.marker(Marker::Circle, x, y, 5.0, INK2, true, 2),
        "unknown stable line",
    );
    entry(
        &|x, y| cv.marker(Marker::Cross, x, y, 4.0, ORANGE, false, 2),
        "codec artifact",
    );

    // ---- a2 prominence map ----------------------------------------------------------------
    let tc: Vec<f64> = s.blocks.iter().map(|(t0, d)| t0 + d / 2.0).collect();
    let mut ext = vec![tc[0] - s.blocks[0].1 / 2.0];
    for i in 0..nb - 1 {
        ext.push((tc[i] + tc[i + 1]) / 2.0);
    }
    ext.push(tc[nb - 1] + s.blocks[nb - 1].1 / 2.0);
    let mut fe = vec![f[0] * (-s.step / 2.0).exp()];
    for j in 0..nf - 1 {
        fe.push((f[j] * f[j + 1]).sqrt());
    }
    fe.push(f[nf - 1] * (s.step / 2.0).exp());
    let mut a2 = Ax::new(l, a2t, r, a2b, (1000.0, fmax), (ext[0], ext[nb]));
    a2.xlog = true;
    a2.yflip = true;
    // heat: per pixel column
    let (px0, px1) = (a2.l.round() as i32, a2.r.round() as i32);
    for px in px0..px1 {
        let u = (px as f64 + 0.5 - a2.l) / (a2.r - a2.l);
        let fv = (a2.xr.0.ln() + u * (a2.xr.1.ln() - a2.xr.0.ln())).exp();
        let j = fe
            .partition_point(|&e| e <= fv)
            .saturating_sub(1)
            .min(nf - 1);
        for b in 0..nb {
            let v = s.prom[b][j];
            let col = cmap(if v.is_finite() {
                v.clamp(0.0, 25.0) / 25.0
            } else {
                0.0
            });
            cv.rect(
                px as f64,
                a2.py(ext[b]),
                px as f64 + 1.0,
                a2.py(ext[b + 1]) + 0.5,
                col,
            );
        }
    }
    for c in &top {
        let col = if c.primary { ORANGE } else { INK };
        for (&b, d) in &c.blocks {
            if b < nb {
                let (x, y) = a2.pt(d.f, tc[b]);
                cv.marker(c.cls.style().1, x, y, 4.0, col, false, 1);
            }
        }
    }
    let yt2 = nice_ticks(ext[0], ext[nb], 6);
    a2.spines_and_labels(&cv, &xt, &yt2, true, 13.0);
    cv.text_up(
        "block centre [s]",
        l - 66.0,
        (a2.t + a2.b) / 2.0,
        14.0,
        INK,
        HPos::Center,
        VPos::Top,
    );
    cv.text(
        "frequency [Hz] (log)",
        (a2.l + a2.r) / 2.0,
        a2.b + 30.0,
        14.0,
        INK,
        HPos::Center,
        VPos::Top,
    );
    cv.text_bold(
        "line prominence over the local median [dB] per block (mid/mono); markers: top candidates (orange = primary), detected block frequency",
        l,
        a2.t - 28.0,
        15.0,
        INK,
    );
    // colour bar
    let (cx0, cx1) = (a2.r + 8.0, a2.r + 22.0);
    for py in a2.t.round() as i32..a2.b.round() as i32 {
        let u = 1.0 - (py as f64 - a2.t) / (a2.b - a2.t);
        cv.rect(cx0, py as f64, cx1, py as f64 + 1.5, cmap(u));
    }
    for v in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0] {
        let y = a2.b - v / 25.0 * (a2.b - a2.t);
        cv.path(&[(cx1, y), (cx1 + 4.0, y)], INK2, 1);
        cv.text(
            &format!("{v:.0}"),
            cx1 + 7.0,
            y,
            12.0,
            INK2,
            HPos::Left,
            VPos::Center,
        );
    }
    cv.text_up(
        "dB",
        cx1 + 40.0,
        (a2.t + a2.b) / 2.0,
        13.0,
        INK2,
        HPos::Center,
        VPos::Top,
    );

    // ---- a3 presence ----------------------------------------------------------------------
    let (cw, ch) = ((r - l) / nb as f64, (a3b - a3t) / rows as f64);
    for (i, c) in top.iter().enumerate() {
        for b in 0..nb {
            let (x0, y0) = (l + b as f64 * cw, a3t + i as f64 * ch);
            match c.blocks.get(&b) {
                Some(d) if d.cn0.is_finite() => {
                    cv.rect(x0, y0, x0 + cw + 0.5, y0 + ch + 0.5, cmap(d.cn0 / 50.0));
                    let col = if d.cn0 < 30.0 { INK2 } else { SURF };
                    cv.text(
                        &format!("{:.0}", d.cn0),
                        x0 + cw / 2.0,
                        y0 + ch / 2.0,
                        12.0,
                        col,
                        HPos::Center,
                        VPos::Center,
                    );
                }
                _ => cv.text(
                    ".",
                    x0 + cw / 2.0,
                    y0 + ch / 2.0,
                    12.0,
                    INK2,
                    HPos::Center,
                    VPos::Center,
                ),
            }
        }
        cv.text(
            &format!("#{} {} {:.0} ({})", c.id, c.cls.short(), c.f, c.ch),
            l - 10.0,
            a3t + (i as f64 + 0.5) * ch,
            13.0,
            INK2,
            HPos::Right,
            VPos::Center,
        );
    }
    if top.is_empty() {
        cv.text(
            "none",
            l - 10.0,
            a3t + ch / 2.0,
            13.0,
            INK2,
            HPos::Right,
            VPos::Center,
        );
    }
    for b in 0..nb {
        let x = l + (b as f64 + 0.5) * cw;
        let lab = format!("{:.0}", tc[b]);
        if nb <= 14 {
            cv.text(&lab, x, a3b + 8.0, 13.0, INK2, HPos::Center, VPos::Top);
        } else {
            cv.text_up(&lab, x, a3b + 8.0, 13.0, INK2, HPos::Center, VPos::Top);
        }
    }
    let lab_h = if nb <= 14 { 28.0 } else { 52.0 };
    cv.text(
        "block centre [s]",
        (l + r) / 2.0,
        a3b + 8.0 + lab_h,
        14.0,
        INK,
        HPos::Center,
        VPos::Top,
    );
    cv.text_bold(
        &format!(
            "presence: C/N0 [dB-Hz] per block, best channel; . = not detected (lock < {}, marginal < {})",
            fmt_g(s.lock_db),
            fmt_g(s.margin_db)
        ),
        l,
        a3t - 28.0,
        15.0,
        INK,
    );
    cv.finish()
}
