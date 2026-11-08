//! A tiny axes layer on top of plotters' bitmap backend: pixel-exact layout, linear/log axes,
//! clipping, dashed lines, rotated text. Everything is drawn in float pixel space.
use crate::PlotError;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use plotters::style::{FontTransform, register_font};
use std::sync::Once;

pub const BLUE: RGBColor = RGBColor(0x2a, 0x78, 0xd6);
pub const BLUE_DK: RGBColor = RGBColor(0x18, 0x4f, 0x95);
pub const ORANGE: RGBColor = RGBColor(0xeb, 0x68, 0x34);
pub const AQUA: RGBColor = RGBColor(0x1b, 0xaf, 0x7a);
pub const INK: RGBColor = RGBColor(0x0b, 0x0b, 0x0b);
pub const INK2: RGBColor = RGBColor(0x52, 0x51, 0x4e);
pub const GRID: RGBColor = RGBColor(0xe6, 0xe5, 0xe1);
pub const SHADE: RGBColor = RGBColor(0xec, 0xeb, 0xe7);
pub const GAP: RGBColor = RGBColor(0xd6, 0xd5, 0xcf);
pub const SURF: RGBColor = RGBColor(0xfc, 0xfc, 0xfb);
pub const GREY_PASS1: RGBColor = RGBColor(0x9a, 0x9a, 0x96);

static FONT: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
static FONT_BOLD: &[u8] = include_bytes!("../assets/DejaVuSans-Bold.ttf");
/// Global text scale (plotters' px font size reads small at 110 dpi).
const TEXT_SCALE: f64 = 1.1;
static INIT: Once = Once::new();

fn init_font() {
    INIT.call_once(|| {
        register_font("sans", FontStyle::Normal, FONT)
            .ok()
            .expect("embedded font");
        register_font("sans", FontStyle::Bold, FONT_BOLD)
            .ok()
            .expect("embedded font");
    });
}

pub type Area<'a> = DrawingArea<BitMapBackend<'a>, plotters::coord::Shift>;

pub fn derr<E: std::fmt::Debug>(e: E) -> PlotError {
    PlotError::Draw(format!("{e:?}"))
}

#[derive(Clone, Copy)]
pub struct Col(pub RGBColor, pub f64);
impl Col {
    fn rgba(&self) -> RGBAColor {
        self.0.mix(self.1)
    }
}
impl From<RGBColor> for Col {
    fn from(c: RGBColor) -> Self {
        Col(c, 1.0)
    }
}

/// Canvas: the drawing area plus text helpers.
pub struct Canvas<'a> {
    pub area: Area<'a>,
}

impl<'a> Canvas<'a> {
    pub fn new(path: &'a std::path::Path, w: u32, h: u32) -> Result<Self, PlotError> {
        init_font();
        let area = BitMapBackend::new(path, (w, h)).into_drawing_area();
        area.fill(&SURF).map_err(derr)?;
        Ok(Canvas { area })
    }

    pub fn finish(self) -> Result<(), PlotError> {
        self.area.present().map_err(derr)
    }

    fn style(
        size: f64,
        c: impl Into<Col>,
        hp: HPos,
        vp: VPos,
        rot: bool,
        bold: bool,
    ) -> TextStyle<'static> {
        let c: Col = c.into();
        let fs = if bold {
            FontStyle::Bold
        } else {
            FontStyle::Normal
        };
        let mut font = FontDesc::new(FontFamily::Name("sans"), size * TEXT_SCALE, fs);
        if rot {
            font = font.transform(FontTransform::Rotate270);
        }
        TextStyle {
            font,
            color: c.rgba().to_backend_color(),
            pos: Pos::new(hp, vp),
        }
    }

    /// Width of `s` at `size` px.
    pub fn text_w(&self, s: &str, size: f64) -> i32 {
        let st = Self::style(size, INK, HPos::Left, VPos::Top, false, false);
        self.area
            .estimate_text_size(s, &st)
            .map(|(w, _)| w as i32)
            .unwrap_or(0)
    }

    pub fn text(&self, s: &str, x: f64, y: f64, size: f64, c: impl Into<Col>, hp: HPos, vp: VPos) {
        let st = Self::style(size, c, hp, vp, false, false);
        let _ = self
            .area
            .draw_text(s, &st, (x.round() as i32, y.round() as i32));
    }

    pub fn text_bold(&self, s: &str, x: f64, y: f64, size: f64, c: impl Into<Col>) {
        let st = Self::style(size, c, HPos::Left, VPos::Top, false, true);
        let _ = self
            .area
            .draw_text(s, &st, (x.round() as i32, y.round() as i32));
    }

    /// Text reading bottom-to-top, anchored at (x, y) with the given alignment of the rotated box.
    pub fn text_up(
        &self,
        s: &str,
        x: f64,
        y: f64,
        size: f64,
        c: impl Into<Col>,
        hp: HPos,
        vp: VPos,
    ) {
        let st = Self::style(size, c, hp, vp, true, false);
        let _ = self
            .area
            .draw_text(s, &st, (x.round() as i32, y.round() as i32));
    }

    pub fn rect(&self, x0: f64, y0: f64, x1: f64, y1: f64, c: impl Into<Col>) {
        let c: Col = c.into();
        let (a, b) = (
            (x0.min(x1).round() as i32, y0.min(y1).round() as i32),
            (x0.max(x1).round() as i32, y0.max(y1).round() as i32),
        );
        let _ = self.area.draw(&Rectangle::new([a, b], c.rgba().filled()));
    }

    pub fn path(&self, pts: &[(f64, f64)], c: impl Into<Col>, w: u32) {
        if pts.len() < 2 {
            return;
        }
        let c: Col = c.into();
        let v: Vec<(i32, i32)> = pts
            .iter()
            .map(|&(x, y)| (x.round() as i32, y.round() as i32))
            .collect();
        let _ = self.area.draw(&PathElement::new(
            v,
            ShapeStyle {
                color: c.rgba(),
                filled: false,
                stroke_width: w,
            },
        ));
    }

    pub fn marker(
        &self,
        kind: Marker,
        x: f64,
        y: f64,
        size: f64,
        c: impl Into<Col>,
        filled: bool,
        w: u32,
    ) {
        let c: Col = c.into();
        let (xi, yi) = (x.round() as i32, y.round() as i32);
        let r = size.round() as i32;
        let st = ShapeStyle {
            color: c.rgba(),
            filled,
            stroke_width: w,
        };
        let _ = match kind {
            Marker::Circle => self.area.draw(&Circle::new((xi, yi), r, st)),
            Marker::TriDown => self.area.draw(&Polygon::new(
                vec![(xi - r, yi - r), (xi + r, yi - r), (xi, yi + r)],
                st,
            )),
            Marker::Square => self
                .area
                .draw(&Rectangle::new([(xi - r, yi - r), (xi + r, yi + r)], st)),
            Marker::Cross => {
                let _ = self.area.draw(&PathElement::new(
                    vec![(xi - r, yi - r), (xi + r, yi + r)],
                    st,
                ));
                self.area.draw(&PathElement::new(
                    vec![(xi - r, yi + r), (xi + r, yi - r)],
                    st,
                ))
            }
        };
    }
}

#[derive(Clone, Copy)]
pub enum Marker {
    Circle,
    TriDown,
    Square,
    Cross,
}

/// Line style.
#[derive(Clone, Copy)]
pub struct Pen {
    pub col: Col,
    pub w: u32,
    /// (dash, gap) in px; None = solid
    pub dash: Option<(f64, f64)>,
}
impl Pen {
    pub fn solid(c: RGBColor, alpha: f64, w: u32) -> Self {
        Pen {
            col: Col(c, alpha),
            w,
            dash: None,
        }
    }
    pub fn dashed(c: RGBColor, w: u32, dash: f64, gap: f64) -> Self {
        Pen {
            col: Col(c, 1.0),
            w,
            dash: Some((dash, gap)),
        }
    }
}

/// Axes rectangle with data transforms.
#[derive(Clone)]
pub struct Ax {
    pub l: f64,
    pub t: f64,
    pub r: f64,
    pub b: f64,
    pub xr: (f64, f64),
    pub yr: (f64, f64),
    pub xlog: bool,
    pub ylog: bool,
    /// y axis increases downward (images)
    pub yflip: bool,
}

fn tr(v: f64, r: (f64, f64), log: bool) -> f64 {
    if log {
        (v.ln() - r.0.ln()) / (r.1.ln() - r.0.ln())
    } else {
        (v - r.0) / (r.1 - r.0)
    }
}

impl Ax {
    pub fn new(l: f64, t: f64, r: f64, b: f64, xr: (f64, f64), yr: (f64, f64)) -> Self {
        Ax {
            l,
            t,
            r,
            b,
            xr,
            yr,
            xlog: false,
            ylog: false,
            yflip: false,
        }
    }
    pub fn px(&self, x: f64) -> f64 {
        self.l + tr(x, self.xr, self.xlog) * (self.r - self.l)
    }
    pub fn py(&self, y: f64) -> f64 {
        let u = tr(y, self.yr, self.ylog);
        if self.yflip {
            self.t + u * (self.b - self.t)
        } else {
            self.b - u * (self.b - self.t)
        }
    }
    pub fn pt(&self, x: f64, y: f64) -> (f64, f64) {
        (self.px(x), self.py(y))
    }

    /// Clipped polyline (data coords), skipping non-finite points; solid or dashed.
    pub fn line(&self, cv: &Canvas, xs: &[f64], ys: &[f64], pen: Pen) {
        let pts: Vec<(f64, f64)> = xs.iter().zip(ys).map(|(&x, &y)| self.pt(x, y)).collect();
        self.line_px(cv, &pts, pen);
    }

    pub fn line_px(&self, cv: &Canvas, pts: &[(f64, f64)], pen: Pen) {
        let mut run: Vec<(f64, f64)> = vec![];
        let flush = |run: &mut Vec<(f64, f64)>| {
            if run.len() >= 2 {
                match pen.dash {
                    None => cv.path(run, pen.col, pen.w),
                    Some((d, g)) => dashed(cv, run, pen, d, g),
                }
            }
            run.clear();
        };
        for i in 0..pts.len() {
            let p = pts[i];
            if !(p.0.is_finite() && p.1.is_finite()) {
                flush(&mut run);
                continue;
            }
            if i == 0 || !(pts[i - 1].0.is_finite() && pts[i - 1].1.is_finite()) {
                if self.inside(p) {
                    run.push(p);
                }
                continue;
            }
            let q = pts[i - 1];
            match self.clip_seg(q, p) {
                None => flush(&mut run),
                Some((a, b)) => {
                    if run
                        .last()
                        .is_none_or(|&l| (l.0 - a.0).abs() > 1e-6 || (l.1 - a.1).abs() > 1e-6)
                    {
                        flush(&mut run);
                        run.push(a);
                    }
                    run.push(b);
                    if b != p {
                        flush(&mut run);
                    }
                }
            }
        }
        flush(&mut run);
    }

    fn inside(&self, p: (f64, f64)) -> bool {
        p.0 >= self.l && p.0 <= self.r && p.1 >= self.t && p.1 <= self.b
    }

    /// Liang-Barsky clip to the axes rectangle.
    fn clip_seg(&self, p: (f64, f64), q: (f64, f64)) -> Option<((f64, f64), (f64, f64))> {
        let (dx, dy) = (q.0 - p.0, q.1 - p.1);
        let (mut u0, mut u1) = (0.0f64, 1.0f64);
        for (pp, qq) in [
            (-dx, p.0 - self.l),
            (dx, self.r - p.0),
            (-dy, p.1 - self.t),
            (dy, self.b - p.1),
        ] {
            if pp == 0.0 {
                if qq < 0.0 {
                    return None;
                }
            } else {
                let r = qq / pp;
                if pp < 0.0 {
                    u0 = u0.max(r);
                } else {
                    u1 = u1.min(r);
                }
            }
        }
        if u0 > u1 {
            return None;
        }
        Some((
            (p.0 + u0 * dx, p.1 + u0 * dy),
            if u1 >= 1.0 {
                q
            } else {
                (p.0 + u1 * dx, p.1 + u1 * dy)
            },
        ))
    }

    pub fn vline(&self, cv: &Canvas, x: f64, pen: Pen) {
        let px = self.px(x);
        self.line_px(cv, &[(px, self.t), (px, self.b)], pen);
    }
    pub fn hline(&self, cv: &Canvas, y: f64, pen: Pen) {
        let py = self.py(y);
        self.line_px(cv, &[(self.l, py), (self.r, py)], pen);
    }

    /// Vertical span between x values, full height, clamped to the axes.
    pub fn span_x(&self, cv: &Canvas, a: f64, b: f64, c: RGBColor) {
        let (x0, x1) = (self.px(a).max(self.l), self.px(b).min(self.r));
        if x1 >= x0 {
            cv.rect(x0, self.t, x1.max(x0 + 1.0), self.b, c);
        }
    }

    pub fn fill_rect(&self, cv: &Canvas, c: RGBColor) {
        cv.rect(self.l, self.t, self.r, self.b, c);
    }

    /// Grid + ticks + tick labels. `xt`/`yt`: (value, label). Draws spines left and bottom.
    pub fn frame(&self, cv: &Canvas, xt: &[(f64, String)], yt: &[(f64, String)]) {
        let g = Pen::solid(GRID, 1.0, 1);
        for (v, _) in xt {
            let x = self.px(*v);
            if x >= self.l - 0.5 && x <= self.r + 0.5 {
                self.line_px(
                    cv,
                    &[
                        (x.clamp(self.l, self.r), self.t),
                        (x.clamp(self.l, self.r), self.b),
                    ],
                    g,
                );
            }
        }
        for (v, _) in yt {
            let y = self.py(*v);
            if y >= self.t - 0.5 && y <= self.b + 0.5 {
                self.line_px(
                    cv,
                    &[
                        (self.l, y.clamp(self.t, self.b)),
                        (self.r, y.clamp(self.t, self.b)),
                    ],
                    g,
                );
            }
        }
    }

    pub fn spines_and_labels(
        &self,
        cv: &Canvas,
        xt: &[(f64, String)],
        yt: &[(f64, String)],
        xlabels: bool,
        fs: f64,
    ) {
        let sp = Pen::solid(INK2, 1.0, 1);
        cv.path(
            &[(self.l, self.t), (self.l, self.b), (self.r, self.b)],
            sp.col,
            1,
        );
        for (v, s) in xt {
            let x = self.px(*v);
            if x < self.l - 0.5 || x > self.r + 0.5 {
                continue;
            }
            cv.path(&[(x, self.b), (x, self.b + 4.0)], sp.col, 1);
            if xlabels {
                cv.text(s, x, self.b + 7.0, fs, INK2, HPos::Center, VPos::Top);
            }
        }
        for (v, s) in yt {
            let y = self.py(*v);
            if y < self.t - 0.5 || y > self.b + 0.5 {
                continue;
            }
            cv.path(&[(self.l - 4.0, y), (self.l, y)], sp.col, 1);
            cv.text(s, self.l - 7.0, y, fs, INK2, HPos::Right, VPos::Center);
        }
    }
}

fn dashed(cv: &Canvas, pts: &[(f64, f64)], pen: Pen, d: f64, g: f64) {
    let mut on = true;
    let mut left = d;
    let mut cur: Vec<(f64, f64)> = vec![pts[0]];
    for w in pts.windows(2) {
        let (mut a, b) = (w[0], w[1]);
        let mut len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        while len > left {
            let u = left / len;
            let m = (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u);
            if on {
                cur.push(m);
                cv.path(&cur, pen.col, pen.w);
                cur.clear();
            } else {
                cur = vec![m];
            }
            on = !on;
            len -= left;
            left = if on { d } else { g };
            a = m;
        }
        left -= len;
        if on {
            cur.push(b);
        }
    }
    if on && cur.len() >= 2 {
        cv.path(&cur, pen.col, pen.w);
    }
}

/// Linear "nice" ticks (1, 2, 5 x 10^k) with about `target` marks.
pub fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<(f64, String)> {
    let span = hi - lo;
    if !(span > 0.0) || !span.is_finite() {
        return vec![];
    }
    let raw = span / target.max(2) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw * 0.999)
        .unwrap_or(10.0 * mag);
    let dec = (-(step.log10().floor()) as i32).max(0) as usize
        + if (step / 10f64.powf(step.log10().floor()) - 2.5).abs() < 1e-9 {
            1
        } else {
            0
        };
    let mut out = vec![];
    let mut v = (lo / step).ceil() * step;
    while v <= hi + step * 1e-9 {
        let vv = if v.abs() < step * 1e-9 { 0.0 } else { v };
        out.push((vv, fmt_minus(&format!("{vv:.dec$}"))));
        v += step;
    }
    out
}

pub fn fmt_minus(s: &str) -> String {
    s.replace('-', "\u{2212}")
}

fn superscript(n: i32) -> String {
    n.to_string()
        .chars()
        .map(|c| match c {
            '-' => '\u{207b}',
            '0' => '\u{2070}',
            '1' => '\u{00b9}',
            '2' => '\u{00b2}',
            '3' => '\u{00b3}',
            d => char::from_u32(0x2070 + d as u32 - '0' as u32).unwrap_or(d),
        })
        .collect()
}

/// Log ticks: decades labelled 10^k; for a range under 2 decades 1-2-5 marks.
pub fn log_ticks(lo: f64, hi: f64) -> Vec<(f64, String)> {
    let (a, b) = (lo.log10().ceil() as i32, hi.log10().floor() as i32);
    if b - a >= 1 {
        return (a..=b)
            .map(|k| (10f64.powi(k), format!("10{}", superscript(k))))
            .collect();
    }
    let mut out = vec![];
    for k in (lo.log10().floor() as i32)..=(hi.log10().ceil() as i32) {
        for m in [1.0, 2.0, 5.0] {
            let v = m * 10f64.powi(k);
            if v >= lo && v <= hi {
                out.push((v, format!("{v}")));
            }
        }
    }
    out
}

/// One legend entry.
pub struct LegendItem {
    pub label: String,
    pub pen: Pen,
}

fn legend_sample(cv: &Canvas, it: &LegendItem, x: f64, y: f64) {
    let a = Ax::new(x, y - 3.0, x + 26.0, y + 3.0, (0.0, 1.0), (0.0, 1.0));
    let mut pen = it.pen;
    pen.col.1 = pen.col.1.max(0.8);
    a.line_px(cv, &[(x, y), (x + 26.0, y)], pen);
}

/// Legend entries in one row, right-aligned at `right`, vertically centred on `y`.
pub fn legend_row(cv: &Canvas, items: &[LegendItem], right: f64, y: f64, fs: f64) {
    let widths: Vec<i32> = items
        .iter()
        .map(|i| 26 + 7 + cv.text_w(&i.label, fs))
        .collect();
    let total: i32 = widths.iter().sum::<i32>() + 22 * (items.len() as i32 - 1);
    let mut x = right - total as f64;
    for (it, w) in items.iter().zip(&widths) {
        legend_sample(cv, it, x, y);
        cv.text(&it.label, x + 33.0, y, fs, INK, HPos::Left, VPos::Center);
        x += *w as f64 + 22.0;
    }
}

/// Legend entries in one column, right-aligned at `right`, starting at `top`.
pub fn legend_col(cv: &Canvas, items: &[LegendItem], right: f64, top: f64, fs: f64) {
    let mut y = top + fs * 0.7;
    for it in items {
        let w = 26 + 7 + cv.text_w(&it.label, fs);
        let x = right - w as f64;
        legend_sample(cv, it, x, y);
        cv.text(&it.label, x + 33.0, y, fs, INK, HPos::Left, VPos::Center);
        y += fs * 1.5;
    }
}

/// `%g`-style label text (6 significant digits, trailing zeros stripped).
pub fn fmt_g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let e = x.abs().log10().floor() as i32;
    if !(-5..6).contains(&e) {
        let s = format!("{:.5e}", x);
        let (m, ex) = s.split_once('e').unwrap();
        let m = if m.contains('.') {
            m.trim_end_matches('0').trim_end_matches('.')
        } else {
            m
        };
        let ex: i32 = ex.parse().unwrap();
        return format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs());
    }
    let dec = (5 - e).max(0) as usize;
    let s = format!("{x:.dec$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}
