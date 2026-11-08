//! Frequency laws: [`Comp`] and `fit_law`.

use ts_dsp::linalg::lstsq;
use ts_dsp::optimize::minimize_scalar_bounded;
use ts_dsp::stats::{argmin, np_sum};
use ts_dsp::window::linspace;

/// `Comp.kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Fixed,
    Chirp,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Fixed => "fixed",
            Kind::Chirp => "chirp",
        }
    }
}

/// Periodic component family, `f(t) = a + b / sqrt(1 + beta t)` [Hz].
/// Equality ignores `pts`.
#[derive(Debug, Clone)]
pub struct Comp {
    pub name: String,
    pub a: f64,
    pub b: f64,
    pub beta: f64,
    pub harmonics: Vec<u32>,
    pub kind: Kind,
    /// L2 frequency (mean of the law over the fitted span).
    pub f_mean: f64,
    /// Detection strength (info).
    pub amp_db: f64,
    /// Ridge points `(t, f/h, weight)`.
    pub pts: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
}

impl PartialEq for Comp {
    fn eq(&self, o: &Self) -> bool {
        self.name == o.name
            && self.a == o.a
            && self.b == o.b
            && self.beta == o.beta
            && self.harmonics == o.harmonics
            && self.kind == o.kind
            && self.f_mean == o.f_mean
            && self.amp_db == o.amp_db
    }
}

impl Comp {
    /// `Comp(name, a, b, beta)` with defaults: harmonics (1, 2, 3), kind fixed,
    /// `f_mean = a + b` (`__post_init__`).
    pub fn new(name: &str, a: f64, b: f64, beta: f64) -> Self {
        Comp {
            name: name.to_string(),
            a,
            b,
            beta,
            harmonics: vec![1, 2, 3],
            kind: Kind::Fixed,
            f_mean: a + b,
            amp_db: 0.0,
            pts: None,
        }
    }

    /// `Comp.freq` at one time.
    #[inline]
    pub fn freq_at(&self, t: f64) -> f64 {
        if self.beta == 0.0 {
            self.a + self.b
        } else {
            self.a + self.b / (1.0 + self.beta * t).max(1e-6).sqrt()
        }
    }

    pub fn freq(&self, t: &[f64]) -> Vec<f64> {
        t.iter().map(|&v| self.freq_at(v)).collect()
    }

    /// `Comp.phase` [cycles], phase(0) = 0.
    #[inline]
    pub fn phase_at(&self, t: f64) -> f64 {
        if self.beta == 0.0 {
            (self.a + self.b) * t
        } else {
            let s = (1.0 + self.beta * t).max(1e-6).sqrt();
            self.a * t + 2.0 * self.b * (s - 1.0) / self.beta
        }
    }

    pub fn phase(&self, t: &[f64]) -> Vec<f64> {
        t.iter().map(|&v| self.phase_at(v)).collect()
    }

    /// `float(np.mean(c.freq(np.linspace(*span, 200))))`.
    pub fn mean_freq(&self, span: (f64, f64)) -> f64 {
        np_sum(&self.freq(&linspace(span.0, span.1, 200))) / 200.0
    }
}

/// Result of [`fit_law`]: `(a, b, beta, rms, kind)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Law {
    pub a: f64,
    pub b: f64,
    pub beta: f64,
    pub rms: f64,
    pub kind: Kind,
}

/// `np.geomspace(start, stop, num)` for positive `start`, `stop`.
fn geomspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let mut r: Vec<f64> = linspace(start.log10(), stop.log10(), num)
        .into_iter()
        .map(|y| 10f64.powf(y))
        .collect();
    if num > 0 {
        r[0] = start;
        if num > 1 {
            r[num - 1] = stop;
        }
    }
    r
}

/// `fit_law(tf, ff, w, span, n_beta)`: fits `f = a + b/sqrt(1 + beta t)` to points; chooses
/// between constant and law by BIC; beta restricted so that `1 + beta t > 0.1` on the span.
pub fn fit_law(
    tf: &[f64],
    ff: &[f64],
    w: Option<&[f64]>,
    span: Option<(f64, f64)>,
    n_beta: usize,
) -> Law {
    let n = tf.len();
    let ones;
    let w = match w {
        Some(w) => w,
        None => {
            ones = vec![1.0; n];
            &ones
        }
    };
    let (t0, t1) = span.unwrap_or_else(|| (0.0, tf.iter().copied().fold(f64::NAN, f64::max)));
    let sw = np_sum(w);
    let wf: Vec<f64> = w.iter().zip(ff).map(|(a, b)| a * b).collect();
    let fc = np_sum(&wf) / sw;
    let r0: Vec<f64> = w
        .iter()
        .zip(ff)
        .map(|(a, b)| a * ((b - fc) * (b - fc)))
        .collect();
    let sse0 = np_sum(&r0);
    // (a, b, beta, sse)
    let mut best = (fc, 0.0, 0.0, sse0);
    if n >= 5 {
        let bmin = -0.9 / t1.max(1.0);
        let h = n_beta / 2;
        let mut betas: Vec<f64> = geomspace(1e-6, -bmin, h)
            .into_iter()
            .rev()
            .map(|v| -v)
            .collect();
        betas.extend(geomspace(1e-6, 10.0 / t1.max(1.0), h));
        let sqw: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
        let rhs: Vec<f64> = ff.iter().zip(&sqw).map(|(f, s)| f * s).collect();
        let mut a_mat = vec![0.0; 2 * n];
        let mut g = vec![0.0; n];
        let mut res = vec![0.0; n];
        let mut ls = |be: f64| -> (f64, f64, f64, f64) {
            for i in 0..n {
                g[i] = 1.0 / (1.0 + be * tf[i]).sqrt();
                a_mat[2 * i] = sqw[i];
                a_mat[2 * i + 1] = g[i] * sqw[i];
            }
            let coef = match lstsq(&a_mat, n, 2, &rhs, None) {
                Ok(l) => l.x,
                Err(_) => vec![f64::NAN; 2],
            };
            for i in 0..n {
                let e = (ff[i] - coef[0]) - coef[1] * g[i];
                res[i] = w[i] * (e * e);
            }
            (coef[0], coef[1], be, np_sum(&res))
        };
        let all: Vec<(f64, f64, f64, f64)> = betas.iter().map(|&be| ls(be)).collect();
        let sses: Vec<f64> = all.iter().map(|q| q.3).collect();
        let i = argmin(&sses).unwrap_or(0);
        if all[i].3 < best.3 {
            best = all[i];
            let lo = betas[i.saturating_sub(1)];
            let hi = betas[(i + 1).min(betas.len() - 1)];
            if let Ok(o) = minimize_scalar_bounded(|be| ls(be).3, (lo, hi), 1e-12, 500) {
                let q = ls(o.x);
                if q.3 < best.3 {
                    best = q;
                }
            }
        }
    }
    let nf = n as f64;
    let bic0 = nf * (sse0 / nf + 1e-30).ln() + nf.ln();
    let bic1 = nf * (best.3 / nf + 1e-30).ln() + 3.0 * nf.ln();
    if best.2 != 0.0 && bic1 < bic0 - 6.0 {
        let (a, b, be, sse) = best;
        let g0 = 1.0 / (1.0 + be * t0).sqrt();
        let g1 = 1.0 / (1.0 + be * t1).sqrt();
        let lim = 0.01f64.max(0.002 * fc.abs()).max(4.0 * (sse / sw).sqrt());
        if (b * (g1 - g0)).abs() > lim {
            return Law {
                a,
                b,
                beta: be,
                rms: (sse / sw).sqrt(),
                kind: Kind::Chirp,
            };
        }
    }
    Law {
        a: fc,
        b: 0.0,
        beta: 0.0,
        rms: (sse0 / sw).sqrt(),
        kind: Kind::Fixed,
    }
}
