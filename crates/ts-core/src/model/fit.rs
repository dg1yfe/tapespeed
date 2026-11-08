//! L1/L2/L3 weighted least-squares fit.
//!
//! The Jacobian is never stored. Rows (valid samples, decimated) are grouped into runs of
//! consecutive rows with the same column pattern (same drift B-spline interval, same L3
//! amplitude knot). For each run a dense local Gram `Σ_i a_i (w_i a_i)ᵀ` (lower triangle) and
//! `Σ_i w_i a_i y_i` are computed (runs in parallel, rayon), then scattered into the global
//! sparse normal matrix in run order (fixed chunks of runs, sequential merge) → bit-identical
//! for any thread count.
//!
//! `N + R` is solved by faer's supernodal sparse Cholesky. The parameters are put in a
//! time-major elimination order (each parameter at the end of its basis support: L3 knot `m`
//! at `t0 + (m+1) Ta`, drift coefficient `j` at `knots[j+4]`, L2 amplitudes last). The
//! normal matrix is block tridiagonal in time, so this order fills like AMD (L3 43 min,
//! 14 comps: 9.23 M vs 9.16 M supernodal factor values) but lets the matrix be assembled
//! directly in factor order: faer runs without reordering and keeps no permuted copy of `A`
//! (−70 MB at that size). The symbolic analysis is shared by the IRLS iterations.

use indexmap::IndexMap;
use num_complex::Complex64;
use rayon::prelude::*;
use ts_dsp::ValueError;
use ts_dsp::bspline;
use ts_dsp::linalg::{SpdSolver, SymCsc};
use ts_dsp::stats::{interp, np_sum};

use super::law::Comp;
use super::{Amp, FitStats, Model, hat};

/// Keyword arguments of `fit(t, r, w, comps, level, Td, Ta, lam, irls, r0, dec, lam_d)`.
#[derive(Debug, Clone, Copy)]
pub struct FitOpts {
    pub level: u8,
    pub td: f64,
    pub ta: f64,
    pub lam: f64,
    pub irls: usize,
    pub r0: Option<f64>,
    pub dec: usize,
    pub lam_d: f64,
}

impl Default for FitOpts {
    fn default() -> Self {
        FitOpts {
            level: 3,
            td: 8.0,
            ta: 20.0,
            lam: 0.3,
            irls: 3,
            r0: None,
            dec: 2,
            lam_d: 1e-3,
        }
    }
}

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
/// Maximum rows per run (bounds the per-thread run buffers).
const MAX_RUN: usize = 1024;
/// Runs computed in parallel per merge step (fixed: independent of the thread count).
const RUN_CHUNK: usize = 64;

/// `np.median` computed in place (reorders `a`); NaN for empty input or any NaN.
fn median_mut(a: &mut [f64]) -> f64 {
    let n = a.len();
    if n == 0 || a.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let h = n / 2;
    let (lo, m, _) = a.select_nth_unstable_by(h, |x, y| x.partial_cmp(y).unwrap());
    let m = *m;
    if n % 2 == 1 {
        m
    } else {
        (lo.iter().copied().fold(f64::NEG_INFINITY, f64::max) + m) / 2.0
    }
}

/// Robust L1 level: median of the valid samples + 5 Huber-ish refinements
/// (`ww = min(1, 1.5 s / (|e| + 1e-30))`, `r0 += Σ ww e / Σ ww`).
fn robust_level(rv: &[f64]) -> f64 {
    let mut s = rv.to_vec();
    let mut r0 = median_mut(&mut s);
    let ww = |e: f64, sc: f64| {
        let q = 1.5 * sc / (e.abs() + 1e-30);
        if q < 1.0 { q } else { 1.0 }
    };
    for _ in 0..5 {
        for (o, &v) in s.iter_mut().zip(rv) {
            *o = (v - r0).abs();
        }
        let sc = 1.4826 * median_mut(&mut s) + 1e-12;
        for (o, &v) in s.iter_mut().zip(rv) {
            let e = v - r0;
            *o = ww(e, sc) * e;
        }
        let num = np_sum(&s);
        for (o, &v) in s.iter_mut().zip(rv) {
            *o = ww(v - r0, sc);
        }
        r0 += num / np_sum(&s);
    }
    r0
}

/// Row design of a level-2/3 fit (all values computed on the fly).
struct Design<'a> {
    level: u8,
    comps: &'a [Comp],
    /// (k, h)
    keys: Vec<(usize, u32)>,
    nd: usize,
    m_knots: usize,
    ta: f64,
    knots: Vec<f64>,
    span: (f64, f64),
    ts: Vec<f64>,
    /// first drift column of each row (B-spline interval − 3)
    boff: Vec<u32>,
    mi: Vec<u32>,
    ui: Vec<f64>,
}

impl Design<'_> {
    fn p(&self) -> usize {
        self.nd + 2 * self.keys.len() * self.m_knots
    }

    fn row_len(&self) -> usize {
        4 + self.keys.len() * if self.level >= 3 { 4 } else { 2 }
    }

    /// B-spline values of row `i` (`BSpline.design_matrix` row, bit-identical).
    fn bvals(&self, i: usize, work: &mut [f64; 8]) {
        let x = self.ts[i].clamp(self.span.0, self.span.1);
        bspline::de_boor(&self.knots, x, 3, self.boff[i] as usize + 3, work);
    }

    /// Ascending column indices of row `i`.
    fn row_cols(&self, i: usize, out: &mut [u32]) {
        let b = self.boff[i];
        for (a, o) in out[..4].iter_mut().enumerate() {
            *o = b + a as u32;
        }
        let nd = self.nd as u32;
        if self.level >= 3 {
            let mm = self.m_knots as u32;
            let m = self.mi[i];
            for q in 0..self.keys.len() as u32 {
                let c0 = nd + 2 * q * mm + m;
                let s0 = c0 + mm;
                let o = &mut out[4 + 4 * q as usize..8 + 4 * q as usize];
                o.copy_from_slice(&[c0, c0 + 1, s0, s0 + 1]);
            }
        } else {
            for q in 0..self.keys.len() as u32 {
                out[4 + 2 * q as usize] = nd + 2 * q;
                out[5 + 2 * q as usize] = nd + 2 * q + 1;
            }
        }
    }

    /// Values of row `i` in [`Self::row_cols`] order (B-spline row, then per key
    /// `wt * cos`, `wt * (-sin)` blocks on knots m, m+1 (L3) or `cos`, `-sin` (L2)).
    fn row_vals(&self, i: usize, out: &mut [f64]) {
        let mut work = [0.0; 8];
        self.bvals(i, &mut work);
        out[..4].copy_from_slice(&work[..4]);
        let t = self.ts[i];
        let mut last_k = usize::MAX;
        let mut ph = 0.0;
        for (q, &(k, h)) in self.keys.iter().enumerate() {
            if k != last_k {
                let c = &self.comps[k];
                ph = TWO_PI
                    * if self.level >= 3 {
                        c.phase_at(t)
                    } else {
                        c.f_mean * t
                    };
                last_k = k;
            }
            let x = h as f64 * ph;
            let co = x.cos();
            let si = -x.sin();
            if self.level >= 3 {
                let u = self.ui[i];
                let o = &mut out[4 + 4 * q..8 + 4 * q];
                o[0] = (1.0 - u) * co;
                o[1] = u * co;
                o[2] = (1.0 - u) * si;
                o[3] = u * si;
            } else {
                out[4 + 2 * q] = co;
                out[5 + 2 * q] = si;
            }
        }
    }

    fn same_pattern(&self, i: usize, j: usize) -> bool {
        self.boff[i] == self.boff[j] && (self.level < 3 || self.mi[i] == self.mi[j])
    }

    /// Runs `[a, b)` of consecutive rows with identical column patterns.
    fn runs(&self) -> Vec<(usize, usize)> {
        let n = self.ts.len();
        let mut out = Vec::new();
        let mut a = 0;
        while a < n {
            let mut b = a + 1;
            while b < n && b - a < MAX_RUN && self.same_pattern(a, b) {
                b += 1;
            }
            out.push((a, b));
            a = b;
        }
        out
    }

    /// Time-major elimination order: `perm[j]` = factor position of parameter `j`.
    fn order(&self) -> (Vec<u32>, Vec<u32>) {
        let p = self.p();
        let mm = self.m_knots;
        let key = |j: usize| -> f64 {
            if j < self.nd {
                self.knots[j + 4]
            } else if self.level >= 3 {
                let m = (j - self.nd) % mm;
                self.span.0 + (m + 1) as f64 * self.ta
            } else {
                f64::INFINITY
            }
        };
        let mut inv: Vec<u32> = (0..p as u32).collect();
        inv.sort_by(|&a, &b| {
            key(a as usize)
                .partial_cmp(&key(b as usize))
                .unwrap()
                .then(a.cmp(&b))
        });
        let mut perm = vec![0u32; p];
        for (pos, &j) in inv.iter().enumerate() {
            perm[j as usize] = pos as u32;
        }
        (perm, inv)
    }
}

/// `(DᵀD)[i][j]` of the second-difference operator `D` ((n-2) × n, rows [1, -2, 1]).
fn d2td2(i: usize, j: usize, n: usize) -> f64 {
    const C: [f64; 3] = [1.0, -2.0, 1.0];
    let (i, j) = if i >= j { (i, j) } else { (j, i) };
    if i - j > 2 || n < 3 {
        return 0.0;
    }
    let lo = i.saturating_sub(2);
    let hi = j.min(n - 3);
    let mut s = 0.0;
    for r in lo..=hi {
        s += C[i - r] * C[j - r];
    }
    s
}

/// Regularisation neighbours of parameter `j` (pattern of the 2nd-difference priors).
fn reg_neighbours(d: &Design, j: usize, l3_reg: bool, out: &mut Vec<usize>) {
    out.clear();
    let nd = d.nd;
    if j < nd {
        if nd >= 3 {
            out.extend(j.saturating_sub(2)..(j + 3).min(nd));
        }
    } else if l3_reg {
        let mm = d.m_knots;
        let o = nd + (j - nd) / mm * mm;
        let m = (j - nd) % mm;
        out.extend((o + m.saturating_sub(2))..(o + (m + 3).min(mm)));
    }
}

/// Sparsity pattern of `N + R` (lower triangle) in factor order, built column by column
/// without intermediate per-column lists.
fn build_pattern(
    d: &Design,
    runs: &[(usize, usize)],
    l3_reg: bool,
    perm: &[u32],
    inv: &[u32],
) -> SymCsc {
    let p = d.p();
    let rl = d.row_len();
    let mut cols = vec![0u32; rl];
    // runs containing each parameter: a contiguous range (rows are time-sorted)
    let mut first = vec![u32::MAX; p];
    let mut last = vec![0u32; p];
    for (ri, &(a, _)) in runs.iter().enumerate() {
        d.row_cols(a, &mut cols);
        for &c in &cols {
            let c = c as usize;
            first[c] = first[c].min(ri as u32);
            last[c] = ri as u32;
        }
    }
    let mut mark = vec![u32::MAX; p];
    let mut col_ptr = Vec::with_capacity(p + 1);
    col_ptr.push(0u32);
    let mut row_idx: Vec<u32> = Vec::new();
    let mut tmp: Vec<u32> = Vec::new();
    let mut nb = Vec::new();
    for jn in 0..p as u32 {
        let j = inv[jn as usize] as usize;
        tmp.clear();
        let mut add = |c: u32, tmp: &mut Vec<u32>| {
            if c >= jn && mark[c as usize] != jn {
                mark[c as usize] = jn;
                tmp.push(c);
            }
        };
        add(jn, &mut tmp);
        if first[j] != u32::MAX {
            let mut prev = usize::MAX;
            for ri in first[j]..=last[j] {
                let a = runs[ri as usize].0;
                if prev != usize::MAX && d.same_pattern(prev, a) {
                    continue;
                }
                prev = a;
                d.row_cols(a, &mut cols);
                for &c in &cols {
                    add(perm[c as usize], &mut tmp);
                }
            }
        }
        reg_neighbours(d, j, l3_reg, &mut nb);
        for &c in &nb {
            add(perm[c], &mut tmp);
        }
        tmp.sort_unstable();
        row_idx.extend_from_slice(&tmp);
        col_ptr.push(row_idx.len() as u32);
    }
    row_idx.shrink_to_fit();
    SymCsc {
        n: p,
        val: vec![0.0; row_idx.len()],
        col_ptr,
        row_idx,
    }
}

/// Dot product with 8 independent accumulators (fixed order: deterministic).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut acc = [0.0f64; 8];
    let (ac, ar) = a.split_at(n - n % 8);
    let (bc, br) = b.split_at(n - n % 8);
    for (x, y) in ac.chunks_exact(8).zip(bc.chunks_exact(8)) {
        for l in 0..8 {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ar.iter().zip(br) {
        s = x.mul_add(*y, s);
    }
    s
}

/// Per-thread buffers of [`run_gram`].
#[derive(Default)]
struct RunBufs {
    xt: Vec<f64>,
    xw: Vec<f64>,
    row: Vec<f64>,
}

/// Local Gram (lower triangle packed by column: (q, p ≥ q)) and right-hand side of a run.
fn run_gram(
    d: &Design,
    a: usize,
    b: usize,
    wr: &[f64],
    ys: &[f64],
    bufs: &mut RunBufs,
) -> (Vec<f64>, Vec<f64>) {
    let rl = d.row_len();
    let r = b - a;
    bufs.xt.resize(rl * r, 0.0);
    bufs.xw.resize(rl * r, 0.0);
    bufs.row.resize(rl, 0.0);
    let (xt, xw, row) = (&mut bufs.xt, &mut bufs.xw, &mut bufs.row);
    // column-major: xt[c * r + i]
    for i in 0..r {
        d.row_vals(a + i, row);
        let w = wr[a + i];
        for c in 0..rl {
            xt[c * r + i] = row[c];
            xw[c * r + i] = row[c] * w;
        }
    }
    let mut g = Vec::with_capacity(rl * (rl + 1) / 2);
    for q in 0..rl {
        let wq = &xw[q * r..(q + 1) * r];
        for p in q..rl {
            g.push(dot(&xt[p * r..(p + 1) * r], wq));
        }
    }
    let yl = &ys[a..b];
    let bl = (0..rl).map(|c| dot(&xw[c * r..(c + 1) * r], yl)).collect();
    (g, bl)
}

/// `N = Aᵀ W A` (into `nmat.val` in factor order, zeroed first) and `b = Aᵀ W y` (original
/// order).
fn assemble(
    d: &Design,
    runs: &[(usize, usize)],
    wr: &[f64],
    ys: &[f64],
    perm: &[u32],
    nmat: &mut SymCsc,
) -> Vec<f64> {
    nmat.val.iter_mut().for_each(|v| *v = 0.0);
    let mut bvec = vec![0.0; d.p()];
    let rl = d.row_len();
    // packed offset of local column q
    let mut off = Vec::with_capacity(rl);
    let mut acc = 0;
    for q in 0..rl {
        off.push(acc);
        acc += rl - q;
    }
    let mut cols = vec![0u32; rl];
    let mut mapped: Vec<(u32, u32)> = Vec::with_capacity(rl);
    for chunk in runs.chunks(RUN_CHUNK) {
        let parts: Vec<(Vec<f64>, Vec<f64>)> = chunk
            .par_iter()
            .map_init(RunBufs::default, |bufs, &(a, b)| {
                run_gram(d, a, b, wr, ys, bufs)
            })
            .collect();
        for (&(a, _), (g, bl)) in chunk.iter().zip(parts) {
            d.row_cols(a, &mut cols);
            for (q, &c) in cols.iter().enumerate() {
                bvec[c as usize] += bl[q];
            }
            mapped.clear();
            mapped.extend(
                cols.iter()
                    .enumerate()
                    .map(|(q, &c)| (perm[c as usize], q as u32)),
            );
            mapped.sort_unstable();
            for s in 0..rl {
                let (cs, ls) = mapped[s];
                let ls = ls as usize;
                let mut pos = nmat.col_ptr[cs as usize] as usize;
                for &(cr, lr) in &mapped[s..] {
                    while nmat.row_idx[pos] != cr {
                        pos += 1;
                    }
                    let lr = lr as usize;
                    let (lo, hi) = if lr >= ls { (ls, lr) } else { (lr, ls) };
                    nmat.val[pos] += g[off[lo] + hi - lo];
                }
            }
        }
    }
    bvec
}

/// Residuals `ys - A x` (row sums in ascending column order, as scipy's CSR matvec).
fn residuals(d: &Design, x: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = ys.len();
    let rl = d.row_len();
    let mut e = vec![0.0; n];
    e.par_chunks_mut(4096).enumerate().for_each(|(c, ec)| {
        let mut cols = vec![0u32; rl];
        let mut vals = vec![0.0; rl];
        for (q, ev) in ec.iter_mut().enumerate() {
            let i = c * 4096 + q;
            d.row_cols(i, &mut cols);
            d.row_vals(i, &mut vals);
            let mut s = 0.0;
            for (&cc, &v) in cols.iter().zip(&vals) {
                s += v * x[cc as usize];
            }
            *ev = ys[i] - s;
        }
    });
    e
}

/// Tukey biweights (`c = 4.685`) (`w > 0` for all fitted rows); `e` is overwritten.
fn tukey_into(e: &mut [f64], out: &mut [f64]) {
    for (o, v) in out.iter_mut().zip(e.iter()) {
        *o = *v;
    }
    for v in e.iter_mut() {
        *v = v.abs();
    }
    let s = 1.4826 * median_mut(e) + 1e-12;
    let cs = 4.685 * s;
    for o in out.iter_mut() {
        let u = *o / cs;
        *o = if u.abs() < 1.0 {
            let q = 1.0 - u * u;
            q * q
        } else {
            0.0
        };
    }
}

/// Support mask `s > 0.05 * median(s[s > 0])` (all true if no positive entry).
fn support(s: &[f64]) -> Vec<bool> {
    let mut pos: Vec<f64> = s.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        return vec![true; s.len()];
    }
    let thr = 0.05 * median_mut(&mut pos);
    s.iter().map(|&v| v > thr).collect()
}

/// `fit(t, r, w, comps, level, Td, Ta, lam, irls, r0, dec, lam_d)`: model level 1..3 on the
/// samples with weight `w > 0` (decimated by `dec`), Tukey IRLS.
pub fn fit(
    t: &[f64],
    r: &[f64],
    w: &[f64],
    comps: &[Comp],
    o: &FitOpts,
) -> Result<Model, ValueError> {
    if t.is_empty() || r.len() != t.len() || w.len() != t.len() {
        return Err(ValueError(
            "fit: t, r, w must have the same non-zero length".into(),
        ));
    }
    let span = (t[0], t[t.len() - 1]);
    let r0 = match o.r0 {
        Some(v) => v,
        None => {
            let rv: Vec<f64> = r
                .iter()
                .zip(w)
                .filter(|&(_, &wv)| wv > 0.0)
                .map(|(&v, _)| v)
                .collect();
            robust_level(&rv)
        }
    };
    let mut m = Model {
        level: o.level,
        r0,
        span,
        comps: comps.to_vec(),
        td: o.td,
        drift_knots: None,
        drift_coef: None,
        ta: o.ta,
        amp: IndexMap::new(),
        stats: None,
    };
    if o.level <= 1 {
        return Ok(m);
    }
    let sel: Vec<usize> = (0..t.len())
        .filter(|&i| w[i] > 0.0)
        .step_by(o.dec.max(1))
        .collect();
    if sel.is_empty() {
        return Err(ValueError("fit: no samples with w > 0".into()));
    }
    let ts: Vec<f64> = sel.iter().map(|&i| t[i]).collect();
    let ys: Vec<f64> = sel.iter().map(|&i| r[i] - r0).collect();
    let ws: Vec<f64> = sel.iter().map(|&i| w[i]).collect();
    drop(sel);
    let n = ts.len();

    // drift basis: cubic B-spline, knot spacing ~Td
    let (t0, t1) = span;
    let nseg = (((t1 - t0) / o.td).ceil() as i64).max(1) as usize;
    let step = (t1 - t0) / nseg as f64;
    let knots: Vec<f64> = (-3..nseg as i64 + 4)
        .map(|i| t0 + step * i as f64)
        .collect();
    let mut boff = Vec::with_capacity(n);
    let mut ind = 3;
    for &v in &ts {
        let l = bspline::find_interval(&knots, 3, v.clamp(t0, t1), ind, false)
            .ok_or_else(|| ValueError("fit: drift basis out of bounds".into()))?;
        ind = l;
        boff.push((l - 3) as u32);
    }
    let nd = nseg + 3;
    let m_knots = if o.level >= 3 {
        ((t1 - t0) / o.ta).ceil() as usize + 1
    } else {
        1
    };
    let (mi, ui): (Vec<u32>, Vec<f64>) = if o.level >= 3 {
        ts.iter()
            .map(|&v| {
                let (a, b) = hat(v, t0, o.ta, m_knots);
                (a as u32, b)
            })
            .unzip()
    } else {
        (vec![0; n], vec![0.0; n])
    };
    let keys: Vec<(usize, u32)> = comps
        .iter()
        .enumerate()
        .flat_map(|(k, c)| c.harmonics.iter().map(move |&h| (k, h)))
        .collect();
    let d = Design {
        level: o.level,
        comps,
        keys,
        nd,
        m_knots,
        ta: o.ta,
        knots,
        span,
        ts,
        boff,
        mi,
        ui,
    };
    let p = d.p();
    let l3_reg = o.level >= 3 && m_knots >= 3 && p > nd;
    let runs = d.runs();
    let (perm, inv) = d.order();
    let mut nmat = build_pattern(&d, &runs, l3_reg, &perm, &inv);
    let solver = SpdSolver::analyze_identity(&nmat)?;

    let mut wr = ws.clone();
    let mut x = vec![0.0; p];
    let mut dg = vec![0.0; p];
    let mut bp = vec![0.0; p];
    for it in 0..=o.irls {
        let bvec = assemble(&d, &runs, &wr, &ys, &perm, &mut nmat);
        for (j, v) in dg.iter_mut().enumerate() {
            let jp = perm[j] as usize;
            *v = nmat.val[nmat.col_ptr[jp] as usize]; // diagonal = first entry of its column
        }
        let mut sc = if p > nd {
            dg[nd..].to_vec()
        } else {
            dg.clone()
        };
        let scale = median_mut(&mut sc);
        let ridge = 1e-6 * scale + 1e-30;
        let mut dpos: Vec<f64> = dg[..nd].iter().copied().filter(|&v| v > 0.0).collect();
        let sd = if dpos.is_empty() {
            1.0
        } else {
            median_mut(&mut dpos)
        };
        let dfac = o.lam_d * sd;
        let lfac = o.lam * scale;
        // N + R, R = (ridge·I + lam_d·sd·DdᵀDd) + lam·scale·D2ᵀD2 (per amplitude block)
        for jn in 0..p {
            let j = inv[jn] as usize;
            let (s, e) = (nmat.col_ptr[jn] as usize, nmat.col_ptr[jn + 1] as usize);
            for pos in s..e {
                let i = inv[nmat.row_idx[pos] as usize] as usize;
                let mut rv = if i == j { ridge } else { 0.0 };
                if nd >= 3 && i < nd && j < nd {
                    rv += dfac * d2td2(i, j, nd);
                }
                if l3_reg && i >= nd && j >= nd && (i - nd) / m_knots == (j - nd) / m_knots {
                    rv += lfac * d2td2((i - nd) % m_knots, (j - nd) % m_knots, m_knots);
                }
                nmat.val[pos] += rv;
            }
        }
        for (j, &v) in bvec.iter().enumerate() {
            bp[perm[j] as usize] = v;
        }
        let xp = solver.solve(&nmat, &bp)?;
        for (j, v) in x.iter_mut().enumerate() {
            *v = xp[perm[j] as usize];
        }
        if it < o.irls {
            let mut e = residuals(&d, &x, &ys);
            tukey_into(&mut e, &mut wr);
            for (a, &b) in wr.iter_mut().zip(&ws) {
                *a *= b;
            }
        }
    }
    drop(nmat);
    drop(solver);

    // knots without data support: drift interpolated / held, L3 amplitudes interpolated
    // inside and 0 beyond the data
    let mut sdv = vec![0.0; nd];
    let mut work = [0.0; 8];
    for (i, &wi) in wr.iter().enumerate() {
        let b = d.boff[i] as usize;
        d.bvals(i, &mut work);
        for a in 0..4 {
            sdv[b + a] += work[a] * wi;
        }
    }
    let sup = support(&sdv);
    let mut xd = x[..nd].to_vec();
    if sup.iter().any(|&v| v) && !sup.iter().all(|&v| v) {
        let g: Vec<f64> = (0..nd).filter(|&i| sup[i]).map(|i| i as f64).collect();
        let gv: Vec<f64> = (0..nd).filter(|&i| sup[i]).map(|i| xd[i]).collect();
        let miss: Vec<usize> = (0..nd).filter(|&i| !sup[i]).collect();
        let mf: Vec<f64> = miss.iter().map(|&i| i as f64).collect();
        for (&i, v) in miss.iter().zip(interp(&mf, &g, &gv)?) {
            xd[i] = v;
        }
    }
    let mm = m_knots;
    let (supa, cap) = if o.level >= 3 {
        let mut s1 = vec![0.0; mm];
        let mut s2 = vec![0.0; mm];
        for i in 0..n {
            s1[d.mi[i] as usize] += (1.0 - d.ui[i]) * wr[i];
            s2[d.mi[i] as usize + 1] += d.ui[i] * wr[i];
        }
        let sa: Vec<f64> = s1.iter().zip(&s2).map(|(a, b)| a + b).collect();
        let wm: Vec<f64> = wr.iter().map(|&v| v.max(1e-12)).collect();
        let yw: Vec<f64> = ys.iter().zip(&wm).map(|(y, w)| (y * y) * w).collect();
        let avg = np_sum(&yw) / np_sum(&wm);
        (support(&sa), 2.0 * 2f64.sqrt() * avg.sqrt())
    } else {
        (vec![], 0.0)
    };
    let mut j = nd;
    for &key in &d.keys {
        if o.level == 2 {
            m.amp.insert(key, Amp::L2(Complex64::new(x[j], x[j + 1])));
            j += 2;
        } else {
            let mut a: Vec<Complex64> = (0..mm)
                .map(|i| Complex64::new(x[j + i], x[j + mm + i]))
                .collect();
            let ok: Vec<bool> = (0..mm).map(|i| supa[i] && a[i].norm() <= cap).collect();
            let any = ok.iter().any(|&v| v);
            if any && !ok.iter().all(|&v| v) {
                let g: Vec<usize> = (0..mm).filter(|&i| ok[i]).collect();
                let gf: Vec<f64> = g.iter().map(|&i| i as f64).collect();
                let gre: Vec<f64> = g.iter().map(|&i| a[i].re).collect();
                let gim: Vec<f64> = g.iter().map(|&i| a[i].im).collect();
                let all: Vec<f64> = (0..mm).map(|i| i as f64).collect();
                let ire = interp(&all, &gf, &gre)?;
                let iim = interp(&all, &gf, &gim)?;
                let (g0, g1) = (g[0], g[g.len() - 1]);
                for i in 0..mm {
                    if !ok[i] {
                        a[i] = Complex64::new(ire[i], iim[i]);
                    }
                    if i < g0 || i > g1 {
                        a[i] = Complex64::new(0.0, 0.0);
                    }
                }
            }
            if !any {
                a = vec![Complex64::new(0.0, 0.0); mm];
            }
            m.amp.insert(key, Amp::L3(a));
            j += 2 * mm;
        }
    }
    m.drift_knots = Some(d.knots.clone());
    m.drift_coef = Some(xd);
    let nlow = wr.iter().filter(|&&r| r < 0.05).count();
    m.stats = Some(FitStats {
        n,
        outlier_frac: nlow as f64 / n as f64,
    });
    Ok(m)
}
