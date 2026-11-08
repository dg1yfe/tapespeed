//! Statistics with NumPy/SciPy definitions and NumPy NaN semantics.
//!
//! - Reductions (`np_sum`, `mean`, `std`) use numpy's pairwise summation (blocks of 128,
//!   8 accumulators) for stable rounding; contiguous 1-D results round like NumPy's.
//! - `median`/`percentile`/`max`/`min`/`ptp` return NaN if any input is NaN (numpy).
//! - `argmax`/`argmin` return the first occurrence; the first NaN wins (numpy). Never use
//!   `Iterator::max_by` for these: it returns the last maximum.
//! - [`argsort`] follows NumPy's default (`kind='quicksort'`) float64 introsort as run on
//!   aarch64 (no SIMD dispatch there), so equal keys come out in the same order.

use crate::{ValueError, verr};

// ---------------------------------------------------------------- summation ----------

const PW_BLOCKSIZE: usize = 128;

fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let body = n - n % 8;
        for c in a[8..body].chunks_exact(8) {
            for k in 0..8 {
                r[k] += c[k];
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[body..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum` of a contiguous float64 array (pairwise reduction, same rounding as NumPy).
pub fn np_sum(a: &[f64]) -> f64 {
    pairwise_sum(a)
}

/// `np.mean` (NaN for empty input, numpy warns).
pub fn mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// `np.std(a)` with ddof = 0: `sqrt(sum((a - mean)**2) / n)`.
pub fn std(a: &[f64]) -> f64 {
    let n = a.len() as f64;
    let m = np_sum(a) / n;
    let d: Vec<f64> = a
        .iter()
        .map(|&v| {
            let x = v - m;
            x * x
        })
        .collect();
    (np_sum(&d) / n).sqrt()
}

// ---------------------------------------------------------------- order statistics ---

/// numpy's sort order for floats: NaN sorts last (`a < b || (b != b && a == a)`).
#[inline]
fn np_less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// Sorted copy without NaNs, or `None` if `a` contains NaN.
fn sorted_finite(a: &[f64]) -> Option<Vec<f64>> {
    if a.iter().any(|v| v.is_nan()) {
        return None;
    }
    let mut v = a.to_vec();
    v.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
    Some(v)
}

/// `np.median`: NaN if empty or any NaN; even length → `(a + b) / 2` of the middle pair.
pub fn median(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 || a.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut v = a.to_vec();
    let h = n / 2;
    let (lo, m, _) = v.select_nth_unstable_by(h, |x, y| x.partial_cmp(y).unwrap());
    let m = *m;
    if n % 2 == 1 {
        m
    } else {
        let l = lo.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (l + m) / 2.0
    }
}

/// `np.nanmedian`: median of the non-NaN values; NaN if none.
pub fn nanmedian(a: &[f64]) -> f64 {
    let v: Vec<f64> = a.iter().copied().filter(|x| !x.is_nan()).collect();
    median(&v)
}

/// numpy `_lerp(a, b, t)` as used by `percentile(method='linear')`.
#[inline]
fn np_lerp(a: f64, b: f64, t: f64) -> f64 {
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

fn percentile_sorted(v: &[f64], q: f64) -> f64 {
    let n = v.len();
    let qq = q / 100.0;
    // numpy _QuantileMethods['linear']: virtual index = (n - 1) * q
    let vi = (n - 1) as f64 * qq;
    if vi.is_nan() {
        return f64::NAN;
    }
    let (prev, next, gamma);
    if vi >= (n - 1) as f64 {
        prev = n - 1;
        next = n - 1;
        gamma = 0.0;
    } else if vi < 0.0 {
        prev = 0;
        next = 0;
        gamma = vi; // irrelevant: a == b
    } else {
        let f = vi.floor();
        prev = f as usize;
        next = prev + 1;
        gamma = vi - f;
    }
    np_lerp(v[prev], v[next], gamma)
}

/// `np.percentile(a, q)` (method 'linear'); NaN if any NaN. Errors on empty input
/// (numpy raises IndexError) or q outside [0, 100].
pub fn percentile(a: &[f64], q: f64) -> Result<f64, ValueError> {
    Ok(percentiles(a, &[q])?[0])
}

/// `np.percentile(a, [q0, q1, ...])` (method 'linear').
pub fn percentiles(a: &[f64], qs: &[f64]) -> Result<Vec<f64>, ValueError> {
    if qs.iter().any(|&q| !(0.0..=100.0).contains(&q)) {
        return verr("Percentiles must be in the range [0, 100]");
    }
    if a.is_empty() {
        return verr("percentile of an empty array");
    }
    match sorted_finite(a) {
        None => Ok(vec![f64::NAN; qs.len()]),
        Some(v) => Ok(qs.iter().map(|&q| percentile_sorted(&v, q)).collect()),
    }
}

/// `np.max` (NaN propagates; `None` for empty input, numpy raises).
pub fn max(a: &[f64]) -> Option<f64> {
    argmax(a).map(|i| a[i])
}

/// `np.min` (NaN propagates; `None` for empty input).
pub fn min(a: &[f64]) -> Option<f64> {
    argmin(a).map(|i| a[i])
}

/// `np.ptp` = max − min (NaN propagates; `None` for empty input).
pub fn ptp(a: &[f64]) -> Option<f64> {
    Some(max(a)? - min(a)?)
}

/// `np.argmax`: first maximum; the first NaN wins. `None` for empty input (numpy raises).
pub fn argmax(a: &[f64]) -> Option<usize> {
    let mut it = a.iter().enumerate();
    let (mut bi, &first) = it.next()?;
    if first.is_nan() {
        return Some(0);
    }
    let mut bv = first;
    for (i, &v) in it {
        if v.is_nan() {
            return Some(i);
        }
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    Some(bi)
}

/// `np.argmin`: first minimum; the first NaN wins. `None` for empty input.
pub fn argmin(a: &[f64]) -> Option<usize> {
    let mut it = a.iter().enumerate();
    let (mut bi, &first) = it.next()?;
    if first.is_nan() {
        return Some(0);
    }
    let mut bv = first;
    for (i, &v) in it {
        if v.is_nan() {
            return Some(i);
        }
        if v < bv {
            bv = v;
            bi = i;
        }
    }
    Some(bi)
}

// ---------------------------------------------------------------- numpy argsort ------

const SMALL_QUICKSORT: usize = 15;

/// `np.argsort(v)` with the default `kind='quicksort'`: same algorithm as NumPy's `aquicksort_`
/// (median-of-3 introsort, insertion sort for partitions of <= 16 elements (`pr - pl <= 15`,
/// determined empirically against numpy 2.5 on aarch64), heapsort fallback at depth
/// `2*floor(log2(n))`). numpy dispatches to a SIMD argsort only on x86 (AVX-512); on
/// aarch64 this code path is what runs, so equal keys come out in numpy's order.
pub fn argsort(v: &[f64]) -> Vec<usize> {
    let num = v.len();
    let mut tosort: Vec<usize> = (0..num).collect();
    if num < 2 {
        return tosort;
    }
    let t = &mut tosort[..];
    let mut stack: Vec<(usize, usize, i32)> = Vec::new();
    let mut pl: usize = 0;
    let mut pr: usize = num - 1;
    let mut cdepth: i32 = (usize::BITS - 1 - num.leading_zeros()) as i32 * 2;
    loop {
        if cdepth < 0 {
            aheapsort(v, &mut t[pl..=pr]);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if np_less(v[t[pm]], v[t[pl]]) {
                    t.swap(pm, pl);
                }
                if np_less(v[t[pr]], v[t[pm]]) {
                    t.swap(pr, pm);
                }
                if np_less(v[t[pm]], v[t[pl]]) {
                    t.swap(pm, pl);
                }
                let vp = v[t[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(pm, pj);
                loop {
                    loop {
                        pi += 1;
                        if !np_less(v[t[pi]], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !np_less(vp, v[t[pj]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    t.swap(pi, pj);
                }
                let pk = pr - 1;
                t.swap(pi, pk);
                // push largest partition on stack
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
            // insertion sort
            for pi in pl + 1..=pr {
                let vi = t[pi];
                let vp = v[vi];
                let mut pj = pi;
                while pj > pl && np_less(vp, v[t[pj - 1]]) {
                    t[pj] = t[pj - 1];
                    pj -= 1;
                }
                t[pj] = vi;
            }
        }
        match stack.pop() {
            None => break,
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
        }
    }
    tosort
}

/// numpy `aheapsort_` (1-based heap over the index slice).
fn aheapsort(v: &[f64], tosort: &mut [usize]) {
    let n = tosort.len();
    // a[1..=n] maps to tosort[0..n]
    let get = |t: &[usize], i: usize| t[i - 1];
    let mut i = n / 2;
    while i > 0 {
        let tmp = get(tosort, i);
        let mut ii = i;
        let mut j = i << 1;
        while j <= n {
            if j < n && np_less(v[get(tosort, j)], v[get(tosort, j + 1)]) {
                j += 1;
            }
            if np_less(v[tmp], v[get(tosort, j)]) {
                tosort[ii - 1] = tosort[j - 1];
                ii = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[ii - 1] = tmp;
        i -= 1;
    }
    let mut nn = n;
    while nn > 1 {
        let tmp = tosort[nn - 1];
        tosort[nn - 1] = tosort[0];
        nn -= 1;
        let mut ii = 1;
        let mut j = 2;
        while j <= nn {
            if j < nn && np_less(v[get(tosort, j)], v[get(tosort, j + 1)]) {
                j += 1;
            }
            if np_less(v[tmp], v[get(tosort, j)]) {
                tosort[ii - 1] = tosort[j - 1];
                ii = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[ii - 1] = tmp;
    }
}

// ---------------------------------------------------------------- interp -------------

/// `np.interp(x, xp, fp)` with default `left=fp[0]`, `right=fp[-1]` (clamped). `xp` must be
/// increasing (numpy does not check). NaN `x` → NaN. Empty `xp` → error (numpy raises).
pub fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Result<Vec<f64>, ValueError> {
    if xp.is_empty() {
        return verr("array of sample points is empty");
    }
    if xp.len() != fp.len() {
        return verr("fp and xp are not of the same length.");
    }
    Ok(x.iter().map(|&v| interp_one(v, xp, fp)).collect())
}

/// Scalar `np.interp` (see [`interp`]; `xp` non-empty, same length as `fp`).
pub fn interp_one(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    interp_one_by(x, xp, |j| fp[j], &mut 0)
}

/// `np.interp` kernel with `fp` given by an accessor (e.g. a lazily evaluated linspace) and a
/// search hint for monotone query sequences (checks the hinted and the next interval before
/// bisecting; the result does not depend on the hint). Same arithmetic as numpy's
/// `arr_interp`: `slope*(x - xp[j]) + fp[j]`, retried from the right end on NaN.
#[inline]
pub fn interp_one_by<F: Fn(usize) -> f64>(x: f64, xp: &[f64], fp: F, hint: &mut usize) -> f64 {
    let n = xp.len();
    if n == 1 {
        return fp(0); // numpy: left/right default to fp[0] (also for NaN x)
    }
    if x.is_nan() {
        return x;
    }
    if x > xp[n - 1] {
        return fp(n - 1);
    }
    if x < xp[0] {
        return fp(0);
    }
    // last j with xp[j] <= x (numpy's bisection result)
    let h = *hint;
    let j = if h + 1 < n && xp[h] <= x && x < xp[h + 1] {
        h
    } else if h + 2 < n && xp[h + 1] <= x && x < xp[h + 2] {
        h + 1
    } else {
        xp.partition_point(|&v| x >= v) - 1
    };
    *hint = j;
    if j == n - 1 || xp[j] == x {
        return fp(j);
    }
    let (f0, f1) = (fp(j), fp(j + 1));
    let slope = (f1 - f0) / (xp[j + 1] - xp[j]);
    let mut r = slope.mul_add(x - xp[j], f0); // fused, as NumPy's arm64 build contracts it
    if r.is_nan() {
        r = slope.mul_add(x - xp[j + 1], f1);
        if r.is_nan() && f0 == f1 {
            r = f0;
        }
    }
    r
}

// ---------------------------------------------------------------- correlation --------

/// `np.corrcoef(x, y)[0, 1]`: `clip((c01 / s0) / s1, -1, 1)` with
/// `c = dot(Xc, Xc.T) * (1 / (N - 1))`. NaN propagates; constant input → NaN (0/0).
pub fn corrcoef(x: &[f64], y: &[f64]) -> Result<f64, ValueError> {
    if x.len() != y.len() {
        return verr("all the input array dimensions must match");
    }
    let n = x.len();
    let (mx, my) = (mean(x), mean(y));
    let xc: Vec<f64> = x.iter().map(|&v| v - mx).collect();
    let yc: Vec<f64> = y.iter().map(|&v| v - my).collect();
    let fact = 1.0 / (n as f64 - 1.0);
    let dot = |a: &[f64], b: &[f64]| {
        let p: Vec<f64> = a.iter().zip(b).map(|(u, v)| u * v).collect();
        np_sum(&p)
    };
    let c01 = dot(&xc, &yc) * fact;
    let s0 = (dot(&xc, &xc) * fact).sqrt();
    let s1 = (dot(&yc, &yc) * fact).sqrt();
    let r = (c01 / s0) / s1;
    Ok(if r.is_nan() { r } else { r.clamp(-1.0, 1.0) })
}

/// `scipy.stats.rankdata(a, method='average')` (1-based ranks, ties get the mean rank).
/// NaN input → all NaN (scipy ≥ 1.10 with nan_policy='propagate').
pub fn rankdata(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    if a.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; n];
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap()); // stable (mergesort)
    let mut r = vec![0.0; n];
    let mut s = 0;
    while s < n {
        let mut e = s + 1;
        while e < n && a[idx[e]] == a[idx[s]] {
            e += 1;
        }
        // ranks s+1 ..= e  → average 0.5 * (s + 1 + e)
        let avg = 0.5 * ((s + 1 + e) as f64);
        for &k in &idx[s..e] {
            r[k] = avg;
        }
        s = e;
    }
    r
}

/// `scipy.stats.spearmanr(a, b)[0]`: Pearson correlation of average ranks. NaN if
/// `n <= 1`, any NaN (nan_policy='propagate') or constant input (ConstantInputWarning).
pub fn spearmanr(a: &[f64], b: &[f64]) -> Result<f64, ValueError> {
    if a.len() != b.len() {
        return verr("all the input array dimensions must match");
    }
    if a.len() <= 1 || a.iter().chain(b).any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    corrcoef(&rankdata(a), &rankdata(b))
}

// ---------------------------------------------------------------- detrend / round ----

/// `scipy.signal.detrend(x, type='constant')` = `x - mean(x)`.
pub fn detrend_constant(x: &[f64]) -> Vec<f64> {
    let m = mean(x);
    x.iter().map(|&v| v - m).collect()
}

/// `scipy.signal.detrend(x, type='linear')`: subtract the least-squares line. scipy solves
/// `lstsq([t/N, 1], x)`; here the normal equations are solved in centred form (same
/// minimiser, rounding differs at ~1e-16 relative). Non-finite input → error (scipy's
/// `lstsq` raises "array must not contain infs or NaNs"). N = 1 → `[0]`.
pub fn detrend_linear(x: &[f64]) -> Result<Vec<f64>, ValueError> {
    if x.iter().any(|v| !v.is_finite()) {
        return verr("array must not contain infs or NaNs");
    }
    let n = x.len();
    if n <= 1 {
        return Ok(vec![0.0; n]);
    }
    let nf = n as f64;
    let tm = (nf - 1.0) / 2.0;
    let xm = mean(x);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let d = i as f64 - tm;
        sxy += d * (v - xm);
        sxx += d * d;
    }
    let slope = sxy / sxx;
    Ok(x.iter()
        .enumerate()
        .map(|(i, &v)| (v - xm) - slope * (i as f64 - tm))
        .collect())
}

// `np.round(x, decimals)`: use `ts_io::numfmt::np_round` (identical rint-half-even
// semantics). Not duplicated here to keep ts-dsp free of a ts-io dependency.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_first_and_nan() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[1.0, f64::NAN, 5.0, f64::NAN]), Some(1));
        assert_eq!(argmin(&[2.0, 1.0, 1.0]), Some(1));
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn median_basic() {
        assert_eq!(median(&[3.0, 1.0, 2.0, 10.0]), 2.5);
        assert!(median(&[1.0, f64::NAN]).is_nan());
        assert!(median(&[]).is_nan());
        assert_eq!(nanmedian(&[1.0, f64::NAN, 3.0]), 2.0);
        assert!(nanmedian(&[f64::NAN]).is_nan());
    }
}
