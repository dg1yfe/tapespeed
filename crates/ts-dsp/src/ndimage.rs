//! `scipy.ndimage` 1-D filters and binary morphology, `scipy.signal.medfilt`.
//!
//! Window placement (origin 0) for a size-`s` filter at output `i`:
//! `[i - s/2, i + (s-1)/2]` (even sizes lean left). Binary dilation reflects the structure,
//! so its window is `[i - (s-1)/2, i + s/2]` (identical for odd `s`).
//!
//! NaN inputs to the median/rank filters are unspecified (scipy's own result depends on its
//! selection order); no call site feeds NaN.

use rayon::prelude::*;

use crate::{ValueError, verr};

/// Boundary mode (`mode=` of scipy.ndimage).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// `'nearest'`: a a a | a b c d | d d d
    Nearest,
    /// `'reflect'` (half-sample symmetric, scipy default): c b a | a b c d | d c b
    Reflect,
    /// `'constant'` with `cval`.
    Constant(f64),
}

/// Value of `x` at (possibly out-of-range) index `j` under `mode` (ndimage `NI_ExtendLine`
/// semantics, including extensions longer than the line).
#[inline]
fn ext_at(x: &[f64], j: isize, mode: Mode) -> f64 {
    let n = x.len() as isize;
    if (0..n).contains(&j) {
        return x[j as usize];
    }
    match mode {
        Mode::Constant(c) => c,
        Mode::Nearest => x[j.clamp(0, n - 1) as usize],
        Mode::Reflect => {
            let p = 2 * n;
            let k = j.rem_euclid(p);
            x[if k < n { k } else { p - 1 - k } as usize]
        }
    }
}

/// Line buffer: `before` extended samples, the line, `after` extended samples.
fn extend(x: &[f64], before: usize, after: usize, mode: Mode) -> Vec<f64> {
    let n = x.len() as isize;
    let mut v = Vec::with_capacity(x.len() + before + after);
    v.extend((-(before as isize)..0).map(|j| ext_at(x, j, mode)));
    v.extend_from_slice(x);
    v.extend((n..n + after as isize).map(|j| ext_at(x, j, mode)));
    v
}

/// `scipy.ndimage.uniform_filter1d(x, size, mode=mode)`. Same running-sum recurrence as
/// scipy 1.18 (`acc += in[i+s-1] - in[i-1]; out[i] = acc / s`), so results are
/// bit-identical (verified against the generator vectors).
pub fn uniform_filter1d(x: &[f64], size: usize, mode: Mode) -> Result<Vec<f64>, ValueError> {
    if size < 1 {
        return verr("incorrect filter size");
    }
    let n = x.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let s1 = size / 2;
    let s2 = size - s1 - 1;
    let buf = extend(x, s1, s2, mode);
    let fs = size as f64;
    let mut out = Vec::with_capacity(n);
    let mut tmp = 0.0;
    for &v in &buf[..size] {
        tmp += v;
    }
    out.push(tmp / fs);
    for i in 1..n {
        tmp += buf[i + size - 1] - buf[i - 1];
        out.push(tmp / fs);
    }
    Ok(out)
}

const RANK_CHUNK: usize = 1 << 16;
const SORTED_WINDOW_MAX: usize = 384;

/// `scipy.ndimage.gaussian_filter1d(x, sigma, mode=mode)` (order 0, truncate 4.0):
/// radius `int(4 sigma + 0.5)`, weights `exp(-0.5/sigma² · k²) / sum` (numpy pairwise
/// sum), correlated with ndimage's symmetric-kernel loop
/// (`out = in[0]·w[0] + Σ_{j=r..1} (in[-j] + in[j])·w[j]`, outermost pair first).
pub fn gaussian_filter1d(x: &[f64], sigma: f64, mode: Mode) -> Result<Vec<f64>, ValueError> {
    if sigma.is_nan() || sigma <= 0.0 {
        return verr("sigma must be positive");
    }
    let n = x.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let r = (4.0 * sigma + 0.5) as usize;
    let s2 = sigma * sigma;
    let phi: Vec<f64> = (-(r as i64)..=r as i64)
        .map(|k| (-0.5 / s2 * (k * k) as f64).exp())
        .collect();
    let tot = crate::stats::np_sum(&phi);
    let w: Vec<f64> = phi.iter().map(|v| v / tot).collect(); // w[r] is the centre
    let buf = extend(x, r, r, mode);
    Ok((0..n)
        .map(|i| {
            let c = i + r;
            let mut acc = buf[c] * w[r];
            for j in (1..=r).rev() {
                acc += (buf[c - j] + buf[c + j]) * w[r - j];
            }
            acc
        })
        .collect())
}

/// Rank filter over the extended buffer: `out[i] = sorted(buf[i..i+w])[rank]`.
/// Order statistics involve no arithmetic, so the fixed output chunks (rayon) give
/// bit-identical results for any thread count.
fn rank_filter_ext(buf: &[f64], w: usize, rank: usize, n: usize) -> Vec<f64> {
    let chunk = RANK_CHUNK.max(4 * w);
    let mut out = vec![0.0; n];
    out.par_chunks_mut(chunk).enumerate().for_each(|(c, dst)| {
        let c0 = c * chunk;
        let seg = &buf[c0..c0 + dst.len() + w - 1];
        if w <= SORTED_WINDOW_MAX {
            rank_filter_sorted(seg, w, rank, dst);
        } else {
            rank_filter_bit(seg, w, rank, dst);
        }
    });
    out
}

/// Small/medium windows: sorted window, binary-search delete + insert (one memmove).
fn rank_filter_sorted(seg: &[f64], w: usize, rank: usize, dst: &mut [f64]) {
    let mut win: Vec<f64> = seg[..w].to_vec();
    win.sort_unstable_by(f64::total_cmp);
    dst[0] = win[rank];
    for i in 1..dst.len() {
        let old = seg[i - 1];
        let new = seg[i + w - 1];
        let p = win
            .binary_search_by(|v| v.total_cmp(&old))
            .expect("outgoing sample is in the window");
        let q = win.partition_point(|v| v.total_cmp(&new).is_lt());
        if q <= p {
            win.copy_within(q..p, q + 1);
            win[q] = new;
        } else {
            win.copy_within(p + 1..q, p);
            win[q - 1] = new;
        }
        dst[i] = win[rank];
    }
}

/// Order-preserving map f64 → u64 (IEEE total order, as `f64::total_cmp`).
#[inline]
fn total_key(v: f64) -> u64 {
    let b = v.to_bits();
    if b >> 63 == 1 { !b } else { b | (1 << 63) }
}

/// Large windows: Fenwick tree over the segment's value ranks, O(len log len).
fn rank_filter_bit(seg: &[f64], w: usize, rank: usize, dst: &mut [f64]) {
    let nn = seg.len();
    let mut keyed: Vec<(u64, u32)> = seg
        .iter()
        .enumerate()
        .map(|(i, &v)| (total_key(v), i as u32))
        .collect();
    keyed.sort_unstable();
    let mut rk = vec![0u32; nn];
    for (r, &(_, j)) in keyed.iter().enumerate() {
        rk[j as usize] = r as u32;
    }
    let mut tree = vec![0i32; nn + 1];
    let add = |tree: &mut [i32], r: u32, d: i32| {
        let mut i = r as usize + 1;
        while i <= nn {
            tree[i] += d;
            i += i & i.wrapping_neg();
        }
    };
    let top = 1usize << (usize::BITS - 1 - nn.leading_zeros());
    // smallest position p with prefix(p) >= k (k is 1-based)
    let kth = |tree: &[i32], mut k: i32| -> usize {
        let mut pos = 0usize;
        let mut step = top;
        while step > 0 {
            let nx = pos + step;
            if nx <= nn && tree[nx] < k {
                pos = nx;
                k -= tree[nx];
            }
            step >>= 1;
        }
        pos
    };
    for &r in &rk[..w] {
        add(&mut tree, r, 1);
    }
    for i in 0..dst.len() {
        if i > 0 {
            add(&mut tree, rk[i - 1], -1);
            add(&mut tree, rk[i + w - 1], 1);
        }
        let p = kth(&tree, rank as i32 + 1);
        dst[i] = seg[keyed[p].1 as usize];
    }
}

/// `scipy.ndimage.median_filter(x, size, mode=mode)` (1-D; rank `size // 2`, so even sizes
/// take the upper middle element, as scipy).
pub fn median_filter(x: &[f64], size: usize, mode: Mode) -> Result<Vec<f64>, ValueError> {
    if size < 1 {
        return verr("incorrect filter size");
    }
    let n = x.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let s1 = size / 2;
    let buf = extend(x, s1, size - s1 - 1, mode);
    Ok(rank_filter_ext(&buf, size, size / 2, n))
}

/// `scipy.signal.medfilt(x, kernel_size)` (1-D): zero-padded rank filter, odd sizes only.
pub fn medfilt(x: &[f64], kernel_size: usize) -> Result<Vec<f64>, ValueError> {
    if kernel_size % 2 != 1 {
        return verr("Each element of kernel_size should be odd.");
    }
    median_filter(x, kernel_size, Mode::Constant(0.0))
}

// ---------------------------------------------------------------- binary morphology --

fn prefix_counts(x: &[bool]) -> Vec<u32> {
    let mut c = Vec::with_capacity(x.len() + 1);
    c.push(0u32);
    let mut s = 0u32;
    for &b in x {
        s += b as u32;
        c.push(s);
    }
    c
}

fn check_struct(n: usize) -> Result<(), ValueError> {
    if n == 0 {
        return verr("structure must not be empty");
    }
    Ok(())
}

/// `ndimage.binary_erosion(x, np.ones(n, bool), border_value=0)`:
/// `out[i] = all(x[i - n/2 ..= i + (n-1)/2])`, outside the array counts as False.
pub fn binary_erosion(x: &[bool], n: usize) -> Result<Vec<bool>, ValueError> {
    check_struct(n)?;
    let len = x.len() as isize;
    let c = prefix_counts(x);
    let (l, r) = ((n / 2) as isize, ((n - 1) / 2) as isize);
    Ok((0..len)
        .map(|i| {
            let (a, b) = (i - l, i + r);
            a >= 0 && b < len && (c[(b + 1) as usize] - c[a as usize]) as usize == n
        })
        .collect())
}

/// `ndimage.binary_dilation(x, np.ones(n, bool))` (border_value 0):
/// `out[i] = any(x[i - (n-1)/2 ..= i + n/2])` (structure reflected by scipy).
pub fn binary_dilation(x: &[bool], n: usize) -> Result<Vec<bool>, ValueError> {
    check_struct(n)?;
    let len = x.len() as isize;
    let c = prefix_counts(x);
    let (l, r) = (((n - 1) / 2) as isize, (n / 2) as isize);
    Ok((0..len)
        .map(|i| {
            let a = (i - l).max(0) as usize;
            let b = (i + r).min(len - 1) as usize;
            c[b + 1] > c[a]
        })
        .collect())
}

/// `ndimage.binary_opening(x, np.ones(n, bool))` = dilation(erosion(x)).
pub fn binary_opening(x: &[bool], n: usize) -> Result<Vec<bool>, ValueError> {
    binary_dilation(&binary_erosion(x, n)?, n)
}

/// `ndimage.binary_closing(x, np.ones(n, bool))` = erosion(dilation(x)), border_value 0.
pub fn binary_closing(x: &[bool], n: usize) -> Result<Vec<bool>, ValueError> {
    binary_erosion(&binary_dilation(x, n)?, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_equals_direct() {
        let x: Vec<f64> = (0..500).map(|i| ((i * 7919) % 101) as f64 * 0.37).collect();
        for &w in &[49usize, 50, 64, 101, 700] {
            let buf = extend(&x, w / 2, w - w / 2 - 1, Mode::Nearest);
            let mut a = vec![0.0; x.len()];
            rank_filter_bit(&buf, w, w / 2, &mut a);
            let mut a2 = vec![0.0; x.len()];
            rank_filter_sorted(&buf, w, w / 2, &mut a2);
            assert_eq!(a, a2, "w={w}");
            let b: Vec<f64> = (0..x.len())
                .map(|i| {
                    let mut t = buf[i..i + w].to_vec();
                    t.sort_by(f64::total_cmp);
                    t[w / 2]
                })
                .collect();
            assert_eq!(a, b, "w={w}");
        }
    }

    #[test]
    fn chunk_boundaries() {
        let n = 3 * RANK_CHUNK + 123;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 2_654_435_761usize) % 1009) as f64)
            .collect();
        for &w in &[5usize, 31, 1001] {
            let got = median_filter(&x, w, Mode::Nearest).unwrap();
            let buf = extend(&x, w / 2, w - w / 2 - 1, Mode::Nearest);
            for &i in &[0, 1, RANK_CHUNK - 1, RANK_CHUNK, 2 * RANK_CHUNK + 7, n - 1] {
                let mut t = buf[i..i + w].to_vec();
                t.sort_by(f64::total_cmp);
                assert_eq!(got[i], t[w / 2], "w={w} i={i}");
            }
        }
    }

    #[test]
    fn reflect_long_extension() {
        let x = [1.0, 2.0, 3.0];
        let e = extend(&x, 7, 7, Mode::Reflect);
        let want = [
            1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0,
        ];
        assert_eq!(e, want);
    }
}
