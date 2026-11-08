//! Dense and sparse linear algebra: `np.linalg.lstsq` (SVD, gelsd semantics),
//! `np.roots` / `np.poly` (companion eigenvalues) and a sparse SPD solve (faer supernodal
//! Cholesky, AMD ordering) for the model normal equations (`(N + R) x = b`).
//!
//! All faer calls run with `Par::Seq` (faer is built without its rayon feature): results do
//! not depend on the thread count.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, SymbolicCholesky, SymmetricOrdering, factorize_symbolic_cholesky,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, Mat, Par, Side, Spec};
use num_complex::Complex64;

use crate::{ValueError, verr};

// ---------------------------------------------------------------- lstsq --------------

/// Result of [`lstsq`]: minimum-norm solution, effective rank, singular values (descending).
#[derive(Debug, Clone)]
pub struct Lstsq {
    pub x: Vec<f64>,
    pub rank: usize,
    pub sv: Vec<f64>,
}

/// `np.linalg.lstsq(a, b, rcond)` for row-major `a` (`m × n`) and one right-hand side.
/// SVD solution as LAPACK `gelsd`: singular values `s_i <= rcond * s_max` are treated as zero,
/// `rcond = None` → `eps * max(m, n)` (numpy default); the minimum-norm solution is returned.
pub fn lstsq(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    rcond: Option<f64>,
) -> Result<Lstsq, ValueError> {
    if a.len() != m * n || b.len() != m {
        return verr("lstsq: incompatible dimensions");
    }
    if n == 0 {
        return Ok(Lstsq {
            x: vec![],
            rank: 0,
            sv: vec![],
        });
    }
    if m == 0 {
        return Ok(Lstsq {
            x: vec![0.0; n],
            rank: 0,
            sv: vec![],
        });
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return verr("SVD did not converge in Linear Least Squares");
    }
    let rcond = rcond.unwrap_or(f64::EPSILON * m.max(n) as f64);
    let am = Mat::<f64>::from_fn(m, n, |i, j| a[i * n + j]);
    let svd = am
        .thin_svd()
        .map_err(|_| ValueError("SVD did not converge in Linear Least Squares".into()))?;
    let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
    let k = m.min(n);
    let sv: Vec<f64> = (0..k).map(|i| s[i]).collect();
    let cut = rcond * sv[0];
    let mut x = vec![0.0; n];
    let mut rank = 0;
    for (i, &si) in sv.iter().enumerate() {
        if si <= cut || si == 0.0 {
            continue;
        }
        rank += 1;
        let mut c = 0.0;
        for (r, &bv) in b.iter().enumerate() {
            c += u[(r, i)] * bv;
        }
        let c = c / si;
        for (j, xv) in x.iter_mut().enumerate() {
            *xv += v[(j, i)] * c;
        }
    }
    Ok(Lstsq { x, rank, sv })
}

// ---------------------------------------------------------------- roots / poly -------

/// `np.roots(p)` (coefficients, highest power first): leading zeros stripped, trailing zeros
/// → zero roots appended, eigenvalues of the companion matrix
/// `A = diag(ones(N-2), -1); A[0, :] = -p[1:] / p[0]`. Eigenvalue order follows faer, not
/// LAPACK `geev` (callers use the set: `np.poly`, reflection, …).
pub fn roots(p: &[f64]) -> Result<Vec<Complex64>, ValueError> {
    let nz: Vec<usize> = (0..p.len()).filter(|&i| p[i] != 0.0).collect();
    let (Some(&first), Some(&last)) = (nz.first(), nz.last()) else {
        return Ok(vec![]);
    };
    if p.iter().any(|v| !v.is_finite()) {
        return verr("Array must not contain infs or NaNs");
    }
    let trailing = p.len() - last - 1;
    let q = &p[first..=last];
    let nn = q.len();
    let mut out = Vec::with_capacity(nn - 1 + trailing);
    if nn > 1 {
        let d = nn - 1;
        let a = Mat::<f64>::from_fn(d, d, |i, j| {
            if i == 0 {
                -q[j + 1] / q[0]
            } else if j + 1 == i {
                1.0
            } else {
                0.0
            }
        });
        let ev = a
            .eigenvalues()
            .map_err(|_| ValueError("Eigenvalues did not converge".into()))?;
        out.extend(ev.into_iter().map(|c| Complex64::new(c.re, c.im)));
    }
    out.extend(std::iter::repeat_n(Complex64::new(0.0, 0.0), trailing));
    Ok(out)
}

/// `np.poly(z)` for a 1-D sequence of roots: `a = [1]; a = convolve(a, [1, -z_k])` in order.
/// numpy returns the real part when the roots are closed under conjugation; callers take
/// `.re` (here the complex coefficients are always returned).
pub fn poly(z: &[Complex64]) -> Vec<Complex64> {
    let mut a = vec![Complex64::new(1.0, 0.0)];
    for &zk in z {
        let mz = -zk;
        let mut b = Vec::with_capacity(a.len() + 1);
        b.push(a[0]);
        for k in 1..a.len() {
            b.push(a[k] + a[k - 1] * mz);
        }
        b.push(a[a.len() - 1] * mz);
        a = b;
    }
    a
}

// ---------------------------------------------------------------- sparse SPD ---------

/// Symmetric sparse matrix stored as its lower triangle in CSC form (row indices sorted and
/// unique within each column, diagonal included). Indices are `u32` (memory: model L3 has up
/// to ~10⁷ entries).
#[derive(Debug, Clone)]
pub struct SymCsc {
    pub n: usize,
    pub col_ptr: Vec<u32>,
    pub row_idx: Vec<u32>,
    pub val: Vec<f64>,
}

impl SymCsc {
    /// Build the pattern from per-column sorted unique row lists (`rows[j]`, all `>= j`).
    pub fn from_columns(rows: Vec<Vec<u32>>) -> Self {
        let n = rows.len();
        let nnz: usize = rows.iter().map(Vec::len).sum();
        assert!(nnz < u32::MAX as usize, "SymCsc: too many entries");
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::with_capacity(nnz);
        col_ptr.push(0u32);
        for (j, r) in rows.into_iter().enumerate() {
            debug_assert!(r.windows(2).all(|w| w[0] < w[1]));
            debug_assert!(r.first().is_none_or(|&i| i as usize >= j));
            row_idx.extend_from_slice(&r);
            col_ptr.push(row_idx.len() as u32);
        }
        SymCsc {
            n,
            col_ptr,
            row_idx,
            val: vec![0.0; nnz],
        }
    }

    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }

    /// Position of entry `(i, j)` (`i >= j`) in `val`, if present.
    pub fn find(&self, i: usize, j: usize) -> Option<usize> {
        let (a, b) = (self.col_ptr[j] as usize, self.col_ptr[j + 1] as usize);
        self.row_idx[a..b]
            .binary_search(&(i as u32))
            .ok()
            .map(|p| a + p)
    }

    /// Diagonal values.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.n)
            .map(|j| self.find(j, j).map_or(0.0, |p| self.val[p]))
            .collect()
    }

    /// Add `v` to entry `(i, j)` (either triangle); panics if outside the pattern.
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        let p = self
            .find(i, j)
            .unwrap_or_else(|| panic!("SymCsc::add: ({i}, {j}) not in pattern"));
        self.val[p] += v;
    }

    fn symbolic(&self) -> SymbolicSparseColMatRef<'_, u32> {
        SymbolicSparseColMatRef::new_checked(self.n, self.n, &self.col_ptr, None, &self.row_idx)
    }
}

/// Sparse SPD solver: symbolic analysis (AMD ordering, supernodal/simplicial choice by faer)
/// done once per pattern, numeric `LLᵀ` per matrix. Falls back to `LDLᵀ` if the matrix is not
/// numerically positive definite.
pub struct SpdSolver {
    symbolic: SymbolicCholesky<u32>,
    n: usize,
    nnz: usize,
}

impl SpdSolver {
    /// Symbolic factorisation of `a`'s pattern.
    pub fn analyze(a: &SymCsc) -> Result<Self, ValueError> {
        let symbolic = factorize_symbolic_cholesky(
            a.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .map_err(|e| ValueError(format!("symbolic Cholesky: {e:?}")))?;
        Ok(SpdSolver {
            symbolic,
            n: a.n,
            nnz: a.nnz(),
        })
    }

    /// Symbolic factorisation without reordering: `a` must already be in a fill-reducing order
    /// (the caller permutes; this avoids faer's internal permuted copy of `A` at every
    /// numeric factorisation). The upper-triangular pattern faer analyses is built
    /// temporarily from `a`'s lower triangle.
    pub fn analyze_identity(a: &SymCsc) -> Result<Self, ValueError> {
        let n = a.n;
        let mut cnt = vec![0u32; n + 1];
        for &i in &a.row_idx {
            cnt[i as usize + 1] += 1;
        }
        for i in 0..n {
            cnt[i + 1] += cnt[i];
        }
        let mut next = cnt.clone();
        let mut ri = vec![0u32; a.nnz()];
        for j in 0..n {
            for &i in &a.row_idx[a.col_ptr[j] as usize..a.col_ptr[j + 1] as usize] {
                let p = &mut next[i as usize];
                ri[*p as usize] = j as u32;
                *p += 1;
            }
        }
        drop(next);
        let up = SymbolicSparseColMatRef::new_checked(n, n, &cnt, None, &ri);
        let symbolic = factorize_symbolic_cholesky(
            up,
            Side::Upper,
            SymmetricOrdering::Identity,
            CholeskySymbolicParams::default(),
        )
        .map_err(|e| ValueError(format!("symbolic Cholesky: {e:?}")))?;
        Ok(SpdSolver {
            symbolic,
            n,
            nnz: a.nnz(),
        })
    }

    /// Number of stored values of the factor `L`.
    pub fn factor_len(&self) -> usize {
        self.symbolic.len_val()
    }

    /// Solve `a x = b`; `a` must have the analysed pattern.
    pub fn solve(&self, a: &SymCsc, b: &[f64]) -> Result<Vec<f64>, ValueError> {
        if a.n != self.n || a.nnz() != self.nnz || b.len() != self.n {
            return verr("SpdSolver::solve: pattern / size mismatch");
        }
        let par = Par::Seq;
        let am = SparseColMatRef::new(a.symbolic(), &a.val);
        let req = self
            .symbolic
            .factorize_numeric_llt_scratch::<f64>(par, Spec::default())
            .or(self
                .symbolic
                .factorize_numeric_ldlt_scratch::<f64>(par, Spec::default()))
            .or(self.symbolic.solve_in_place_scratch::<f64>(1, par));
        let mut mem = MemBuffer::try_new(req).map_err(|_| ValueError("out of memory".into()))?;
        let stack = MemStack::new(&mut mem);
        let mut lv = vec![0.0f64; self.symbolic.len_val()];
        let mut x = Mat::<f64>::from_fn(self.n, 1, |i, _| b[i]);
        let llt = self.symbolic.factorize_numeric_llt::<f64>(
            &mut lv,
            am,
            Side::Lower,
            LltRegularization::default(),
            par,
            stack,
            Spec::default(),
        );
        match llt {
            Ok(llt) => llt.solve_in_place_with_conj(Conj::No, x.as_mut(), par, stack),
            Err(_) => {
                let ldlt = self
                    .symbolic
                    .factorize_numeric_ldlt::<f64>(
                        &mut lv,
                        am,
                        Side::Lower,
                        LdltRegularization::default(),
                        par,
                        stack,
                        Spec::default(),
                    )
                    .map_err(|e| ValueError(format!("sparse LDLT failed: {e:?}")))?;
                ldlt.solve_in_place_with_conj(Conj::No, x.as_mut(), par, stack);
            }
        }
        let out: Vec<f64> = (0..self.n).map(|i| x[(i, 0)]).collect();
        if out.iter().any(|v| !v.is_finite()) {
            return verr("sparse solve: non-finite solution (singular matrix)");
        }
        Ok(out)
    }
}

/// One-shot `spsolve(a, b)` for a symmetric positive definite `a`.
pub fn spd_solve(a: &SymCsc, b: &[f64]) -> Result<Vec<f64>, ValueError> {
    SpdSolver::analyze(a)?.solve(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lstsq_exact_and_min_norm() {
        let a = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let r = lstsq(&a, 3, 2, &[1.0, 2.0, 3.0], None).unwrap();
        assert!((r.x[0] - 1.0).abs() < 1e-14 && (r.x[1] - 2.0).abs() < 1e-14);
        assert_eq!(r.rank, 2);
        // rank deficient: the minimum-norm solution splits evenly
        let a = [1.0, 1.0, 2.0, 2.0];
        let r = lstsq(&a, 2, 2, &[2.0, 4.0], None).unwrap();
        assert_eq!(r.rank, 1);
        assert!((r.x[0] - 1.0).abs() < 1e-14 && (r.x[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn roots_poly_roundtrip() {
        let p = [0.0, 1.0, -6.0, 11.0, -6.0, 0.0];
        let mut z: Vec<f64> = roots(&p).unwrap().iter().map(|c| c.re).collect();
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in z.iter().zip([0.0, 1.0, 2.0, 3.0]) {
            assert!((a - b).abs() < 1e-12, "{z:?}");
        }
        let c = poly(&[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
        assert_eq!(c.iter().map(|v| v.re).collect::<Vec<_>>(), [1.0, -3.0, 2.0]);
    }

    #[test]
    fn spd_small() {
        let mut a = SymCsc::from_columns(vec![vec![0, 1], vec![1, 2], vec![2]]);
        a.add(0, 0, 4.0);
        a.add(1, 0, 1.0);
        a.add(1, 1, 3.0);
        a.add(2, 1, 1.0);
        a.add(2, 2, 2.0);
        let x = spd_solve(&a, &[1.0, 2.0, 3.0]).unwrap();
        let x2 = SpdSolver::analyze_identity(&a)
            .unwrap()
            .solve(&a, &[1.0, 2.0, 3.0])
            .unwrap();
        assert!(x.iter().zip(&x2).all(|(a, b)| (a - b).abs() < 1e-15));
        let r = [
            4.0 * x[0] + x[1] - 1.0,
            x[0] + 3.0 * x[1] + x[2] - 2.0,
            x[1] + 2.0 * x[2] - 3.0,
        ];
        assert!(r.iter().all(|v| v.abs() < 1e-14), "{r:?}");
    }
}
