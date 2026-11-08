//! Window shapes: scipy `hann` (periodic, used by welch/stft), numpy
//! `np.hanning` (symmetric, used for frame spectra in survey/model/analyze) and scipy
//! `kaiser`. Formulas follow the SciPy/NumPy definitions operation by operation.
//!
//! `i0` here is the crate's canonical `np.i0` (re-exported as `fir::i0`).

pub use crate::ValueError;

/// `numpy.linspace(start, stop, num)` (endpoint=True) with numpy's formula:
/// `arange(num) * step + start`, last element set to `stop`.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => vec![],
        1 => vec![start],
        _ => {
            let step = (stop - start) / (num - 1) as f64;
            let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
            v[num - 1] = stop;
            v
        }
    }
}

/// `numpy.arange(start, stop, step)` for floats: `len = ceil((stop - start) / step)`,
/// values `start + i * delta` with `delta = (start + step) - start` (numpy's fill rule).
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil();
    if len.is_nan() || len <= 0.0 {
        return vec![];
    }
    let n = len as usize;
    let delta = (start + step) - start;
    let mut v = Vec::with_capacity(n);
    v.push(start);
    for i in 1..n {
        v.push(start + i as f64 * delta);
    }
    v
}

/// scipy `general_cosine(M, [0.5, 0.5], sym)`: `fac = linspace(-pi, pi, M)`,
/// `w = 0 + 0.5*cos(0*fac) + 0.5*cos(1*fac)`.
fn general_hann(m: usize, sym: bool) -> Vec<f64> {
    if m <= 1 {
        return vec![1.0; m];
    }
    let mm = if sym { m } else { m + 1 };
    let fac = linspace(-std::f64::consts::PI, std::f64::consts::PI, mm);
    let mut w: Vec<f64> = fac
        .iter()
        .map(|&f| (0.0 + 0.5 * (0.0 * f).cos()) + 0.5 * (1.0 * f).cos())
        .collect();
    if !sym {
        w.pop();
    }
    w
}

/// `scipy.signal.get_window('hann', n)` = `hann(n, sym=False)` (periodic; welch/stft default).
pub fn hann_periodic(n: usize) -> Vec<f64> {
    general_hann(n, false)
}

/// `scipy.signal.windows.hann(n, sym=True)`.
pub fn hann_symmetric(n: usize) -> Vec<f64> {
    general_hann(n, true)
}

/// `numpy.hanning(M)`: `0.5 + 0.5*cos(pi*k/(M-1))`, `k = arange(1-M, M, 2)`; M = 1 → [1].
pub fn hanning(m: usize) -> Vec<f64> {
    match m {
        0 => vec![],
        1 => vec![1.0],
        _ => {
            let den = (m - 1) as f64;
            (0..m)
                .map(|i| {
                    let k = (2 * i as i64 + 1 - m as i64) as f64;
                    0.5 + 0.5 * (std::f64::consts::PI * k / den).cos()
                })
                .collect()
        }
    }
}

/// `scipy.signal.windows.kaiser(M, beta, sym)`:
/// `w = i0(beta*sqrt(1 - ((n - alpha)/alpha)**2)) / i0(beta)`, `alpha = (M-1)/2`.
pub fn kaiser(m: usize, beta: f64, sym: bool) -> Vec<f64> {
    if m <= 1 {
        return vec![1.0; m];
    }
    let mm = if sym { m } else { m + 1 };
    let alpha = (mm as f64 - 1.0) / 2.0;
    let den = i0(beta);
    let mut w: Vec<f64> = (0..mm)
        .map(|n| {
            let r = (n as f64 - alpha) / alpha;
            i0(beta * (1.0 - r * r).sqrt()) / den
        })
        .collect();
    if !sym {
        w.pop();
    }
    w
}

// ---- Cephes i0 (Chebyshev expansion), coefficients verbatim from numpy `_i0A/_i0B` ------

#[allow(clippy::excessive_precision)]
const I0_A: [f64; 30] = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1,
];

#[allow(clippy::excessive_precision)]
const I0_B: [f64; 25] = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1,
];

fn chbevl(x: f64, coef: &[f64]) -> f64 {
    let mut b0 = coef[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &c in &coef[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    0.5 * (b0 - b2)
}

/// Modified Bessel function of the first kind, order 0: `np.i0` (Cephes Chebyshev expansion,
/// numpy's coefficients and op order). Canonical implementation; `fir::i0` re-exports it.
pub fn i0(x: f64) -> f64 {
    let x = x.abs();
    if x <= 8.0 {
        x.exp() * chbevl(x / 2.0 - 2.0, &I0_A)
    } else {
        x.exp() * chbevl(32.0 / x - 2.0, &I0_B) / x.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sizes() {
        assert!(hann_periodic(0).is_empty());
        assert_eq!(hann_periodic(1), vec![1.0]);
        assert_eq!(hanning(1), vec![1.0]);
        assert_eq!(hann_periodic(2), vec![0.0, 1.0]);
        assert_eq!(kaiser(1, 5.0, true), vec![1.0]);
    }

    #[test]
    fn i0_values() {
        assert!((i0(0.0) - 1.0).abs() < 1e-16);
        assert!((i0(1.0) / 1.266_065_877_752_008_4 - 1.0).abs() < 1e-15);
        assert!((i0(10.0) / 2_815.716_628_466_254 - 1.0).abs() < 1e-14);
    }
}
