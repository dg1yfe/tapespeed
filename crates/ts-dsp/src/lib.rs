//! DSP primitives with NumPy/SciPy-compatible definitions (same conventions and edge cases,
//! same operation order where it matters for rounding).
//!
//! Canonical shared helpers (one implementation each): [`ValueError`] (crate error, also
//! re-exported as `window::ValueError`, `fir::ValueError`), `stats::np_sum` (numpy pairwise
//! sum, re-exported as `fir::np_sum`), `window::i0` / `window::kaiser` (re-exported /
//! wrapped in `fir`), `stats::interp_one_by` (np.interp kernel).

use std::fmt;

/// Invalid-argument error, raised under the same conditions as the SciPy/NumPy `ValueError` of
/// the corresponding definition (own message text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueError(pub String);

/// Crate-level error type.
pub type Error = ValueError;

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValueError: {}", self.0)
    }
}

impl std::error::Error for ValueError {}

pub(crate) fn verr<T>(msg: impl Into<String>) -> Result<T, ValueError> {
    Err(ValueError(msg.into()))
}

pub mod bspline;
pub mod conv;
pub mod fir;
pub mod iir;
pub mod interp;
pub mod linalg;
pub mod ndimage;
pub mod optimize;
pub mod peaks;
pub mod spectral;
pub mod spline;
pub mod stats;
pub mod window;
