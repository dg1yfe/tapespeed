//! Stage 1 of `analyze()`: heterodyne at `ref_hz` and linear-phase FIR decimation to `fs_d`
//! (covers ±search plus the widest loop band). Kernel: [`crate::hetdec::het_decimate`].

use num_complex::Complex64;
use ts_dsp::ValueError;

use crate::hetdec::{Channel, HetStream, het_decimate, lpf};

/// Stage-1 result: baseband `z` at `fs_d = sr / d1`, `z[k] <-> x[k*d1]`, and the FIR used.
#[derive(Debug, Clone)]
pub struct Stage1 {
    pub z: Vec<Complex64>,
    pub fs_d: f64,
    pub d1: usize,
    /// `lpf(fh1, fs_d - fh1, sr)`
    pub h: Vec<f64>,
    /// stage-1 half bandwidth [Hz]
    pub fh1: f64,
}

/// Design part of stage 1 (everything before the heterodyne decimation):
/// `W = 4·max(bws) + 0.003·ref`, `fh1 = search·ref·1.1 + W`, `D1 = max(1, int(sr / (2.5·fh1)))`.
/// Returns `(d1, fs_d, fh1, h)`.
pub fn design(
    sr: f64,
    ref_hz: f64,
    search: f64,
    bws: &[f64],
) -> Result<(usize, f64, f64, Vec<f64>), ValueError> {
    // max: first of equal values; NaN-free inputs
    let bmax = bws
        .iter()
        .copied()
        .reduce(|a, b| if b > a { b } else { a })
        .ok_or_else(|| ValueError("max() arg is an empty sequence".into()))?;
    let w = 4.0 * bmax + 0.003 * ref_hz;
    let fh1 = search * ref_hz * 1.1 + w;
    let d1 = ((sr / (2.5 * fh1)) as usize).max(1);
    let fs_d = sr / d1 as f64;
    let h = lpf(fh1, fs_d - fh1, sr, 90.0)?;
    Ok((d1, fs_d, fh1, h))
}

/// Stage 1 of `analyze(x, sr, ref_hz, search, bw, channel)`; `x` interleaved f32 frames with
/// `ch` channels, `bws` the candidate loop bandwidths (`BW_AUTO` or `[bw]`).
pub fn stage1(
    x: &[f32],
    ch: usize,
    sr: f64,
    ref_hz: f64,
    search: f64,
    bws: &[f64],
    channel: Channel,
) -> Result<Stage1, ValueError> {
    let (d1, fs_d, fh1, h) = design(sr, ref_hz, search, bws)?;
    let z = het_decimate(x, ch, sr, ref_hz, d1, &h, channel)?;
    Ok(Stage1 {
        z,
        fs_d,
        d1,
        h,
        fh1,
    })
}

/// [`stage1`] fed chunk by chunk (e.g. straight from the decoder): bit-identical to
/// `stage1` on the concatenated input, for any chunking.
pub struct Stage1Stream {
    st: HetStream,
    fs_d: f64,
    d1: usize,
    h: Vec<f64>,
    fh1: f64,
}

impl Stage1Stream {
    /// Same arguments as [`stage1`] minus the signal.
    pub fn new(
        ch: usize,
        sr: f64,
        ref_hz: f64,
        search: f64,
        bws: &[f64],
        channel: Channel,
    ) -> Result<Self, ValueError> {
        let (d1, fs_d, fh1, h) = design(sr, ref_hz, search, bws)?;
        let st = HetStream::new(ch, sr, ref_hz, d1, &h, channel)?;
        Ok(Stage1Stream {
            st,
            fs_d,
            d1,
            h,
            fh1,
        })
    }

    /// Concurrency cap of each emission ([`HetStream::set_par`]).
    pub fn set_par(&mut self, n: Option<usize>) {
        self.st.set_par(n);
    }

    /// Reserve the baseband for about `frames` input frames.
    pub fn reserve_frames(&mut self, frames: usize) {
        self.st.reserve_frames(frames);
    }

    /// Append interleaved frames.
    pub fn push(&mut self, frames: &[f32]) {
        self.st.push(frames);
    }

    pub fn finish(self) -> Stage1 {
        Stage1 {
            z: self.st.finish(),
            fs_d: self.fs_d,
            d1: self.d1,
            h: self.h,
            fh1: self.fh1,
        }
    }
}
