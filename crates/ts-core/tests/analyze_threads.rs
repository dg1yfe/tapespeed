//! Analyze stages 1-5: thread-count determinism on a synthetic signal and bit-exact scalar
//! helpers against stored reference results (tests/data/e).
#![allow(clippy::needless_range_loop)]
mod e_common;
#[path = "../../testdata.rs"]
mod testdata;
use e_common::*;

use ts_core::analyze::{coarse, pll, stage1, track, track::BW_AUTO};
use ts_core::hetdec::Channel;

/// Stages 1-5 on a synthetic 30 s stereo signal: 1 thread vs the default pool, bit-identical
/// (results must not depend on the thread count).
#[test]
fn threads_bit_identical() {
    use ts_core::analyze::track::TrackOut;
    let (sr, n) = (44100.0f64, 30 * 44100usize);
    let mut x = vec![0f32; 2 * n];
    let mut ph = 0.0f64;
    let mut s = 12345u64;
    for i in 0..n {
        let t = i as f64 / sr;
        ph += std::f64::consts::TAU
            * 9000.0
            * (1.0 + 0.002 * (std::f64::consts::TAU * 0.7 * t).sin())
            / sr;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let nz = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02;
        let v = 0.05 * ph.sin() + nz;
        x[2 * i] = v as f32;
        x[2 * i + 1] = (v * 0.9) as f32;
    }
    let run = || -> (Vec<u64>, Vec<TrackOut>) {
        let s1 = stage1::stage1(&x, 2, sr, 9000.0, 0.03, &BW_AUTO, Channel::Mid).unwrap();
        let c = coarse::coarse(
            &s1.z,
            s1.fs_d,
            0.03 * 9000.0,
            9000.0,
            coarse::CoarseParams::default(),
        )
        .unwrap();
        let fcd = coarse::fcd(s1.z.len(), s1.fs_d, &c.tc, &c.f).unwrap();
        use rayon::prelude::*;
        let tr: Vec<TrackOut> = BW_AUTO
            .par_iter()
            .map(|&b| track::track(&s1.z, s1.fs_d, &fcd, 9000.0, b, 25.0).unwrap())
            .collect();
        let mut bits: Vec<u64> =
            s1.z.iter()
                .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
                .collect();
        bits.extend(c.f.iter().map(|v| v.to_bits()));
        (bits, tr)
    };
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(run);
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(run);
    assert_eq!(one.0, many.0, "stage 1/2 bits");
    for (a, b) in one.1.iter().zip(&many.1) {
        let fb = |t: &TrackOut| -> Vec<u64> {
            [
                &t.f, &t.ff, &t.fb, &t.lk_f, &t.lk_b, &t.snr_db, &t.conf, &t.cn0, &t.agree,
            ]
            .iter()
            .flat_map(|v| v.iter().map(|x| x.to_bits()))
            .collect()
        };
        assert_eq!(fb(a), fb(b), "track bw {} bits", a.bw);
        assert_eq!(
            (&a.valid, &a.soft, &a.slip, &a.dip, &a.intf),
            (&b.valid, &b.soft, &b.slip, &b.dip, &b.intf)
        );
        assert!(
            a.valid.iter().any(|&v| v),
            "bw {}: synthetic tone should be tracked",
            a.bw
        );
    }
}

fn data_e(name: &str) -> std::path::PathBuf {
    testdata::dir("e").join(name)
}

/// Correctly rounded `hypot` and the complex `abs` / multiply conventions of the analyze
/// stages, bit for bit against stored reference results.
#[test]
fn scalar_ops_bitwise() {
    let (x, y, h) = (
        ld_f64(&data_e("hypot_x.npy")),
        ld_f64(&data_e("hypot_y.npy")),
        ld_f64(&data_e("hypot_out.npy")),
    );
    let bad = (0..x.len())
        .filter(|&i| pll::hypot_accurate(x[i], y[i]).to_bits() != h[i].to_bits())
        .count();
    assert_eq!(bad, 0, "hypot mismatches");
    let (a, b) = (
        ld_c128(&data_e("cplx_a.npy")),
        ld_c128(&data_e("cplx_b.npy")),
    );
    let abs = ld_f64(&data_e("cplx_abs.npy"));
    let (mul, mulc) = (
        ld_c128(&data_e("cplx_mul.npy")),
        ld_c128(&data_e("cplx_mulconj.npy")),
    );
    for i in 0..a.len() {
        assert_eq!(coarse::np_cabs(a[i]).to_bits(), abs[i].to_bits(), "abs {i}");
        assert_eq!(track::np_cmul(a[i], b[i]), mul[i], "mul {i}");
        assert_eq!(track::np_cmul(a[i], b[i].conj()), mulc[i], "mulconj {i}");
    }
}
