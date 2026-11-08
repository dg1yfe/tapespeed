//! Determinism rule: every parallel primitive gives bit-identical results on a
//! 1-thread pool and a many-thread pool (`RAYON_NUM_THREADS=1` vs default).

use num_complex::Complex64;
use ts_dsp::conv::{self, ConvMode};
use ts_dsp::spectral;

fn sig(n: usize, seed: u64) -> Vec<f64> {
    // xorshift; deterministic, no rand dependency
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn csig(n: usize, seed: u64) -> Vec<Complex64> {
    let (a, b) = (sig(n, seed), sig(n, seed + 1));
    a.into_iter()
        .zip(b)
        .map(|(r, i)| Complex64::new(r, i))
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn cbits(v: &[Complex64]) -> Vec<u64> {
    v.iter()
        .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
        .collect()
}

fn on_pools<T: PartialEq + std::fmt::Debug + Send>(f: impl Fn() -> T + Sync) {
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let nmax = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .max(4);
    let pn = rayon::ThreadPoolBuilder::new()
        .num_threads(nmax)
        .build()
        .unwrap();
    let a = p1.install(&f);
    let b = pn.install(&f);
    let c = f(); // global (default) pool
    assert!(a == b && b == c, "results differ between thread counts");
}

#[test]
fn conv_is_thread_invariant() {
    let x = sig(200_000, 1);
    let h = sig(300, 2);
    let hs = sig(31, 3);
    let xc = csig(150_000, 4);
    for mode in [ConvMode::Full, ConvMode::Same, ConvMode::Valid] {
        on_pools(|| bits(&conv::fftconvolve(&x, &h, mode)));
        on_pools(|| bits(&conv::fftconvolve(&x, &hs, mode)));
        on_pools(|| cbits(&conv::oaconvolve_c(&xc, &h, mode)));
    }
    let m = sig(2049 * 40, 5);
    on_pools(|| {
        bits(
            &conv::fftconvolve_axis0(&m, 2049, 40, &hs, ConvMode::Same)
                .unwrap()
                .0,
        )
    });
    on_pools(|| bits(&conv::upfirdn(&h, &x, 1, 3).unwrap()));
}

#[test]
fn resample_is_thread_invariant() {
    let xc = csig(300_000, 6);
    let x = sig(300_000, 7);
    let h = sig(731, 8);
    on_pools(|| cbits(&conv::resample_poly_down_c(&xc, 7, &h).unwrap()));
    on_pools(|| bits(&conv::resample_poly_down(&x, 10, &h).unwrap()));
}

#[test]
fn spectral_is_thread_invariant() {
    let x = sig(400_000, 9);
    on_pools(|| bits(&spectral::welch(&x, 100.0, 1024).unwrap().1));
    on_pools(|| bits(&spectral::welch(&x, 100.0, 4096).unwrap().1));
    on_pools(|| cbits(&spectral::stft(&x, 44100.0, 512, None).unwrap().z));
    on_pools(|| {
        cbits(
            &spectral::stft_frame_major(&x, 44100.0, 4096, Some(2048))
                .unwrap()
                .z,
        )
    });
    let w = ts_dsp::window::hanning(1000);
    on_pools(|| cbits(&spectral::rfft_frames(&x, &w, 333, 2048).unwrap().0));
}
