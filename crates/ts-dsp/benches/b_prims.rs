//! Benches: welch / stft / resample_poly at call-site sizes. Each runs on a 1-thread
//! rayon pool ("1t") and the default pool ("mt").

use criterion::{Criterion, criterion_group, criterion_main};
use num_complex::Complex64;
use std::hint::black_box;
use ts_dsp::{conv, spectral};

fn sig(n: usize, seed: u64) -> Vec<f64> {
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

fn benches(c: &mut Criterion) {
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let x_welch = sig(1 << 19, 1);
    let x_stft = sig(60 * 44100, 2);
    let xc: Vec<Complex64> = sig(1 << 21, 3)
        .into_iter()
        .zip(sig(1 << 21, 4))
        .map(|(a, b)| Complex64::new(a, b))
        .collect();
    let h731 = sig(731, 5);
    let x_real = sig(1 << 20, 6);
    let h361 = sig(361, 7);
    let x_conv = sig(1 << 20, 8);

    let mut g = c.benchmark_group("b");
    g.sample_size(10);
    let x_med = sig(1 << 20, 9);
    g.bench_function("median_filter_1M_w3267", |b| {
        b.iter(|| {
            black_box(
                ts_dsp::ndimage::median_filter(
                    black_box(&x_med),
                    3267,
                    ts_dsp::ndimage::Mode::Nearest,
                )
                .unwrap(),
            )
        })
    });
    g.bench_function("median_filter_1M_w31", |b| {
        b.iter(|| {
            black_box(
                ts_dsp::ndimage::median_filter(
                    black_box(&x_med),
                    31,
                    ts_dsp::ndimage::Mode::Nearest,
                )
                .unwrap(),
            )
        })
    });
    for (tag, single) in [("1t", true), ("mt", false)] {
        let run = |f: &mut (dyn FnMut() + Send)| {
            if single { p1.install(f) } else { f() }
        };
        g.bench_function(format!("welch_512k_nper4096_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    black_box(spectral::welch(black_box(&x_welch), 200.0, 4096).unwrap());
                })
            })
        });
        g.bench_function(format!("stft_60s_44k1_nper4096_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    black_box(
                        spectral::stft(black_box(&x_stft), 44100.0, 4096, Some(2048)).unwrap(),
                    );
                })
            })
        });
        g.bench_function(format!("stft_frame_major_60s_44k1_nper4096_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    black_box(
                        spectral::stft_frame_major(black_box(&x_stft), 44100.0, 4096, Some(2048))
                            .unwrap(),
                    );
                })
            })
        });
        g.bench_function(format!("resample_poly_c_2M_D16_h731_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    black_box(conv::resample_poly_down_c(black_box(&xc), 16, &h731).unwrap());
                })
            })
        });
        g.bench_function(format!("resample_poly_1M_D18_h361_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    black_box(conv::resample_poly_down(black_box(&x_real), 18, &h361).unwrap());
                })
            })
        });
        g.bench_function(format!("oaconvolve_valid_1M_h731_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    black_box(conv::oaconvolve(
                        black_box(&x_conv),
                        &h731,
                        conv::ConvMode::Valid,
                    ));
                })
            })
        });
    }
    g.finish();
}

criterion_group!(b, benches);
criterion_main!(b);
