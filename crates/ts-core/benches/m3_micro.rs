//! Micro benches (criterion) of the hot kernels on deterministic synthetic inputs generated
//! here (no external data): correct() on 10 s stereo, the heterodyne decimation on 60 s
//! stereo, the PLL on 1 M samples and fit_hier on a 720 s mechanical speed curve.
//! Run: `cargo bench -p ts-core --bench m3_micro [-- FILTER]`. "1t" = 1-thread rayon pool,
//! "mt" = default pool (all cores).
use criterion::{Criterion, criterion_group, criterion_main};
use num_complex::Complex64;
use std::f64::consts::TAU;
use std::hint::black_box;
use ts_core::analyze::{pll, stage1, track::BW_AUTO};
use ts_core::correct::{self, Params, Post};
use ts_core::hetdec::{Channel, het_decimate};
use ts_core::model::{DT, HierOpts, IdentOpts, explained, fit_hier};

const SR: f64 = 44100.0;
const REF_HZ: f64 = 9000.0;

/// Deterministic uniform noise in [-0.5, 0.5) (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Interleaved stereo f32: a 9 kHz line with 0.7 Hz wow, a 1 kHz programme tone and noise.
fn stereo(seconds: f64, seed: u64) -> Vec<f32> {
    let n = (seconds * SR) as usize;
    let mut rng = Lcg(seed);
    let (mut ph, mut x) = (0.0f64, Vec::with_capacity(2 * n));
    for i in 0..n {
        let t = i as f64 / SR;
        ph += TAU * REF_HZ * (1.0 + 0.002 * (TAU * 0.7 * t).sin()) / SR;
        let prog = 0.3 * (TAU * 1000.0 * t).sin();
        let v = 0.05 * ph.sin() + prog + 0.02 * rng.next();
        x.push(v as f32);
        x.push((0.9 * v + 0.01 * rng.next()) as f32);
    }
    x
}

/// Speed curve on the model grid: drift, 1.5 Hz wow and 7.5 Hz flutter around 1.0.
fn curve(seconds: f64) -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..=(seconds / DT).round() as usize)
        .map(|i| i as f64 * DT)
        .collect();
    let r = t
        .iter()
        .map(|&t| {
            1.0 + 0.001 * t / seconds
                + 0.002 * (TAU * 1.5 * t).sin()
                + 0.0005 * (TAU * 7.5 * t).sin()
        })
        .collect();
    (t, r)
}

/// Decimated complex baseband at `fs`: a slowly wandering carrier with additive noise.
fn baseband(n: usize, fs: f64) -> Vec<Complex64> {
    let mut rng = Lcg(7);
    let mut ph = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            ph += TAU * (2.0 + 15.0 * (TAU * 0.6 * t).sin() + 3.0 * (TAU * 7.46 * t).sin()) / fs;
            Complex64::from_polar(1.0, ph) + Complex64::new(0.2 * rng.next(), 0.2 * rng.next())
        })
        .collect()
}

/// 720 s mechanical speed curve: capstan (7.46 Hz, 2 harmonics), pinch roller (1.484 Hz),
/// take-up reel with falling rate (1.05 -> 0.6 Hz), slow drift, noise and a few invalid gaps.
fn mech(seconds: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(42);
    let n = (seconds / DT).round() as usize + 1;
    let (mut t, mut r, mut w) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut reel = 0.0f64;
    for i in 0..n {
        let x = i as f64 * DT;
        reel += TAU * (1.05 - 0.45 * x / seconds) * DT;
        let v = 1.0
            + 0.004 * (TAU * x / seconds).sin()
            + 0.003 * (TAU * 7.46 * x).sin()
            + 0.001 * (2.0 * TAU * 7.46 * x + 0.3).sin()
            + 0.0015 * (TAU * 1.484 * x + 1.1).sin()
            + 0.002 * reel.sin()
            + 2e-4 * rng.next();
        let gap = (x % 97.0) < 1.5;
        t.push(x);
        r.push(v);
        w.push(if gap { 0.0 } else { 1.0 });
    }
    (t, r, w)
}

fn benches(c: &mut Criterion) {
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let x10 = stereo(10.0, 1);
    let x60 = stereo(60.0, 2);
    let (d1, fs_d, _, h) = stage1::design(SR, REF_HZ, 0.03, &BW_AUTO).unwrap();
    let (ct, cr) = curve(10.0);
    let zp = baseband(1_000_000, fs_d);
    let (mt, mr, mw) = mech(720.0);
    let ho = HierOpts {
        ta: 5.0,
        ident: IdentOpts::for_duration(mt[mt.len() - 1]),
        ..HierOpts::default()
    };

    let mut g = c.benchmark_group("m3");
    g.sample_size(10);
    g.bench_function("pll_1M_1t", |b| {
        b.iter(|| black_box(pll::pll(black_box(&zp), fs_d, 45.0, 60.0)))
    });
    for (tag, single) in [("1t", true), ("mt", false)] {
        let run = |f: &mut (dyn FnMut() + Send)| {
            if single { p1.install(f) } else { f() }
        };
        g.bench_function(format!("correct_10s_stereo_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    let mut acc = 0f32;
                    correct::run_arrays(
                        black_box(&x10),
                        2,
                        SR as u32,
                        &ct,
                        &cr,
                        &Params::default(),
                        &Post::default(),
                        |y: &[f32]| {
                            acc += y[0];
                            Ok::<(), ()>(())
                        },
                    )
                    .unwrap();
                    black_box(acc);
                })
            })
        });
        g.bench_function(format!("het_decimate_60s_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    black_box(
                        het_decimate(black_box(&x60), 2, SR, REF_HZ, d1, &h, Channel::Mid).unwrap(),
                    );
                })
            })
        });
        g.bench_function(format!("fit_hier_mech720_{tag}"), |b| {
            b.iter(|| {
                run(&mut || {
                    let hh = fit_hier(&mt, &mr, &mw, &ho).unwrap();
                    let vv: Vec<f64> = mw
                        .iter()
                        .zip(&hh.rejected)
                        .map(|(&a, &j)| if a > 0.0 && !j { 1.0 } else { 0.0 })
                        .collect();
                    black_box(explained(
                        &[("m1", &hh.m1), ("m2", &hh.m2), ("m3", &hh.m3)],
                        &mt,
                        &mr,
                        &vv,
                    ));
                })
            })
        });
    }
    g.finish();
}

criterion_group!(m3, benches);
criterion_main!(m3);
