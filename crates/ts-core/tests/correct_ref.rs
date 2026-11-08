//! correct() against stored reference results (tests/data/c):
//! RMS diff <= -120 dB re signal, max abs <= 4e-6; streaming == whole-array <= 1e-7;
//! 1 thread == N threads bit-identical.
mod common;
use common::*;
use ts_core::correct::{self, Lowpass, Params, Post};

const CASES: [&str; 6] = ["ident", "drift", "slow", "wow", "three", "chain"];

struct Case {
    x: Vec<f32>,
    ch: usize,
    sr: u32,
    t: Vec<f64>,
    r: Vec<f64>,
}

fn case(name: &str) -> Case {
    let (shape, x) = ld_f32(name, "x");
    Case {
        x,
        ch: shape[1],
        sr: ld(name, "sr").1[0] as u32,
        t: ld(name, "t").1,
        r: ld(name, "r").1,
    }
}

fn chain_post() -> Post {
    Post {
        notches: vec![(19000.0, 30.0), (3361.0, 3361.0 / 150.0)],
        lowpass: Some(Lowpass::new(8000.0)),
    }
}

#[test]
fn time_map_ref() {
    for name in CASES {
        let c = case(name);
        let (tm, rs) =
            correct::time_map(&c.t, &c.r, (c.x.len() / c.ch) as f64 / c.sr as f64).unwrap();
        let te = ld(name, "tm_x").1;
        for (want, got) in [(ld(name, "tm_T").1, &tm), (ld(name, "tm_r").1, &rs)] {
            let s = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let e = te
                .iter()
                .zip(&want)
                .map(|(&t, w)| (got.eval(t) - w).abs())
                .fold(0.0, f64::max)
                / s;
            assert!(e < 1e-12, "{name}: time map rel {e:e}");
        }
    }
}

#[test]
fn correct_ref() {
    println!("case     frames ch   RMS diff dB   max abs");
    for name in CASES {
        let c = case(name);
        let y = correct::correct(&c.x, c.ch, c.sr, &c.t, &c.r, &Params::default()).unwrap();
        let (_, yp) = ld_f32(name, "y");
        let (db, mx) = cmp(&y, &yp);
        println!(
            "{name:8} {:6} {:2}   {db:9.1}    {mx:.2e}",
            y.len() / c.ch,
            c.ch
        );
        assert!(db <= -120.0 && mx <= 4e-6, "{name}: {db} dB, {mx}");
    }
}

#[test]
fn chain_ref_and_streaming() {
    let c = case("chain");
    let p = Params::default();
    let post = chain_post();
    let mut y = correct::correct(&c.x, c.ch, c.sr, &c.t, &c.r, &p).unwrap();
    let mut stages = vec![];
    correct::apply_notch(&mut y, c.ch, c.sr, 19000.0, 30.0).unwrap();
    stages.push(("notch", y.clone()));
    correct::apply_notch(&mut y, c.ch, c.sr, 3361.0, 3361.0 / 150.0).unwrap();
    stages.push(("notch2", y.clone()));
    correct::apply_lowpass(&mut y, c.ch, c.sr, &Lowpass::new(8000.0)).unwrap();
    stages.push(("lp", y.clone()));
    for (s, v) in &stages {
        let (db, mx) = cmp(v, &ld_f32("chain", &format!("y_{s}")).1);
        println!("chain {s:7} RMS diff {db:.1} dB, max abs {mx:.2e}");
        assert!(db <= -120.0 && mx <= 4e-6, "{s}");
    }
    let whole = correct::correct_post(&c.x, c.ch, c.sr, &c.t, &c.r, &p, &post).unwrap();
    assert_eq!(whole, y);
    let mut st = Vec::new();
    let n = correct::run_arrays(&c.x, c.ch, c.sr, &c.t, &c.r, &p, &post, |b: &[f32]| {
        st.extend_from_slice(b);
        Ok::<(), ()>(())
    })
    .unwrap();
    assert_eq!(n * c.ch, whole.len());
    let mx = max_abs(&st, &whole);
    println!("streaming vs whole-array: max abs {mx:.2e}");
    assert!(mx <= 1e-7);
}

#[test]
fn streaming_long_multi_block() {
    // several streaming blocks + notch lookahead emissions; compare with the whole-array path
    let sr = 44100u32;
    let n = 3 * correct::CHUNK * correct::BLOCK_CHUNKS / 2 + 12345;
    let ch = 2;
    let x: Vec<f32> = (0..n * ch)
        .map(|i| {
            let k = (i / ch) as f64;
            (0.3 * (k * 0.0123).sin() + 0.2 * (k * 0.91 + (i % ch) as f64).sin()) as f32
        })
        .collect();
    let dur = n as f64 / sr as f64;
    let t: Vec<f64> = (0..(dur / 0.005) as usize + 1)
        .map(|i| i as f64 * 0.005)
        .collect();
    let r: Vec<f64> = t.iter().map(|&v| 1.0 + 0.01 * (v * 2.0).sin()).collect();
    let p = Params::default();
    let post = chain_post();
    let whole = correct::correct_post(&x, ch, sr, &t, &r, &p, &post).unwrap();
    let mut st = Vec::new();
    let mut nb = 0;
    correct::run_arrays(&x, ch, sr, &t, &r, &p, &post, |b: &[f32]| {
        st.extend_from_slice(b);
        nb += 1;
        Ok::<(), ()>(())
    })
    .unwrap();
    let mx = max_abs(&st, &whole);
    println!("long streaming ({nb} sink calls) vs whole: max abs {mx:.2e}");
    assert!(mx <= 1e-7);
    // without post-processing the stream is the resampler output itself, bit for bit
    let mut st = Vec::new();
    correct::run_arrays(&x, ch, sr, &t, &r, &p, &Post::default(), |b: &[f32]| {
        st.extend_from_slice(b);
        Ok::<(), ()>(())
    })
    .unwrap();
    assert_eq!(st, correct::correct(&x, ch, sr, &t, &r, &p).unwrap());
}

#[test]
fn one_thread_equals_many() {
    let c = case("wow");
    let p = Params::default();
    let post = chain_post();
    let go = || {
        let mut st = Vec::new();
        correct::run_arrays(&c.x, c.ch, c.sr, &c.t, &c.r, &p, &post, |b: &[f32]| {
            st.extend_from_slice(b);
            Ok::<(), ()>(())
        })
        .unwrap();
        (
            st,
            correct::correct_post(&c.x, c.ch, c.sr, &c.t, &c.r, &p, &post).unwrap(),
        )
    };
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(go);
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(go);
    assert!(
        one.0 == many.0 && one.1 == many.1,
        "thread count changed the output"
    );
}

#[test]
fn errors_on_short_input() {
    // notch needs > 9 output samples (zero-phase filtering pads 9 samples per side)
    let x = vec![0.0f32; 8];
    let post = Post {
        notches: vec![(1000.0, 30.0)],
        lowpass: None,
    };
    let r = correct::run_arrays(
        &x,
        1,
        48000,
        &[0.0, 1.0],
        &[1.0, 1.0],
        &Params::default(),
        &post,
        |_: &[f32]| Ok::<(), ()>(()),
    );
    assert!(matches!(r, Err(correct::RunError::Value(e)) if e.0.contains("padlen, which is 9")));
    assert!(correct::correct(&x, 1, 48000, &[0.0, 0.0], &[1.0, 1.0], &Params::default()).is_err());
}
