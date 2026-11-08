//! correct() behaviour on synthetic signals (identity, warped tone, length, short curve,
//! anti-alias, notch, lowpass). Prints each measured value next to the stored reference value
//! (tests/data/c) and asserts fixed bounds and <= reference + 0.5 dB.
mod common;
use common::ld;
use std::f64::consts::PI;
use ts_core::correct::{self, Lowpass, Params};

const SR: u32 = 48000;
const EDGE: usize = (0.1 * SR as f64) as usize;
const DUR: f64 = 2.0;

fn reference(key: &str) -> f64 {
    let keys: String = ld("ref_tests", "keys")
        .1
        .iter()
        .map(|&c| c as u8 as char)
        .collect();
    let i = keys
        .split('\n')
        .position(|k| k == key)
        .unwrap_or_else(|| panic!("no {key}"));
    ld("ref_tests", "vals").1[i]
}

fn show(name: &str, got: f64, key: &str) -> f64 {
    let p = reference(key);
    println!("{name:28} got {got:10.4}   reference {p:10.4}");
    p
}

fn db(err: &[f64], refr: &[f64]) -> f64 {
    let m = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
    10.0 * (m(err) / m(refr)).log10()
}

/// amplitude of a tone at f (least-squares projection on sin/cos)
fn amp(y: &[f64], f: f64) -> f64 {
    let (mut ss, mut sc, mut cc, mut ys, mut yc) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (k, &v) in y.iter().enumerate() {
        let ph = 2.0 * PI * f * k as f64 / SR as f64;
        let (s, c) = ph.sin_cos();
        ss += s * s;
        sc += s * c;
        cc += c * c;
        ys += v * s;
        yc += v * c;
    }
    let det = ss * cc - sc * sc;
    let a = (ys * cc - yc * sc) / det;
    let b = (yc * ss - ys * sc) / det;
    a.hypot(b)
}

fn grid(end: f64) -> Vec<f64> {
    (0..=((end + 1e-9) / 0.005) as usize)
        .map(|i| i as f64 * 0.005)
        .collect()
}
fn r_true(t: f64) -> f64 {
    1.015 + 0.003 * (2.0 * PI * 1.3 * t).sin() + 0.004 * t / DUR
}
fn t_true(t: f64) -> f64 {
    1.015 * t + 0.003 * (1.0 - (2.0 * PI * 1.3 * t).cos()) / (2.0 * PI * 1.3) + 0.002 * t * t / DUR
}
fn run1(x: &[f32], t: &[f64], r: &[f64]) -> Vec<f64> {
    correct::correct(x, 1, SR, t, r, &Params::default())
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect()
}
fn ideal_tone(f: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| (2.0 * PI * f * k as f64 / SR as f64).sin())
        .collect()
}
fn resid(y: &[f64], ideal: &[f64]) -> f64 {
    let n = y.len();
    let e: Vec<f64> = (EDGE..n - EDGE).map(|i| y[i] - ideal[i]).collect();
    db(&e, &ideal[EDGE..n - EDGE])
}

#[test]
fn test_identity() {
    let n = (DUR * SR as f64) as usize;
    let fs = [100.0, 1000.0, 5000.0, 10000.0, 15000.0, 18000.0];
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            fs.iter()
                .map(|&f| (2.0 * PI * f * t + f).sin())
                .sum::<f64>()
                / fs.len() as f64
        })
        .collect();
    let mut xi = vec![0.0f32; 2 * n];
    for i in 0..n {
        xi[2 * i] = x[i] as f32;
        xi[2 * i + 1] = (0.5 * x[n - 1 - i]) as f32;
    }
    let tg = grid(DUR);
    let y = correct::correct(&xi, 2, SR, &tg, &vec![1.0; tg.len()], &Params::default()).unwrap();
    assert_eq!(y.len(), xi.len());
    let a = EDGE * 2..(n - EDGE) * 2;
    let e: Vec<f64> = a.clone().map(|i| y[i] as f64 - xi[i] as f64).collect();
    let r: Vec<f64> = a.map(|i| xi[i] as f64).collect();
    let d = db(&e, &r);
    let p = show("identity residual dB", d, "identity");
    assert!(d < -90.0 && d <= p + 0.5);
}

#[test]
fn test_warped_tone() {
    let n = (DUR * SR as f64) as usize;
    let tg = grid(DUR);
    let rg: Vec<f64> = tg.iter().map(|&t| r_true(t)).collect();
    for f in [1000.0, 15000.0] {
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f * t_true(i as f64 / SR as f64)).sin() as f32)
            .collect();
        let y = run1(&x, &tg, &rg);
        let d = resid(&y, &ideal_tone(f, y.len()));
        let p = show(&format!("warped tone {f} Hz dB"), d, &format!("warped{f}"));
        assert!(d < -80.0 && d <= p + 0.5);
    }
}

#[test]
fn test_length() {
    let tg = grid(DUR);
    let rg: Vec<f64> = tg.iter().map(|&t| r_true(t)).collect();
    for n in [(DUR * SR as f64) as usize, (DUR * SR as f64) as usize - 777] {
        let y = correct::correct(&vec![0.0; 2 * n], 2, SR, &tg, &rg, &Params::default()).unwrap();
        let len = (y.len() / 2) as f64;
        let p = show(&format!("length n={n}"), len, &format!("len{n}"));
        assert_eq!(len, p);
        assert!((len - t_true(n as f64 / SR as f64) * SR as f64).abs() <= 1.0);
    }
}

#[test]
fn test_short_curve_holds_end() {
    let n = (DUR * SR as f64) as usize;
    let tg = grid(1.0);
    let rg = vec![1.02; tg.len()];
    let (tm, _) = correct::time_map(&tg, &rg, DUR).unwrap();
    let tv = tm.eval(DUR);
    show("T(DUR)", tv, "short_hold_T");
    assert!((tv - 1.02 * DUR).abs() < 1e-9);
    let x: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 1000.0 * 1.02 * i as f64 / SR as f64).sin() as f32)
        .collect();
    let y = run1(&x, &tg, &rg);
    let d = resid(&y, &ideal_tone(1000.0, y.len()));
    let p = show("short curve residual dB", d, "short_hold");
    assert!(d < -80.0 && d <= p + 0.5);
}

#[test]
fn test_antialias_sign() {
    let n = SR as usize;
    let tg = grid(1.0);
    let x: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 23000.0 * i as f64 / SR as f64).sin() as f32)
        .collect();
    let y = run1(&x, &tg, &vec![0.9; tg.len()]);
    let xs: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let d = db(&y[EDGE..y.len() - EDGE], &xs);
    let p = show("r=0.9, 23 kHz leak dB", d, "alias_db");
    assert!(d < -80.0 && d <= p + 0.5);
    let x: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 17000.0 * i as f64 / SR as f64).sin() as f32)
        .collect();
    let y = run1(&x, &tg, &vec![1.1; tg.len()]);
    let a = amp(&y[EDGE..y.len() - EDGE], 17000.0 / 1.1);
    show("r=1.1, 15.45 kHz amplitude", a, "expand_amp");
    assert!((a - 1.0).abs() < 1e-3);
}

#[test]
fn test_notch() {
    let rf = 19000.0;
    let n = SR as usize;
    let fs = [0.9 * rf, rf, 1.1 * rf];
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            (fs.iter().map(|&f| (2.0 * PI * f * t).sin()).sum::<f64>() / 3.0) as f32
        })
        .collect();
    let tg = grid(1.0);
    let ones = vec![1.0; tg.len()];
    let y0 = run1(&x, &tg, &ones);
    let mut y1 = correct::correct(&x, 1, SR, &tg, &ones, &Params::default()).unwrap();
    correct::apply_notch(&mut y1, 1, SR, rf, 30.0).unwrap();
    let y1: Vec<f64> = y1.iter().map(|&v| v as f64).collect();
    let s = EDGE..n - EDGE;
    let g: Vec<f64> = fs
        .iter()
        .map(|&f| 20.0 * (amp(&y1[s.clone()], f) / amp(&y0[s.clone()], f)).log10())
        .collect();
    for i in 0..3 {
        show(
            &format!("notch gain dB at {:.1} ref", [0.9, 1.0, 1.1][i]),
            g[i],
            &format!("notch{i}"),
        );
    }
    assert!(g[1] < -40.0);
    assert!(g[0].abs() < 0.5 && g[2].abs() < 0.5);
}

#[test]
fn test_lowpass_response_and_alignment() {
    let sr = 44100u32;
    let n = 2 * sr as usize;
    let mut g = std::collections::HashMap::new();
    for f in [1000.0, 7700.0, 8300.0, 8934.0] {
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / sr as f64).sin() as f32)
            .collect();
        let mut y = x.clone();
        correct::apply_lowpass(&mut y, 1, sr, &Lowpass::new(8000.0)).unwrap();
        let m = sr as usize / 2..n - sr as usize / 2;
        let std = |v: &[f32]| {
            let mu = v.iter().map(|&a| a as f64).sum::<f64>() / v.len() as f64;
            (v.iter().map(|&a| (a as f64 - mu).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let gd = 20.0 * (std(&y[m.clone()]) / std(&x[m.clone()])).log10();
        show(&format!("lowpass gain dB at {f} Hz"), gd, &format!("lp{f}"));
        if f == 1000.0 {
            let dev = m
                .clone()
                .map(|i| (y[i] - x[i]).abs() as f64)
                .fold(0.0, f64::max);
            show("lowpass 1 kHz max |y-x|", dev, "lp_maxdev");
            assert!(dev < 1e-3);
        }
        g.insert(f as i64, gd);
    }
    assert!(g[&7700].abs() < 0.01 && g[&8300] < -75.0 && g[&8934] < -75.0);
}
