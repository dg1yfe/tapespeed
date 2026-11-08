//! Evidence (fading, band_stats, bandwidth_policy, lowpass_curve) and interference (detect,
//! keying, notch_spans) against stored reference results (tests/data/h); behaviour tests on
//! synthetic signals, streaming == whole-array, 1 thread == N threads.
//!
//! Acceptance: detections equal (hz <= 0.01 Hz, spans exact, snr <= 0.05 dB), keying <= 1e-6,
//! gated notch <= -120 dB RMS diff, evidence dicts 1e-6 relative.
#[path = "../../testdata.rs"]
mod testdata;
use std::path::PathBuf;

use ts_core::curve::Curve;
use ts_core::evidence::{self as ev, BandStats, Channel, CorrLimit, FadingParams};
use ts_core::interf::{self as itf, DetectParams, Detection, Gate};

// ------------------------------------------------------------------ data helpers

fn dir() -> PathBuf {
    testdata::dir("h")
}

fn npy(p: PathBuf) -> (Vec<usize>, npyz::NpyFile<std::io::BufReader<std::fs::File>>) {
    let f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let n = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    (n.shape().iter().map(|&s| s as usize).collect(), n)
}

fn ld_in(d: PathBuf, case: &str, name: &str) -> (Vec<usize>, Vec<f64>) {
    let (shape, n) = npy(d.join(format!("{case}__{name}.npy")));
    let descr = n.dtype().descr();
    let v = if descr.contains("f4") {
        n.into_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    } else if descr.contains("i8") {
        n.into_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as f64)
            .collect()
    } else if descr.contains("b1") {
        n.into_vec::<bool>()
            .unwrap()
            .into_iter()
            .map(|v| v as u8 as f64)
            .collect()
    } else {
        n.into_vec::<f64>().unwrap()
    };
    (shape, v)
}

fn ld(case: &str, name: &str) -> Vec<f64> {
    ld_in(dir(), case, name).1
}

fn ldb(case: &str, name: &str) -> Vec<bool> {
    ld(case, name).into_iter().map(|v| v != 0.0).collect()
}

fn ld_f32_in(d: PathBuf, case: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, n) = npy(d.join(format!("{case}__{name}.npy")));
    (shape, n.into_vec::<f32>().unwrap())
}

fn txt(case: &str, name: &str) -> Vec<String> {
    let p = dir().join(format!("{case}__{name}.txt"));
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .lines()
        .map(String::from)
        .collect()
}

fn pf(s: &str) -> f64 {
    s.parse().unwrap_or_else(|_| panic!("float {s:?}"))
}

fn spans_of(s: &str) -> Vec<(f64, f64)> {
    s.split(';')
        .map(|p| {
            let (a, b) = p.split_once(',').unwrap();
            (pf(a), pf(b))
        })
        .collect()
}

struct RefDet {
    d: Detection,
    contrast: f64,
    mid: f64,
    keyed: bool,
}

fn ref_dets(case: &str, name: &str) -> Vec<RefDet> {
    txt(case, name)
        .iter()
        .map(|l| {
            let w: Vec<&str> = l.split(' ').collect();
            RefDet {
                d: Detection {
                    hz: pf(w[0]),
                    total_s: pf(w[1]),
                    snr_db: pf(w[2]),
                    spans: spans_of(w[3]),
                },
                contrast: pf(w[4]),
                mid: pf(w[5]),
                keyed: w[6] == "1",
            }
        })
        .collect()
}

/// Max differences (hz, snr, keying) after asserting count / spans / total_s equal.
fn cmp_dets(tag: &str, rs: &[Detection], keys: &[(f64, f64)], want: &[RefDet]) -> (f64, f64, f64) {
    assert_eq!(rs.len(), want.len(), "{tag}: detection count");
    let (mut dh, mut ds, mut dk) = (0.0f64, 0.0f64, 0.0f64);
    for ((r, k), p) in rs.iter().zip(keys).zip(want) {
        assert_eq!(r.spans, p.d.spans, "{tag}: spans of {}", p.d.hz);
        assert_eq!(r.total_s, p.d.total_s, "{tag}: total_s of {}", p.d.hz);
        dh = dh.max((r.hz - p.d.hz).abs());
        ds = ds.max((r.snr_db - p.d.snr_db).abs());
        dk = dk.max((k.0 - p.contrast).abs()).max((k.1 - p.mid).abs());
        assert_eq!(itf::is_keyed(k.0, k.1), p.keyed, "{tag}: keyed {}", p.d.hz);
    }
    println!(
        "{tag}: {} lines, max |dhz| {dh:.3e} Hz, |dsnr| {ds:.3e} dB, |dkey| {dk:.3e}",
        rs.len()
    );
    (dh, ds, dk)
}

fn rms_db(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut e2, mut r2, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (&p, &q) in a.iter().zip(b) {
        let d = p as f64 - q as f64;
        e2 += d * d;
        r2 += (q as f64).powi(2);
        mx = mx.max(d.abs());
    }
    (10.0 * (e2.max(1e-300) / r2).log10(), mx)
}

fn relerr(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    if a == b {
        return 0.0;
    }
    (a - b).abs() / b.abs().max(1e-300)
}

fn keyed_y() -> (Vec<f32>, u32) {
    let y: Vec<f32> = ld("keyed", "y").into_iter().map(|v| v as f32).collect();
    (y, ld("keyed", "sr")[0] as u32)
}

#[allow(clippy::too_many_arguments)]
fn stream_gated(
    y: &[f32],
    ch: usize,
    sr: u32,
    hz: f64,
    q: f64,
    sp: &[(f64, f64)],
    g: &Gate,
    blk: usize,
) -> Vec<f32> {
    let mut st = itf::GatedNotch::new(ch, sr, y.len() / ch, hz, q, sp, g).unwrap();
    let mut out = Vec::new();
    for b in y.chunks(blk * ch) {
        out.extend_from_slice(st.push(b).unwrap());
    }
    out.extend_from_slice(st.finish().unwrap());
    out
}

// ------------------------------------------------------------------ interf

#[test]
fn keyed_detect_keying_ref() {
    let (y, sr) = keyed_y();
    let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let p = DetectParams {
        fmin: 1000.0,
        ..Default::default()
    };
    let d = itf::detect(&y64, sr, &p).unwrap();
    let k: Vec<(f64, f64)> = d
        .iter()
        .map(|d| itf::keying(&y64, sr, d.hz, &d.spans).unwrap())
        .collect();
    let (dh, ds, dk) = cmp_dets("keyed f64", &d, &k, &ref_dets("keyed", "det"));
    assert!(dh <= 0.01 && ds <= 0.05 && dk <= 1e-6);
    // pipeline input (float32): the reference spectra are single precision (complex64) -> looser
    let (dh, ds, dk) = cmp_dets("keyed f32-pipeline", &d, &k, &ref_dets("keyed", "det_f32"));
    assert!(dh <= 0.01 && ds <= 0.05 && dk <= 1e-3);

    // streaming == whole array, bit for bit
    let mut det = itf::Detector::new(sr, &p).unwrap();
    for b in y64.chunks(100_003) {
        det.push(b.iter().copied()).unwrap();
    }
    assert_eq!(det.finish().unwrap(), d);
    let lines: Vec<(f64, Vec<(f64, f64)>)> = d.iter().map(|d| (d.hz, d.spans.clone())).collect();
    let mut ks = itf::KeyingStream::new(sr, &lines);
    for b in y64.chunks(77_777) {
        ks.push(b).unwrap();
    }
    assert_eq!(ks.finish().unwrap(), k);
}

/// A keyed line is detected and the gated notch removes it.
#[test]
fn test_detect_and_gated_notch() {
    let (y, sr) = keyed_y();
    let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let d = itf::detect(
        &y64,
        sr,
        &DetectParams {
            fmin: 1000.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(d.len() == 1 && (d[0].hz - 6000.0).abs() < 1.0);
    assert_eq!(d[0].spans.len(), 1);
    let (t0, t1) = d[0].spans[0];
    assert!((10.0..=16.0).contains(&t0) && (49.0..=56.0).contains(&t1));
    let z = itf::notch_spans(
        &y,
        1,
        sr,
        d[0].hz,
        d[0].hz / 150.0,
        Some(&d[0].spans),
        &Gate::default(),
    )
    .unwrap();
    let line_db = |x: &[f32], a: f64, b: f64| {
        let seg: Vec<f64> = x[(a * sr as f64) as usize..(b * sr as f64) as usize]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let (f, p) = ts_dsp::spectral::welch(&seg, sr as f64, 4096).unwrap();
        let s: f64 = f
            .iter()
            .zip(&p)
            .filter(|(f, _)| (*f - 6000.0).abs() < 10.0)
            .map(|(_, p)| p)
            .sum();
        10.0 * s.log10()
    };
    let drop = line_db(&y, 20.0, 45.0) - line_db(&z, 20.0, 45.0);
    println!("line removed by {drop:.1} dB");
    assert!(drop > 30.0);
    let e = 8 * sr as usize;
    assert!(z[..e].iter().zip(&y[..e]).all(|(a, b)| a == b));
}

/// steady = keyed + fading 4 kHz carrier (float64)
fn steady(y: &[f32], sr: u32) -> Vec<f64> {
    let k1 = 2.0 * std::f64::consts::PI * 4000.0;
    let k2 = 2.0 * std::f64::consts::PI * 0.1;
    y.iter()
        .enumerate()
        .map(|(i, &v)| {
            let t = i as f64 / sr as f64;
            v as f64 + 0.01 * (k1 * t).sin() * (1.0 + 0.3 * (k2 * t).sin())
        })
        .collect()
}

/// A keyed line is classified keyed, a steady carrier is not (+ the steady case against the
/// stored reference)
#[test]
fn test_keyed_vs_steady() {
    let (y, sr) = keyed_y();
    let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let p = DetectParams {
        fmin: 1000.0,
        ..Default::default()
    };
    let d = itf::detect(&y64, sr, &p).unwrap();
    assert_eq!(d.len(), 1);
    let (c, m) = itf::keying(&y64, sr, d[0].hz, &d[0].spans).unwrap();
    assert!(itf::is_keyed(c, m));
    let s = steady(&y, sr);
    let ds = itf::detect(&s, sr, &p).unwrap();
    let ks: Vec<(f64, f64)> = ds
        .iter()
        .map(|d| itf::keying(&s, sr, d.hz, &d.spans).unwrap())
        .collect();
    for (d, k) in ds.iter().zip(&ks) {
        assert_eq!(itf::is_keyed(k.0, k.1), (d.hz - 6000.0).abs() < 2.0);
    }
    let (dh, dsn, dk) = cmp_dets("steady f64", &ds, &ks, &ref_dets("steady", "det"));
    assert!(dh <= 0.01 && dsn <= 0.05 && dk <= 1e-6);
}

#[test]
fn keyed_gated_notch_ref_and_streaming() {
    let (y, sr) = keyed_y();
    let l = txt("keyed", "notch");
    let w: Vec<f64> = l[0].split(' ').map(pf).collect();
    let (hz, q) = (w[0], w[1]);
    let sp = ref_dets("keyed", "det")[0].d.spans.clone();
    let zp: Vec<f32> = ld("keyed", "z").into_iter().map(|v| v as f32).collect();
    let g = Gate::default();
    let z = itf::notch_spans(&y, 1, sr, hz, q, Some(&sp), &g).unwrap();
    let (db, mx) = rms_db(&z, &zp);
    println!("keyed gated notch vs reference: {db:.1} dB RMS, max abs {mx:.2e}");
    assert!(db <= -120.0);
    for blk in [4096, 65_536, 1_000_003] {
        let s = stream_gated(&y, 1, sr, hz, q, &sp, &g, blk);
        let (dbs, mxs) = rms_db(&s, &z);
        assert!(mxs <= 1e-7, "streaming vs whole: {dbs:.1} dB, {mxs:.2e}");
    }
}

#[test]
fn notch_spans_edge_cases() {
    let (shape, y) = ld_f32_in(dir(), "edge", "y");
    let (ch, sr) = (shape[1], ld("edge", "sr")[0] as u32);
    for l in txt("edge", "cases") {
        let w: Vec<&str> = l.split(' ').collect();
        let name = w[0];
        let sp = if w[1] == "none" {
            None
        } else {
            Some(spans_of(w[1]))
        };
        let g = if w.len() > 2 {
            Gate {
                margin: pf(w[3]),
                fade: 0.2,
            }
        } else {
            Gate::default()
        };
        let zp = ld_f32_in(dir(), "edge", &format!("z_{name}")).1;
        let z = itf::notch_spans(&y, ch, sr, 1234.5, 1234.5 / 150.0, sp.as_deref(), &g).unwrap();
        let (db, mx) = rms_db(&z, &zp);
        println!("edge {name}: {db:.1} dB, max {mx:.2e}");
        assert!(db <= -120.0, "edge {name}: {db:.1} dB");
        if let Some(sp) = &sp {
            for blk in [1000, 7919, 40_000] {
                let s = stream_gated(&y, ch, sr, 1234.5, 1234.5 / 150.0, sp, &g, blk);
                assert!(rms_db(&s, &z).1 <= 1e-7, "edge {name} streaming blk {blk}");
            }
        }
    }
}

#[test]
fn run_gated_equals_whole_array_chain() {
    use ts_core::correct::{self, Lowpass, Params, Post};
    let (y, sr) = keyed_y();
    let n = 20 * sr as usize;
    let x: Vec<f32> = y[..n].iter().flat_map(|&v| [v, 0.5 * v + 0.01]).collect();
    let t: Vec<f64> = (0..=4000).map(|i| i as f64 * 0.005).collect();
    let r: Vec<f64> = t
        .iter()
        .map(|&t| 1.002 + 0.002 * (2.0 * std::f64::consts::PI * 1.3 * t).sin())
        .collect();
    let p = Params::default();
    let post = Post {
        notches: vec![(2000.0, 30.0)],
        lowpass: Some(Lowpass::new(7000.0)),
    };
    let gated = vec![
        (6000.0, 40.0, vec![(5.0, 10.0)]),
        (3000.0, 20.0, vec![(0.0, 3.0), (12.0, 25.0)]),
    ];
    let g = Gate::default();
    let mut st = Vec::new();
    let nout = itf::run_gated(&x, 2, sr, &t, &r, &p, &post, &gated, &g, |b: &[f32]| {
        st.extend_from_slice(b);
        Ok::<(), ()>(())
    })
    .unwrap();
    let mut w = correct::correct(&x, 2, sr, &t, &r, &p).unwrap();
    correct::apply_notch(&mut w, 2, sr, 2000.0, 30.0).unwrap();
    for (hz, q, sp) in &gated {
        w = itf::notch_spans(&w, 2, sr, *hz, *q, Some(sp), &g).unwrap();
    }
    correct::apply_lowpass(&mut w, 2, sr, post.lowpass.as_ref().unwrap()).unwrap();
    assert_eq!(st.len(), nout * 2);
    let (db, mx) = rms_db(&st, &w);
    println!("run_gated vs whole-array chain: {db:.1} dB, max {mx:.2e}");
    assert!(mx <= 1e-7);
}

#[test]
fn one_thread_equals_many() {
    let (y, sr) = keyed_y();
    let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let go = || {
        let p = DetectParams {
            fmin: 1000.0,
            chunk: 13.0, // several chunks -> parallel chunk groups
            ..Default::default()
        };
        let d = itf::detect(&y64, sr, &p).unwrap();
        let k: Vec<(f64, f64)> = d
            .iter()
            .map(|d| {
                itf::keying(&y64, sr, d.hz, &[(0.0, 20.0), (20.0, 45.0), (45.0, 60.0)]).unwrap()
            })
            .collect();
        let z = itf::notch_spans(
            &y,
            1,
            sr,
            6000.0,
            40.0,
            Some(&[(14.0, 20.0), (30.0, 50.0)]),
            &Gate::default(),
        )
        .unwrap();
        let tl = ld("two", "t");
        let c = Curve {
            t: tl.clone(),
            ratio: ld("two", "r1"),
            confidence: vec![1.0; tl.len()],
            valid: ldb("two", "v1"),
            meta: Default::default(),
        };
        let x: Vec<f32> = ld("two", "x").into_iter().map(|v| v as f32).collect();
        let f = ev::fading_metric(
            &x,
            1,
            44100.0,
            9000.0,
            &c,
            Channel::Mid,
            &FadingParams::default(),
            &[],
        )
        .unwrap();
        let b = ev::band_stats_default(
            &ld("band", "t"),
            &ld("band", "r1"),
            &ld("band", "r2"),
            &ldb("band", "v"),
        )
        .unwrap();
        (d, k, z, f, b)
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
    assert!(one.0 == many.0, "thread count changed detect");
    assert!(one.1 == many.1, "thread count changed keying");
    assert!(one.2 == many.2, "thread count changed notch_spans");
    // Debug prints the shortest round-trip repr: equal strings = equal bits (NaN != NaN)
    assert_eq!(
        format!("{:?}", one.3),
        format!("{:?}", many.3),
        "thread count changed fading"
    );
    assert!(one.4 == many.4, "thread count changed band_stats");
}

#[test]
fn detect_short_tail_errors() {
    // last chunk of <= NPER/2 samples is an error (noverlap >= nperseg, SciPy-compatible STFT)
    let sr = 8000u32;
    let y: Vec<f64> = (0..(10 * sr as usize + 3000))
        .map(|i| (i as f64 * 0.37).sin())
        .collect();
    let p = DetectParams {
        fmin: 1000.0,
        chunk: 10.0,
        ..Default::default()
    };
    assert!(itf::detect(&y[..10 * sr as usize + 1000], sr, &p).is_err());
    let y2 = &y[..]; // 3000-sample tail: one clipped frame, no error
    assert!(itf::detect(y2, sr, &p).is_ok());
}

// ------------------------------------------------------------------ evidence

/// Compare a result dict (flat "key value" lines) with the stored reference text.
fn cmp_dict(tag: &str, rs: &[String], want: &[String], tol: f64) -> f64 {
    assert_eq!(rs.len(), want.len(), "{tag}: key count\n{rs:?}\n{want:?}");
    let mut worst = 0.0f64;
    for (a, b) in rs.iter().zip(want) {
        let (ka, va) = a.split_once(' ').unwrap();
        let (kb, vb) = b.split_once(' ').unwrap();
        assert_eq!(ka, kb, "{tag}: key order");
        if vb.starts_with('[') {
            let pa: Vec<f64> = va
                .trim_matches(['[', ']'])
                .split(',')
                .filter(|s| !s.is_empty())
                .map(pf)
                .collect();
            let pb: Vec<f64> = vb
                .trim_matches(['[', ']'])
                .split(',')
                .filter(|s| !s.is_empty())
                .map(pf)
                .collect();
            assert_eq!(pa.len(), pb.len(), "{tag}: {ka} length");
            for (x, y) in pa.iter().zip(&pb) {
                // rounded to 3 decimals: equal unless within 1e-6 of a rounding tie
                assert!(
                    relerr(*x, *y) == 0.0 || (x - y).abs() <= 1.0001e-3,
                    "{tag}: {ka} {x} vs {y}"
                );
            }
        } else if vb.starts_with("int ") || vb == "{}" {
            assert_eq!(va, vb, "{tag}: {ka}");
        } else {
            let e = relerr(pf(va), pf(vb));
            worst = worst.max(e);
            assert!(e <= tol, "{tag}: {ka} {va} vs {vb} (rel {e:.2e})");
        }
    }
    println!("{tag}: {} keys, max rel {worst:.2e}", rs.len());
    worst
}

fn fading_lines(f: &ev::Fading) -> Vec<String> {
    let sc = f.scalars();
    let (head, tail) = sc.split_at(sc.len() - 2);
    let mut out: Vec<String> = head.iter().map(|(k, v)| format!("{k} {v:?}")).collect();
    out.push(format!(
        "seg_rho [{}]",
        f.seg_rho
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(",")
    ));
    out.extend(tail.iter().map(|(k, v)| format!("{k} {v:?}")));
    out
}

fn band_lines(b: &BandStats) -> Vec<String> {
    let mut out = vec![format!("n_seg int {}", b.n_seg)];
    for (name, v) in [
        ("coh", &b.coh),
        ("rms1", &b.rms1),
        ("rms2", &b.rms2),
        ("rms_diff", &b.rms_diff),
    ] {
        if v.is_empty() {
            out.push(format!("{name} {{}}"));
        }
        out.extend(v.iter().map(|(k, x)| format!("{name}.{k} {x:?}")));
    }
    out
}

fn curve(t: Vec<f64>, r: Vec<f64>, v: Vec<bool>) -> Curve {
    Curve {
        confidence: vec![1.0; t.len()],
        t,
        ratio: r,
        valid: v,
        meta: Default::default(),
    }
}

#[test]
fn fading_ref() {
    let t = ld("two", "t");
    let c1 = curve(t.clone(), ld("two", "r1"), ldb("two", "v1"));
    let c2 = curve(t.clone(), ld("two", "r2"), ldb("two", "v2"));
    let x: Vec<f32> = ld("two", "x").into_iter().map(|v| v as f32).collect();
    let lp3 = ev::lowpass_curve(&t, &c1.ratio, 3.0).unwrap();
    let mp: Vec<f64> = t
        .iter()
        .map(|&t| 1.01 + 0.003 * (2.0 * std::f64::consts::PI * 7.46 * t).sin())
        .collect();
    let ones = vec![true; t.len()];
    let p = FadingParams::default();
    let t0 = std::time::Instant::now();
    let f = ev::fading_metric(
        &x,
        1,
        44100.0,
        9000.0,
        &c1,
        Channel::Mid,
        &p,
        &[("final", &lp3, &c1.valid), ("model", &mp, &ones)],
    )
    .unwrap();
    println!(
        "two fading_metric {:.1} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );
    cmp_dict("two fading", &fading_lines(&f), &txt("two", "fading"), 1e-6);
    let f2 = ev::fading_metric(&x, 1, 44100.0, 15625.0, &c2, Channel::Mid, &p, &[]).unwrap();
    cmp_dict(
        "two fading2",
        &fading_lines(&f2),
        &txt("two", "fading2"),
        1e-6,
    );
    // to_value keeps the reference key order
    let keys: Vec<String> = match f.to_value() {
        ts_io::json::Value::Obj(o) => o.into_iter().map(|(k, _)| k).collect(),
        _ => unreachable!(),
    };
    let ref_keys: Vec<String> = txt("two", "fading")
        .iter()
        .map(|l| l.split(' ').next().unwrap().to_string())
        .collect();
    assert_eq!(keys, ref_keys);

    let (shape, xf) = ld_f32_in(dir(), "fade", "x");
    let cf = curve(ld("fade", "t"), ld("fade", "r"), ldb("fade", "v"));
    let fin: Vec<f64> =
        cf.t.iter()
            .zip(&cf.ratio)
            .map(|(&t, &r)| r + 1e-4 * (2.0 * std::f64::consts::PI * 3.0 * t).sin())
            .collect();
    let pf_ = FadingParams {
        seg: 10.0,
        ..Default::default()
    };
    for (name, chn) in [("mid", Channel::Mid), ("L", Channel::L), ("S", Channel::S)] {
        let f = ev::fading_metric(
            &xf,
            shape[1],
            8000.0,
            3000.0,
            &cf,
            chn,
            &pf_,
            &[("final", &fin, &cf.valid)],
        )
        .unwrap();
        cmp_dict(
            &format!("fade {name}"),
            &fading_lines(&f),
            &txt("fade", &format!("fading_{name}")),
            1e-6,
        );
    }
}

#[test]
fn band_stats_ref() {
    let t = ld("two", "t");
    let (r1, r2) = (ld("two", "r1"), ld("two", "r2"));
    let com: Vec<bool> = ldb("two", "v1")
        .iter()
        .zip(ldb("two", "v2"))
        .map(|(a, b)| *a && b)
        .collect();
    let q = pf(&txt("two", "q")[0]);
    let r1q: Vec<f64> = r1.iter().map(|v| v / q).collect();
    let b = ev::band_stats_default(&t, &r1q, &r2, &com).unwrap();
    cmp_dict("two coh", &band_lines(&b), &txt("two", "coh"), 1e-6);

    let (tb, b1, b2, b3, vb) = (
        ld("band", "t"),
        ld("band", "r1"),
        ld("band", "r2"),
        ld("band", "r3"),
        ldb("band", "v"),
    );
    let s = ev::band_stats_default(&tb, &b1, &b2, &vb).unwrap();
    cmp_dict("band", &band_lines(&s), &txt("band", "stats"), 1e-9);
    let s = ev::band_stats(&tb, &b1, &b2, &vb, &ev::BANDS, 1024, 0.8).unwrap();
    cmp_dict(
        "band nper1024",
        &band_lines(&s),
        &txt("band", "stats_nper1024"),
        1e-9,
    );
    let s = ev::band_stats_default(&tb, &b1, &b3, &vb).unwrap();
    cmp_dict(
        "band incoh",
        &band_lines(&s),
        &txt("band", "stats_incoh"),
        1e-9,
    );
}

#[test]
fn lowpass_curve_ref() {
    let t = ld("two", "t");
    let r = ld("two", "r1");
    let mut worst = 0.0f64;
    let mut chk = |a: &[f64], b: &[f64], tag: &str| {
        assert_eq!(a.len(), b.len());
        let rms = (b.iter().map(|v| v * v).sum::<f64>() / b.len() as f64).sqrt();
        let e = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
            / rms;
        worst = worst.max(e);
        assert!(e <= 1e-10, "{tag}: {e:.2e}");
    };
    for fc in [6.0, 2.5, 0.7] {
        let name = format!("lp_{}", ts_io::fmt_general(fc, 6));
        chk(
            &ev::lowpass_curve(&t, &r, fc).unwrap(),
            &ld("two", &name),
            &name,
        );
    }
    chk(
        &ev::lowpass_curve(&t[..150], &r[..150], 0.7).unwrap(),
        &ld("two", "lp_short"),
        "short",
    );
    let (tb, b1) = (ld("band", "t"), ld("band", "r1"));
    for fc in [6.0, 0.5] {
        let name = format!("lp_{}", ts_io::fmt_general(fc, 6));
        chk(
            &ev::lowpass_curve(&tb, &b1, fc).unwrap(),
            &ld("band", &name),
            &name,
        );
    }
    println!("lowpass_curve: max abs / rms {worst:.2e}");
}

#[test]
fn bandwidth_policy_ref() {
    let coh = |name: &str| -> Option<BandStats> {
        let mk = |v: &[(&str, f64)]| BandStats {
            coh: v.iter().map(|(k, x)| (k.to_string(), *x)).collect(),
            ..Default::default()
        };
        match name {
            "none" => None,
            "empty" => Some(BandStats {
                n_seg: 2,
                ..Default::default()
            }),
            "all" => Some(mk(&[("0-0.5", 0.9), ("0.5-6", 0.9), ("6-25", 0.8)])),
            "to6" => Some(mk(&[("0-0.5", 0.9), ("0.5-6", 0.6), ("6-25", 0.2)])),
            "low" => Some(mk(&[("0-0.5", 0.9), ("0.5-6", 0.3), ("6-25", 0.9)])),
            "edge" => Some(mk(&[("0-0.5", 0.1), ("0.5-6", 0.5), ("6-25", 0.5)])),
            k => {
                let v: Vec<(String, f64)> = txt("policy", &format!("coh_{k}"))
                    .iter()
                    .map(|l| {
                        let (a, b) = l.split_once(' ').unwrap();
                        (a.to_string(), pf(b))
                    })
                    .collect();
                Some(BandStats {
                    coh: v,
                    ..Default::default()
                })
            }
        }
    };
    let rho = |name: &str| match name {
        "none" | "err" => None,
        "nan" => Some(f64::NAN),
        "hi" => Some(0.93),
        "lo" => Some(0.5),
        "at" => Some(0.9),
        _ => unreachable!(),
    };
    let lines = txt("policy", "cases");
    for l in &lines {
        let w: Vec<&str> = l.splitn(7, '|').collect();
        let mode: CorrLimit = w[0].parse().unwrap();
        let c = coh(w[2]);
        let (fc, why) = ev::bandwidth_policy(w[1], rho(w[3]), c.as_ref(), pf(w[4]), mode);
        let want = if w[5] == "none" { None } else { Some(pf(w[5])) };
        assert_eq!((fc, why.as_str()), (want, w[6]), "{l}");
    }
    println!("bandwidth_policy: {} cases equal", lines.len());
    // keep-model-HF: ratio + mp - lowpass(mp)
    let t = ld("two", "t");
    let r = ld("two", "r1");
    let mp: Vec<f64> = t
        .iter()
        .map(|&t| 1.0 + 0.003 * (2.0 * std::f64::consts::PI * 7.46 * t).sin())
        .collect();
    let (o, keep) = ev::apply_bandwidth(&t, &r, Some(2.5), Some(&mp)).unwrap();
    assert!(keep);
    let lr = ev::lowpass_curve(&t, &r, 2.5).unwrap();
    let lm = ev::lowpass_curve(&t, &mp, 2.5).unwrap();
    for i in 0..t.len() {
        assert_eq!(o[i], lr[i] + mp[i] - lm[i]);
    }
    assert_eq!(
        ev::apply_bandwidth(&t, &r, None, Some(&mp)).unwrap(),
        (r.clone(), false)
    );
}
