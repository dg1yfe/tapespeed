//! Synth against stored reference results (recorded random draws, tests/data/n), the
//! SciPy-compatible DSP helpers it uses, and property tests of the synthetic signals.
//! Run with `--nocapture` to see the differences.
#[path = "../../testdata.rs"]
mod testdata;

use std::path::PathBuf;

use ts_core::curve::{read_curve, write_curve};
use ts_core::synth::{self, PcgFactory, Recorded, RecordedDraws, Stream};

fn path(name: &str) -> PathBuf {
    testdata::dir("n").join(name)
}

fn ld(name: &str) -> Vec<f64> {
    let f =
        std::fs::File::open(path(&format!("{name}.npy"))).unwrap_or_else(|e| panic!("{name}: {e}"));
    let npy = npyz::NpyFile::new(std::io::BufReader::new(f)).unwrap();
    if npy.dtype().descr().contains("f4") {
        npy.into_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    } else {
        npy.into_vec::<f64>().unwrap()
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn db(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut e2, mut r2) = (0.0f64, 0.0f64);
    for (&p, &q) in a.iter().zip(b) {
        e2 += (p as f64 - q).powi(2);
        r2 += q * q;
    }
    10.0 * (e2.max(1e-300) / r2).log10()
}

const NAMES: [&str; 5] = ["speed", "env", "prog", "hiss", "mech"];

fn recorded(case: &str) -> Recorded {
    Recorded::new(
        NAMES.map(|s| {
            RecordedDraws::new(ld(&format!("{case}__u_{s}")), ld(&format!("{case}__n_{s}")))
        }),
    )
}

fn case_params(case: &str) -> synth::Params {
    let txt = std::fs::read_to_string(path(&format!("{case}__params.txt"))).unwrap();
    let mut kv: Vec<(String, String)> = txt
        .lines()
        .map(|l| {
            let (k, v) = l.split_once(' ').unwrap_or((l, ""));
            (k.to_string(), v.to_string())
        })
        .collect();
    let preset = kv.remove(0).1;
    let kw: Vec<(&str, &str)> = kv.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    synth::make_params(&preset, &kw).unwrap()
}

#[test]
fn recorded_draw_ref() {
    for case in ["fm19", "am9w_pair", "mech", "clean"] {
        let p = case_params(case);
        let rec = recorded(case);
        let s = synth::run_with(&p, &rec).unwrap();
        assert!(
            rec.all_consumed(),
            "{case}: draw count differs, used {:?}",
            rec.used(Stream::Program)
        );
        let (t, r) = (ld(&format!("{case}__t")), ld(&format!("{case}__ratio")));
        assert_eq!(s.truth.t, t, "{case} t grid");
        let er = max_abs_diff(&s.truth.ratio, &r);
        let x = ld(&format!("{case}__x"));
        let ex = db(&s.audio.data, &x);
        let mx = max_abs_diff(
            &s.audio.data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            &x,
        );
        let exs = if mx == 0.0 {
            "bit-exact".to_string()
        } else {
            format!("{ex:.1} dB")
        };
        eprintln!("[synth-ref] {case:<10} curve max|d| {er:.2e}  audio {exs}  max|d| {mx:.2e}");
        assert!(er <= 1e-9, "{case}: curve {er:e}");
        assert!(ex <= -120.0, "{case}: audio {ex} dB");
        let meta = std::fs::read_to_string(path(&format!("{case}__meta.txt"))).unwrap();
        let got: String = s
            .truth
            .meta
            .iter()
            .map(|(k, v)| format!("{k}={}\n", v.to_text()))
            .collect();
        assert_eq!(got, meta, "{case}: meta");
    }
}

#[test]
fn warp_kernel_and_output() {
    let tab = synth::warp_table(64, 0.9, 9.0, 2048);
    let want = ld("warp__tab");
    let e = max_abs_diff(&tab, &want);
    eprintln!("[synth-ref] warp table max|d| {e:.2e}");
    assert!(e <= 1e-16, "table {e:e}");
    let s = ld("warp__s");
    let cols: Vec<Vec<f64>> = (0..2)
        .map(|c| s.iter().skip(c).step_by(2).copied().collect())
        .collect();
    let u = ld("warp__u");
    let out = synth::warp(&cols, &u, 64, 0.9, 9.0, 2048);
    let inter: Vec<f64> = (0..u.len()).flat_map(|i| [out[0][i], out[1][i]]).collect();
    let e = max_abs_diff(&inter, &ld("warp__out"));
    eprintln!("[synth-ref] warp output max|d| {e:.2e}");
    assert!(e <= 1e-13, "warp {e:e}");
}

#[test]
fn dsp_helpers_ref() {
    let x = ld("gauss__x");
    for sig in ["0.5", "0.001", "3.0", "30.0", "45.0"] {
        let g = ts_dsp::ndimage::gaussian_filter1d(
            &x,
            sig.parse().unwrap(),
            ts_dsp::ndimage::Mode::Nearest,
        )
        .unwrap();
        let e = max_abs_diff(&g, &ld(&format!("gauss__{sig}")));
        assert!(e <= 1e-15, "gaussian sigma {sig}: {e:e}");
    }
    let x = ld("rp__x");
    for (up, down) in [(160, 147), (147, 160), (3, 2), (1, 3), (2, 1), (4, 4)] {
        let y = ts_dsp::conv::resample_poly(&x, up, down, None).unwrap();
        let e = max_abs_diff(&y, &ld(&format!("rp__{up}_{down}")));
        assert!(e <= 1e-13, "resample_poly {up}/{down}: {e:e}");
    }
    for n in [1000usize, 1001] {
        let xx: Vec<f64> = x.iter().chain(&x).take(n).copied().collect();
        let z: Vec<f64> = ts_dsp::spectral::hilbert(&xx)
            .iter()
            .flat_map(|c| [c.re, c.im])
            .collect();
        let e = max_abs_diff(&z, &ld(&format!("hil__{n}")));
        assert!(e <= 1e-13, "hilbert {n}: {e:e}");
    }
}

#[test]
fn rayon_thread_invariance() {
    let p = case_params("mech");
    let f = || {
        let s = synth::run_with(&p, &recorded("mech")).unwrap();
        (
            s.audio.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            s.truth.ratio.clone(),
        )
    };
    let p1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let p8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let (a, b) = (p1.install(f), p8.install(f));
    assert!(a == b, "thread count changes the result");
}

// ---------------------------------------------------------------------- properties ---
// Property tests run on the default PCG source (the recorded-draw source covers the reference).

#[test]
fn test_t_is_integral_of_r() {
    let p = synth::make_params("am9w", &[("duration", "20")]).unwrap();
    let mut rng = synth::RandomFactory::stream(&PcgFactory { seed: 1 }, Stream::Speed);
    let sp = synth::Speed::new(&p, rng.as_mut(), None).unwrap();
    let t = ts_dsp::window::linspace(0.0, 20.0, 400001);
    let r = sp.r_vec(&t);
    let tt = sp.t_vec(&t);
    let mut tn = 0.0;
    let mut e: f64 = 0.0;
    for i in 1..t.len() {
        tn += (r[i] + r[i - 1]) / 2.0 * (t[i] - t[i - 1]);
        e = e.max((tn - tt[i]).abs());
    }
    assert!(e < 1e-9, "{e:e}");
}

#[test]
fn test_clean_tone_inst_freq() {
    let p = synth::make_params(
        "clean",
        &[
            ("duration", "3"),
            ("channels", "1"),
            ("ref_hz", "9000"),
            ("wow", "0.8:0.3,5.5:0.1"),
        ],
    )
    .unwrap();
    let s = synth::run(&p).unwrap();
    let sr = p.sr as f64;
    let x: Vec<f64> = s.audio.data.iter().map(|&v| v as f64).collect();
    let z = ts_dsp::spectral::hilbert(&x);
    // diff(unwrap(angle(z))) == angle(z[k+1] conj z[k]) for steps < pi
    let f: Vec<f64> = z
        .windows(2)
        .map(|w| (w[1] * w[0].conj()).arg() * sr / (2.0 * std::f64::consts::PI))
        .collect();
    let tm: Vec<f64> = (0..f.len()).map(|k| (k as f64 + 0.5) / sr).collect();
    let sel: Vec<usize> = (0..f.len())
        .filter(|&k| tm[k] > 0.3 && tm[k] < 2.7)
        .collect();
    let b = 48;
    let mut worst: f64 = 0.0;
    for blk in sel.chunks_exact(b) {
        let fb = blk.iter().map(|&k| f[k]).sum::<f64>() / b as f64;
        let tb = blk.iter().map(|&k| tm[k]).sum::<f64>() / b as f64;
        worst = worst.max((fb / (p.ref_hz * s.truth.ratio_at(tb)) - 1.0).abs());
    }
    assert!(worst < 1e-5, "{worst:e}");
    let ptp = ts_dsp::stats::ptp(&s.truth.ratio).unwrap();
    assert!(ptp > 0.003);
}

#[test]
fn test_warp_sine() {
    let (sr, f) = (48000.0, 15000.0);
    let s: Vec<f64> = (0..48000)
        .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin())
        .collect();
    let u = ts_dsp::window::linspace(200.0, 47000.0, 30000);
    let y = synth::warp(&[s], &u, 64, 0.9, 9.0, 2048);
    let e = u
        .iter()
        .zip(&y[0])
        .map(|(&uu, &v)| (v - (2.0 * std::f64::consts::PI * f * uu / sr).sin()).abs())
        .fold(0.0, f64::max);
    assert!(e < 1e-4, "{e:e}");
}

#[test]
fn test_deterministic_and_seed() {
    let p = synth::make_params("am9", &[("duration", "1.5")]).unwrap();
    let (a, b) = (synth::run(&p).unwrap(), synth::run(&p).unwrap());
    assert_eq!(a.audio.data, b.audio.data);
    assert_eq!(a.truth.ratio, b.truth.ratio);
    let c = synth::run(&synth::make_params("am9", &[("duration", "1.5"), ("seed", "1")]).unwrap())
        .unwrap();
    assert_ne!(a.audio.data, c.audio.data);
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ts_n_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn test_curve_roundtrip() {
    let p = synth::make_params("fm19", &[("duration", "1.0"), ("seed", "7")]).unwrap();
    let s = synth::run(&p).unwrap();
    assert_eq!((s.audio.frames(), s.audio.channels), (48000, 2));
    let c = &s.truth;
    assert_eq!(c.t[0], 0.0);
    assert!(*c.t.last().unwrap() >= (s.audio.frames() - 1) as f64 / p.sr as f64);
    assert!(c.t.windows(2).all(|w| (w[1] - w[0] - 0.005).abs() < 1e-8));
    let f = tmpdir("rt").join("t.csv");
    write_curve(&f, c).unwrap();
    let d = read_curve(&f).unwrap();
    assert!(max_abs_diff(&d.t, &c.t) <= 1e-6 && max_abs_diff(&d.ratio, &c.ratio) <= 1e-9);
    assert!(d.valid.iter().all(|&v| v) && d.confidence.iter().all(|&v| v == 1.0));
    let m = |k: &str| d.meta[k].to_text();
    assert_eq!(m("ref_hz").parse::<f64>().unwrap(), 19000.0);
    assert_eq!(m("sr").parse::<i64>().unwrap(), 48000);
    assert_eq!(m("preset"), "fm19");
    assert_eq!(m("seed"), "7");
}

#[test]
fn test_cli() {
    let dir = tmpdir("cli");
    let out = dir.join("a.wav");
    let ov = [
        ("duration".to_string(), "1".to_string()),
        ("channels".to_string(), "1".to_string()),
    ];
    synth::cli("am9w", &ov, &out, None).unwrap();
    let a = ts_io::load(&out, None, None).unwrap();
    let c = read_curve(dir.join("a_truth.csv")).unwrap();
    assert_eq!((a.frames(), a.channels, a.sr), (48000, 1, 48000));
    assert_eq!(c.meta["ref_hz"].to_text().parse::<f64>().unwrap(), 9000.0);
}

/// All presets at their default durations (mech: 720 s); `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn full_presets_smoke() {
    for name in ["custom", "fm19", "am9w", "am9", "reel", "clean", "mech"] {
        let t0 = std::time::Instant::now();
        let p = synth::make_params(name, &[]).unwrap();
        let s = synth::run(&p).unwrap();
        let n = (p.duration * p.sr as f64).round() as usize;
        assert_eq!((s.audio.frames(), s.audio.channels), (n, p.channels));
        assert!(
            s.audio
                .data
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 0.99 + 1e-6)
        );
        eprintln!(
            "[N-smoke] {name:<6} {:>6.1} s audio in {:.2?}",
            p.duration,
            t0.elapsed()
        );
    }
}
