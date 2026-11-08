use std::path::PathBuf;
use std::process::Command;
use ts_io::*;

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// deterministic pseudo-noise incl. values a 24 bit encoder would round
fn signal(frames: usize, ch: usize) -> Vec<f32> {
    let mut s = 0x1234_5678_9abc_def0u64;
    (0..frames * ch)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            (0.7 * r * (1.0 + (i % 7) as f64 * 0.01)) as f32
        })
        .collect()
}

#[test]
fn wav_roundtrip_bit_exact() {
    let x = signal(44100 / 2 + 3, 2);
    let (a, b) = (tmp("rt_a.wav"), tmp("rt_b.wav"));
    save(&a, &x, 2, 44100, None).unwrap();
    let l = load(&a, None, None).unwrap();
    assert_eq!((l.sr, l.channels, l.frames()), (44100, 2, x.len() / 2));
    assert!(
        l.data
            .iter()
            .zip(&x)
            .all(|(p, q)| p.to_bits() == q.to_bits()),
        "samples differ"
    );
    save(&b, &l.data, 2, l.sr, None).unwrap();
    assert_eq!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "re-saved WAV differs"
    );
    let i = info(&a).unwrap();
    assert_eq!(
        (i.codec.as_str(), i.sr, i.channels),
        ("pcm_f32le", 44100, 2)
    );
    assert_eq!(probe(&a).unwrap(), (44100, 2));
    assert!((i.duration - x.len() as f64 / 2.0 / 44100.0).abs() < 1e-3);
    assert!(i.bit_rate > 0);
}

#[test]
fn segment_and_chunked_load() {
    let x = signal(48000 * 2, 1);
    let a = tmp("seg.wav");
    save(&a, &x, 1, 48000, None).unwrap();
    let seg = load(&a, Some(0.5), Some(0.25)).unwrap();
    assert_eq!(seg.frames(), 12000);
    assert!(seg.data.iter().zip(&x[24000..]).all(|(p, q)| p == q));
    let mut n = 0;
    let mut chunks = 0;
    load_chunks(&a, None, None, 1000, |c| {
        assert!(c.len() <= 1000);
        assert_eq!(c, &x[n..n + c.len()]);
        n += c.len();
        chunks += 1;
    })
    .unwrap();
    assert_eq!(n, x.len());
    assert!(chunks >= 96);
}

#[test]
fn flac_is_24_bit() {
    let x = signal(30000, 2);
    let a = tmp("t.flac");
    save(&a, &x, 2, 44100, None).unwrap();
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,bits_per_raw_sample",
            "-of",
            "csv=p=0",
        ])
        .arg(&a)
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "flac,24");
    let l = load(&a, None, None).unwrap();
    let q = 8388608.0f64;
    let worst = l
        .data
        .iter()
        .zip(&x)
        .map(|(p, q2)| (*p as f64 - *q2 as f64).abs())
        .fold(0.0, f64::max);
    assert!(worst <= 1.0 / q, "24 bit quantisation error {worst}");
}

#[test]
fn errors_carry_ffmpeg_stderr() {
    assert!(matches!(
        load(tmp("missing.wav"), None, None),
        Err(Error::Failed("ffprobe", _, _))
    ));
}
