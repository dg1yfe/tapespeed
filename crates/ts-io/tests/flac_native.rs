//! In-process FLAC encoder vs the ffmpeg path: decoded samples bit-identical, valid stream.
use std::path::{Path, PathBuf};
use std::process::Command;
use ts_io::flac::{BLOCK, q24};
use ts_io::*;

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// pseudo-noise plus the values where the quantiser can go wrong: ±full scale and beyond, ±inf, NaN,
/// half-LSB ties at 2^-31 and 2^-23, subnormals, -0.0
fn signal(frames: usize, ch: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut x: Vec<f32> = (0..frames * ch)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            // slow sine + noise so the predictors have something to do
            (0.5 * (i as f64 * 0.01).sin() + 0.2 * r) as f32
        })
        .collect();
    let lsb = 1.0 / 8_388_608.0f32;
    let tiny = 1.0 / 2_147_483_648.0f32;
    let special = [
        1.0,
        -1.0,
        1.5,
        -1.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0 - f32::EPSILON / 2.0,
        -1.0 + f32::EPSILON / 2.0,
        0.5 * tiny,
        1.5 * tiny,
        -0.5 * tiny,
        -1.5 * tiny,
        2.5 * tiny,
        0.5 * lsb,
        -0.5 * lsb,
        1.5 * lsb,
        -1.5 * lsb,
        0.5 * lsb - tiny,
        -0.5 * lsb + tiny,
        f32::MIN_POSITIVE / 4.0,
        -0.0,
        0.0,
        255.5 * tiny,
        -255.5 * tiny,
        256.5 * tiny,
        -256.5 * tiny,
    ];
    for (k, v) in special.iter().enumerate() {
        let at = (k * 37 + 5) % x.len().max(1);
        if at < x.len() {
            x[at] = *v;
        }
    }
    x
}

fn decode_s32(p: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(p)
        .args(["-f", "s32le", "-acodec", "pcm_s32le", "-"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "ffmpeg decode: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn probe_fmt(p: &Path) -> String {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_fmt,bits_per_raw_sample,sample_rate,channels",
            "-of",
            "csv=p=0",
        ])
        .arg(p)
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn flac_test(p: &Path) {
    match Command::new("flac").arg("-t").arg("-s").arg(p).output() {
        Ok(o) => assert!(
            o.status.success() && o.stderr.is_empty(),
            "flac -t {}: {}",
            p.display(),
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(_) => eprintln!("flac not installed, skipping flac -t"),
    }
}

fn same_as_ffmpeg(name: &str, frames: usize, ch: usize, sr: u32, chunk: usize) {
    same_as_ffmpeg_x(
        name,
        &signal(frames, ch, 0x9e37_79b9 ^ frames as u64),
        ch,
        sr,
        chunk,
    );
}

fn same_as_ffmpeg_x(name: &str, x: &[f32], ch: usize, sr: u32, chunk: usize) {
    let frames = x.len() / ch;
    let (a, b) = (
        tmp(&format!("{name}_native.flac")),
        tmp(&format!("{name}_ffmpeg.flac")),
    );
    // native path, written in uneven chunks
    let mut w = Writer::create(&a, sr, ch, None).unwrap();
    for c in x.chunks(chunk * ch) {
        w.write(c).unwrap();
    }
    w.finish().unwrap();
    let da = decode_s32(&a);
    assert_eq!(da.len(), frames * ch * 4, "{name}: length");
    // explicit codec -> ffmpeg pipe (the old path); ffmpeg's encoder refuses streams < 16 frames
    match save(&b, x, ch, sr, Some("flac")) {
        Ok(()) => {
            let db = decode_s32(&b);
            if da != db {
                let i = da
                    .chunks(4)
                    .zip(db.chunks(4))
                    .position(|(p, q)| p != q)
                    .unwrap();
                panic!("{name}: sample {i} differs (x = {:e})", x[i]);
            }
        }
        Err(e) => assert!(frames < 16, "{name}: ffmpeg failed: {e}"),
    }
    // ffmpeg's own s32 -> 24 bit is `>> 8`; the decoder returns `q24 << 8`
    for (i, s) in da.chunks(4).enumerate() {
        let v = i32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        assert_eq!(v, q24(x[i]) << 8, "{name}: q24 at {i}");
    }
    assert_eq!(
        probe_fmt(&a),
        format!("flac,s32,{sr},{ch},24"),
        "{name}: ffprobe"
    );
    flac_test(&a);
}

#[test]
fn bit_identical_to_ffmpeg_stereo() {
    same_as_ffmpeg("st", 3 * 16 * BLOCK + 1234, 2, 44100, 7001);
}

#[test]
fn bit_identical_to_ffmpeg_mono_odd_rates() {
    same_as_ffmpeg("mono48", 2 * BLOCK + 17, 1, 48000, 333);
    same_as_ffmpeg("mono16", BLOCK - 1, 1, 16000, 1 << 16);
    same_as_ffmpeg("st22", 100_003, 2, 22050, 4096);
    same_as_ffmpeg("st_odd_sr", 50_000, 2, 44056, 999);
}

#[test]
fn short_streams() {
    same_as_ffmpeg("one", 1, 2, 44100, 1);
    same_as_ffmpeg("tiny", 31, 2, 44100, 5);
    same_as_ffmpeg("min", 16, 1, 44100, 16);
    same_as_ffmpeg("block", BLOCK, 2, 44100, BLOCK);
    same_as_ffmpeg("block1", BLOCK + 1, 2, 44100, 1000);
}

#[test]
fn silence_and_full_scale() {
    for (name, v) in [("zero", 0.0f32), ("plus", 2.0), ("minus", -2.0)] {
        let x = vec![v; 2 * 20_000];
        let a = tmp(&format!("{name}_c.flac"));
        save(&a, &x, 2, 44100, None).unwrap();
        let d = decode_s32(&a);
        assert!(
            d.chunks(4)
                .all(|s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) == q24(v) << 8)
        );
        flac_test(&a);
    }
}

#[test]
fn empty_stream() {
    let a = tmp("empty.flac");
    Writer::create(&a, 44100, 2, None)
        .unwrap()
        .finish()
        .unwrap();
    assert!(decode_s32(&a).is_empty());
}

#[test]
fn streaminfo_total_and_md5() {
    let (frames, ch) = (5 * BLOCK + 77, 2);
    let x = signal(frames, ch, 7);
    let a = tmp("si.flac");
    save(&a, &x, ch, 44100, None).unwrap();
    let b = std::fs::read(&a).unwrap();
    assert_eq!(&b[..4], b"fLaC");
    let si = &b[8..42];
    let packed = u64::from_be_bytes(si[10..18].try_into().unwrap());
    assert_eq!(packed & ((1 << 36) - 1), frames as u64);
    assert_eq!((packed >> 44) as u32, 44100);
    assert_eq!(((packed >> 41) & 7) + 1, ch as u64);
    assert_eq!(((packed >> 36) & 31) + 1, 24);
    assert_ne!(&si[18..34], &[0u8; 16], "md5 set");
    if let Ok(o) = Command::new("metaflac")
        .args(["--show-md5sum", "--show-total-samples"])
        .arg(&a)
        .output()
    {
        let pcm: Vec<u8> = x
            .iter()
            .flat_map(|&v| q24(v).to_le_bytes()[..3].to_vec())
            .collect();
        let md5 = Command::new("md5")
            .arg("-q")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut c| {
                use std::io::Write;
                c.stdin.take().unwrap().write_all(&pcm)?;
                c.wait_with_output()
            });
        let txt = String::from_utf8_lossy(&o.stdout).to_string();
        assert!(txt.contains(&frames.to_string()), "{txt}");
        if let Ok(m) = md5 {
            assert!(
                txt.contains(String::from_utf8_lossy(&m.stdout).trim()),
                "{txt}"
            );
        }
    }
}

#[test]
fn env_switch_uses_ffmpeg() {
    // the switch is read at create(); run in a child process so the env var stays local
    if std::env::var("TS_FLAC_CHILD").is_ok() {
        let a = tmp("env_ffmpeg.flac");
        save(&a, &signal(10_000, 2, 3), 2, 44100, None).unwrap();
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let st = Command::new(exe)
        .args(["--exact", "env_switch_uses_ffmpeg", "--nocapture"])
        .env("TS_FLAC_CHILD", "1")
        .env("TAPESPEED_FLAC", "ffmpeg")
        .status()
        .unwrap();
    assert!(st.success());
    let b = std::fs::read(tmp("env_ffmpeg.flac")).unwrap();
    let vendor = b"tapespeed";
    assert!(
        !b.windows(vendor.len()).any(|w| w == vendor),
        "native encoder used"
    );
    assert!(b.windows(4).any(|w| w == b"Lavf"), "not written by ffmpeg");
}

#[test]
fn inside_a_one_thread_pool() {
    // writer driven from a rayon worker with more batches than the in-flight bound: must not deadlock
    let x = signal(12 * 16 * BLOCK + 5, 2, 11);
    let a = tmp("pool1.flac");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| save(&a, &x, 2, 44100, None)).unwrap();
    let d = decode_s32(&a);
    assert_eq!(d.len(), x.len() * 4);
    assert!(
        d.chunks(4)
            .zip(&x)
            .all(|(s, &v)| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) == q24(v) << 8)
    );
    flac_test(&a);
}

#[test]
fn noise_silence_and_steps() {
    // uncorrelated full-scale noise (verbatim / independent), a silent channel next to a loud one,
    // a square wave (fixed predictors), bursts (Rice escapes into high parameters)
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let n = 3 * BLOCK + 99;
    let noise: Vec<f32> = (0..2 * n).map(|_| rnd() as f32).collect();
    same_as_ffmpeg_x("noise", &noise, 2, 48000, 5000);
    let half: Vec<f32> = (0..2 * n)
        .map(|i| if i % 2 == 0 { 0.0 } else { 0.9 * rnd() as f32 })
        .collect();
    same_as_ffmpeg_x("half", &half, 2, 44100, 5000);
    let sq: Vec<f32> = (0..n)
        .map(|i| if (i / 50) % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    same_as_ffmpeg_x("square", &sq, 1, 44100, 5000);
    let burst: Vec<f32> = (0..2 * n)
        .map(|i| {
            let quiet = (i / 700) % 5 != 0;
            (rnd() * if quiet { 1e-6 } else { 0.99 }) as f32
        })
        .collect();
    same_as_ffmpeg_x("burst", &burst, 2, 44100, 5000);
}
