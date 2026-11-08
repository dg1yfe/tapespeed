//! Pass-2 bench: `cargo run --release -p ts-core --example bench_correct -- SECONDS [CH] [SR] [post]`
//! Synthetic input + wow curve; streaming run() into a checksum sink (no output buffer).
//! `post` adds the reference notch (19 kHz) and an 8 kHz lowpass. Thread count: RAYON_NUM_THREADS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

/// Heap accounting: live and peak bytes (reported next to RSS, which on macOS also counts
/// MADV_FREE'd pages the allocator has released but the kernel not yet reclaimed).
struct Count;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        let v = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(v, Relaxed);
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
}
#[global_allocator]
static A: Count = Count;

fn mb(v: usize) -> f64 {
    v as f64 / 1e6
}
use ts_core::correct::{self, Lowpass, Params, Post};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let secs: f64 = a.get(1).map_or(60.0, |s| s.parse().unwrap());
    let ch: usize = a.get(2).map_or(2, |s| s.parse().unwrap());
    let sr: u32 = a.get(3).map_or(44100, |s| s.parse().unwrap());
    let post = if a.get(4).map(String::as_str) == Some("post") {
        Post {
            notches: vec![(19000.0, 30.0)],
            lowpass: Some(Lowpass::new(8000.0)),
        }
    } else {
        Post::default()
    };
    let n = (secs * sr as f64) as usize;
    let mut x = vec![0.0f32; n * ch];
    // two rotating phasors per channel (cheap, deterministic)
    for c in 0..ch {
        let (w1, w2) = (
            2.0 * std::f64::consts::PI * (440.0 + 100.0 * c as f64) / sr as f64,
            2.0 * std::f64::consts::PI * 9000.0 / sr as f64,
        );
        let (mut p1, mut p2) = (0.0f64, c as f64);
        for i in 0..n {
            if i % 4096 == 0 {
                p1 = w1 * i as f64;
                p2 = w2 * i as f64 + c as f64;
            }
            x[i * ch + c] = (0.3 * p1.sin() + 0.1 * p2.sin()) as f32;
            p1 += w1;
            p2 += w2;
        }
    }
    if a.get(4).map(String::as_str) == Some("gen") {
        println!("input only: {} MB", x.len() * 4 / 1_000_000);
        return;
    }
    let t: Vec<f64> = (0..=(secs / 0.005) as usize)
        .map(|i| i as f64 * 0.005)
        .collect();
    let r: Vec<f64> = t
        .iter()
        .map(|&v| 1.0 + 0.004 * (v * 3.1).sin() + 0.001 * (v * 0.07).sin())
        .collect();
    if a.get(4).map(String::as_str) == Some("setup") {
        let r = correct::Resampler::new(&x, ch, sr, &t, &r, &Params::default()).unwrap();
        println!(
            "setup only: n_out {}, heap peak above input+curve {:.1} MB, live {:.1} MB",
            r.n_out(),
            mb(PEAK.load(Relaxed) - x.len() * 4 - t.len() * 16),
            mb(LIVE.load(Relaxed) - x.len() * 4 - t.len() * 16)
        );
        return;
    }
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let t0 = Instant::now();
    let mut calls = 0usize;
    let mut hash = 0xcbf29ce484222325u64; // FNV-1a over output bits

    let frames = correct::run_arrays(
        &x,
        ch,
        sr,
        &t,
        &r,
        &Params::default(),
        &post,
        |b: &[f32]| {
            calls += 1;
            if std::env::var("RSS_TRACE").is_ok() && calls % 20 == 1 {
                let o = std::process::Command::new("ps")
                    .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                    .output()
                    .unwrap();
                eprintln!(
                    "call {calls}: rss {} MB",
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<u64>()
                        .unwrap_or(0)
                        / 1024
                );
            }
            for v in b {
                hash = (hash ^ v.to_bits() as u64).wrapping_mul(0x100000001b3);
            }
            Ok::<(), ()>(())
        },
    )
    .unwrap();
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "heap peak above input+curve during run: {:.1} MB",
        mb(PEAK.load(Relaxed) - base)
    );
    println!(
        "{secs} s x {ch} ch @ {sr}: {frames} frames in {dt:.3} s ({:.2} s/h audio), threads {}, output hash {hash:016x}",
        dt * 3600.0 / secs,
        rayon::current_num_threads()
    );
}
