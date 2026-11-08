//! Stage profiler for benchmarks, active only with `TAPESPEED_PROF=1` (one stderr line per
//! [`mark`]); outputs are unaffected. Per stage: wall time, process CPU (user + sys, all threads),
//! RSS and phys_footprint peaks (2 ms sampler), and the live-heap peak when the binary installs
//! [`Counting`] as its global allocator (`tapespeed` feature `memprof`; otherwise 0).
//! A mark closes the interval since the previous mark. Meant for single-file runs (batch jobs
//! would interleave their marks).
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static RSS_PK: AtomicU64 = AtomicU64::new(0);
static FP_PK: AtomicU64 = AtomicU64::new(0);

/// Counting allocator (live bytes + peak since the last mark).
pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(c, Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(c, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            if n >= l.size() {
                let c = CUR.fetch_add(n - l.size(), Relaxed) + (n - l.size());
                PEAK.fetch_max(c, Relaxed);
            } else {
                CUR.fetch_sub(l.size() - n, Relaxed);
            }
        }
        q
    }
}

struct State {
    t0: Instant,
    t: Instant,
    cpu: f64,
}

fn state() -> Option<&'static Mutex<State>> {
    static ST: OnceLock<Option<Mutex<State>>> = OnceLock::new();
    ST.get_or_init(|| {
        let on = std::env::var_os("TAPESPEED_PROF").is_some_and(|v| !v.is_empty() && v != "0");
        if !on {
            return None;
        }
        std::thread::spawn(|| {
            loop {
                let (rss, fp) = mem_now();
                RSS_PK.fetch_max(rss, Relaxed);
                FP_PK.fetch_max(fp, Relaxed);
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        let now = Instant::now();
        Some(Mutex::new(State {
            t0: now,
            t: now,
            cpu: cpu_s(libc::RUSAGE_SELF),
        }))
    })
    .as_ref()
}

/// True with `TAPESPEED_PROF` set (non-empty, not `0`).
pub fn enabled() -> bool {
    state().is_some()
}

/// Start the clock (first call wins); later marks measure from here.
pub fn start() {
    let _ = state();
}

fn cpu_s(who: libc::c_int) -> f64 {
    let mut r: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(who, &mut r) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(r.ru_utime) + tv(r.ru_stime)
}

/// (resident bytes, phys_footprint bytes) of this process.
#[cfg(target_os = "macos")]
fn mem_now() -> (u64, u64) {
    let mut ri: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            &mut ri as *mut libc::rusage_info_v4 as *mut libc::rusage_info_t,
        )
    };
    if rc == 0 {
        (ri.ri_resident_size, ri.ri_phys_footprint)
    } else {
        (0, 0)
    }
}

#[cfg(not(target_os = "macos"))]
fn mem_now() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let rss = s
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    (rss * 4096, 0)
}

/// Close the stage `name` (interval since the previous mark) and print its line.
pub fn mark(name: &str) {
    let Some(st) = state() else { return };
    let mut s = st.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let cpu = cpu_s(libc::RUSAGE_SELF);
    let (rss, fp) = mem_now();
    let rss_pk = RSS_PK.swap(rss, Relaxed).max(rss);
    let fp_pk = FP_PK.swap(fp, Relaxed).max(fp);
    let cur = CUR.load(Relaxed);
    let heap_pk = PEAK.swap(cur, Relaxed).max(cur);
    let dt = (now - s.t).as_secs_f64();
    let dc = cpu - s.cpu;
    const G: f64 = 1073741824.0;
    eprintln!(
        "prof {name:<22} wall {dt:8.3} s  cpu {dc:8.2} s ({:4.1}x)  heap_pk {:6.2} GiB  heap {:6.2} GiB  rss_pk {:6.2} GiB  fp_pk {:6.2} GiB  t {:7.2} s",
        if dt > 0.0 { dc / dt } else { 0.0 },
        heap_pk as f64 / G,
        cur as f64 / G,
        rss_pk as f64 / G,
        fp_pk as f64 / G,
        (now - s.t0).as_secs_f64()
    );
    s.t = now;
    s.cpu = cpu;
}

/// Final line: whole-run totals incl. the CPU of waited-for children (ffmpeg decode/encode).
pub fn end() {
    let Some(st) = state() else { return };
    let s = st.lock().unwrap_or_else(|e| e.into_inner());
    eprintln!(
        "prof END wall {:.3} s  self cpu {:.2} s  children cpu (ffmpeg) {:.2} s",
        s.t0.elapsed().as_secs_f64(),
        cpu_s(libc::RUSAGE_SELF),
        cpu_s(libc::RUSAGE_CHILDREN)
    );
}
