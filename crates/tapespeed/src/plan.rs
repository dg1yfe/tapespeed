//! Memory planner (docs/MEMORY.md §4): per-file working-set estimates from the decimated sample
//! counts, a budget from the machine, the mode ladder RAM -> spill -> budgeted bw candidates,
//! and the shared budget of a batch. Throughput first: everything runs in RAM and in parallel
//! while it fits; the fallbacks only engage when the estimate exceeds the budget.
use std::sync::{Condvar, Mutex};

use ts_core::analyze::api::{
    CAND_EXTRA_B, CAND_KEEP_B, CAND_SCRATCH_B, CandBudget, STAGE1_B, Z_B, sizes,
};
use ts_io::machine::Machine;

use crate::cli::Cli;
use crate::pipeline::{Fail, stage1_passes};

const GIB: f64 = (1u64 << 30) as f64;
/// Fixed process base [bytes].
pub const BASE: u64 = 100 << 20;
/// Other stages (pair, aided, evidence, correct) per second of audio [bytes].
pub const OTHER_PER_S: f64 = 180e3;
/// Slack on every estimate (allocator, coefficient spread).
pub const SLACK: f64 = 1.1;
/// Default budget: this fraction of the available RAM.
pub const AVAIL_FRAC: f64 = 0.8;

fn slack(b: u64) -> u64 {
    (b as f64 * SLACK) as u64
}

pub fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / GIB)
}

/// `--max-ram`: `12G`, `512M`, `1.5T`, `123456` (bytes) or `50%` (of the available RAM).
pub fn parse_size(s: &str, avail: u64) -> Result<u64, String> {
    let t = s.trim();
    let bad = || format!("--max-ram: cannot parse {s:?} (e.g. 12G, 512M, 50%)");
    if let Some(p) = t.strip_suffix('%') {
        let v: f64 = p.trim().parse().map_err(|_| bad())?;
        return (v > 0.0)
            .then(|| (avail as f64 * v / 100.0) as u64)
            .ok_or_else(bad);
    }
    let (num, mul) = match t.char_indices().last().map(|(_, c)| c.to_ascii_uppercase()) {
        Some('K') => (&t[..t.len() - 1], 1u64 << 10),
        Some('M') => (&t[..t.len() - 1], 1 << 20),
        Some('G') => (&t[..t.len() - 1], 1 << 30),
        Some('T') => (&t[..t.len() - 1], 1 << 40),
        _ => (t, 1),
    };
    let v: f64 = num.trim().parse().map_err(|_| bad())?;
    (v > 0.0).then_some((v * mul as f64) as u64).ok_or_else(bad)
}

/// Budget B [bytes]: 80 % of available by default; an explicit `--max-ram` sets B directly (it
/// may lower or raise it), capped only at physical RAM and the cgroup headroom. Available is
/// already capped by the cgroup headroom. Returns (B, how it was derived).
pub fn budget_of(m: &Machine, max_ram: Option<&str>) -> Result<(u64, String), String> {
    if m.avail == 0 {
        // RAM unknown (unsupported OS): no automatic limit
        let b = match max_ram {
            Some(s) => parse_size(s, 0)?,
            None => u64::MAX >> 2,
        };
        return Ok((b, "RAM unknown".into()));
    }
    let def = (m.avail as f64 * AVAIL_FRAC) as u64;
    let why = |b: u64, src: &str| (b, format!("{src}, {} available", gib(m.avail)));
    Ok(match max_ram {
        Some(s) => {
            let cap = m.cgroup.map_or(m.total, |c| c.min(m.total));
            let v = parse_size(s, m.avail)?;
            if v <= cap {
                why(v, "--max-ram")
            } else {
                why(cap, "--max-ram capped at physical RAM / cgroup")
            }
        }
        None => why(def, "80 %"),
    })
}

/// Working-set estimate of one file [bytes].
#[derive(Debug, Clone, PartialEq)]
pub struct Est {
    /// decoded input (f32)
    pub x: u64,
    /// stage-1 baseband + coarse track + what each candidate keeps after its grid, + the
    /// basebands of later passes (streamed during the decode)
    pub fixed: u64,
    /// scratch of the largest bw candidate running alone
    pub own: u64,
    /// extra per further concurrent candidate, largest first (`cand[0]` = the largest's)
    pub cand: Vec<u64>,
    /// the other stages (max of pair, aided, evidence, correct)
    pub oth: u64,
}

/// [`Est`] for `frames` frames of `ch` channels at `sr`, for the analyze pass(es) of `a`.
pub fn estimate(frames: u64, ch: usize, sr: u32, a: &Cli) -> Result<Est, Fail> {
    let (o, refs) = stage1_passes(a)?;
    let sz: Vec<(u64, Vec<u64>)> = refs
        .iter()
        .map(|&r| sizes(frames as usize, sr as f64, r, &o))
        .collect::<Result<_, _>>()?;
    // the analyze passes run one after another: the larger one counts. All stage-1 basebands
    // are streamed during the decode, so the later passes' z are resident during the earlier ones
    let mut best: Option<(u64, Vec<u64>)> = None;
    for (i, (nd, nk)) in sz.iter().enumerate() {
        let mut nk = nk.clone();
        nk.sort_unstable_by(|p, q| q.cmp(p));
        let later: u64 = sz[i + 1..].iter().map(|(n, _)| Z_B * n).sum();
        let fixed = STAGE1_B * nd + later + CAND_KEEP_B * nk.iter().sum::<u64>();
        if best
            .as_ref()
            .is_none_or(|(f, n)| fixed + CAND_SCRATCH_B * nk[0] > f + CAND_SCRATCH_B * n[0])
        {
            best = Some((fixed, nk));
        }
    }
    let (fixed, nk) = best.expect("at least one analyze pass");
    Ok(Est {
        x: 4 * frames * ch as u64,
        fixed,
        own: CAND_SCRATCH_B * nk[0],
        cand: nk.iter().map(|n| CAND_EXTRA_B * n).collect(),
        oth: (OTHER_PER_S * frames as f64 / sr as f64) as u64,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// decoded input in RAM
    Ram,
    /// decoded input in an unlinked, memory-mapped temp file
    Spill,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub mode: Mode,
    /// reservation for the whole job: base + input (RAM) + one bw candidate (or the other stages)
    pub hold: u64,
    /// expected bw candidates in parallel (of `n_cand`)
    pub par: usize,
    pub n_cand: usize,
    /// expected peak
    pub peak: u64,
    /// even one candidate at a time exceeds the budget (runs anyway)
    pub over: bool,
}

/// The ladder: RAM with all candidates in parallel, spill with all in parallel, spill with as
/// many candidates as the budget holds (largest first; at least one). `spill_ok`: the temp dir
/// has room for the input.
pub fn decide(e: &Est, budget: u64, spill_ok: bool) -> Plan {
    let one = e.own;
    let all = one + e.cand.iter().skip(1).sum::<u64>();
    let peak = |x: u64, an: u64| slack(BASE + x + (e.fixed + an).max(e.oth));
    let n_cand = e.cand.len();
    for (mode, x) in [(Mode::Ram, e.x), (Mode::Spill, 0)] {
        if (mode == Mode::Ram || spill_ok) && peak(x, all) <= budget {
            return Plan {
                mode,
                hold: peak(x, one),
                par: n_cand,
                n_cand,
                peak: peak(x, all),
                over: false,
            };
        }
    }
    let (mode, x) = if spill_ok {
        (Mode::Spill, 0)
    } else {
        (Mode::Ram, e.x)
    };
    let hold = peak(x, one);
    let (mut par, mut used) = (1, hold);
    for &c in &e.cand[1.min(n_cand)..] {
        if used + slack(c) > budget {
            break;
        }
        used += slack(c);
        par += 1;
    }
    Plan {
        mode,
        hold,
        par: par.min(n_cand),
        n_cand,
        peak: used,
        over: hold > budget,
    }
}

/// Batch jobs in parallel: at most ⌈threads / 2.5⌉ (one job keeps ~3.7 cores busy).
pub fn max_jobs(threads: usize, files: usize) -> usize {
    (threads as f64 / 2.5)
        .ceil()
        .max(1.0)
        .min(files.max(1) as f64) as usize
}

/// One line for stdout (`memory: ...`; machine-dependent, so end-to-end comparisons filter it).
pub fn line(p: &Plan, budget: u64, why: &str, jobs: usize, threads: usize, tmp: &str) -> String {
    let mode = match p.mode {
        Mode::Ram => "ram".to_string(),
        Mode::Spill => format!("spill ({tmp})"),
    };
    format!(
        "memory: mode {mode}, bw {}/{} parallel, est peak {}, budget {} ({why}), jobs {jobs}, threads {threads}{}",
        p.par,
        p.n_cand,
        gib(p.peak),
        gib(budget),
        if p.over {
            "; WARNING: one bw candidate at a time already exceeds the budget"
        } else {
            ""
        }
    )
}

/// The shared budget of a run (bytes). Job reservations block; bw candidates beyond a job's
/// first only take what is free now (never block), so jobs cannot deadlock each other. Waits
/// happen on job driver threads only, never on rayon workers.
pub struct Budget {
    pub cap: u64,
    used: Mutex<u64>,
    cv: Condvar,
}

pub struct Lease<'a> {
    b: &'a Budget,
    n: u64,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.b.release(self.n);
    }
}

impl Budget {
    pub fn new(cap: u64) -> Self {
        Budget {
            cap,
            used: Mutex::new(0),
            cv: Condvar::new(),
        }
    }
    /// Wait until `n` fits (or nothing else is reserved: an oversized job runs alone).
    pub fn acquire(&self, n: u64) -> Lease<'_> {
        let mut u = self.used.lock().unwrap();
        while *u > 0 && *u + n > self.cap {
            u = self.cv.wait(u).unwrap();
        }
        *u += n;
        Lease { b: self, n }
    }
    pub fn used(&self) -> u64 {
        *self.used.lock().unwrap()
    }
    fn release(&self, n: u64) {
        let mut u = self.used.lock().unwrap();
        *u = u.saturating_sub(n);
        drop(u);
        self.cv.notify_all();
    }
}

impl CandBudget for Budget {
    fn try_take(&self, bytes: u64) -> bool {
        let n = slack(bytes);
        let mut u = self.used.lock().unwrap();
        let ok = *u + n <= self.cap;
        if ok {
            *u += n;
        }
        ok
    }
    fn give(&self, bytes: u64) {
        self.release(slack(bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;

    fn cli(args: &[&str]) -> Cli {
        let mut v = vec!["tapespeed", "in.wav", "--ref", "9000", "-o", "o.flac"];
        v.extend_from_slice(args);
        crate::cli::parse_from(v)
    }

    /// 2 h stereo 44.1 kHz, default args.
    fn est2h() -> Est {
        estimate(7200 * 44100, 2, 44100, &cli(&[])).unwrap()
    }

    #[test]
    fn estimate_matches_measured_2h() {
        let e = est2h();
        assert_eq!(e.x, 4 * 7200 * 44100 * 2);
        assert_eq!(e.cand.len(), 5);
        assert!(e.cand.windows(2).all(|w| w[0] >= w[1]), "largest first");
        // MEMORY.md §2 (search 0.03): RAM 5 parallel measured 11.6 GiB, spill + lean 1 at a
        // time 5.3 GiB; the plan must not under-estimate them
        let ram = decide(&e, u64::MAX, true);
        assert_eq!((ram.mode, ram.par), (Mode::Ram, 5));
        let pk = ram.peak as f64 / G as f64;
        assert!((11.6..15.0).contains(&pk), "{pk}");
        let one = slack(BASE + e.fixed + e.own) as f64 / G as f64;
        assert!((5.3..7.0).contains(&one), "{one}");
        // a `line:` second pass at 15625 Hz has more stage-1 samples and dominates
        let el = estimate(7200 * 44100, 2, 44100, &cli(&["--check-line", "15625"])).unwrap();
        assert!(el.fixed > e.fixed);
    }

    #[test]
    fn ladder_each_mode() {
        let e = est2h();
        let all: u64 = e.own + e.cand[1..].iter().sum::<u64>();
        let pk = |x: u64, an: u64| slack(BASE + x + (e.fixed + an).max(e.oth));
        // fits in RAM
        let p = decide(&e, pk(e.x, all), true);
        assert_eq!((p.mode, p.par, p.over), (Mode::Ram, 5, false));
        // one byte less: spill, still all parallel
        let p = decide(&e, pk(e.x, all) - 1, true);
        assert_eq!((p.mode, p.par), (Mode::Spill, 5));
        // no temp space: RAM with fewer candidates
        let p = decide(&e, pk(e.x, all) - 1, false);
        assert_eq!(p.mode, Mode::Ram);
        assert!(p.par < 5 && p.par >= 1);
        // spill, budget for the first two candidates only
        let b = pk(0, e.own) + slack(e.cand[1]);
        let p = decide(&e, b, true);
        assert_eq!((p.mode, p.par, p.over), (Mode::Spill, 2, false));
        assert!(p.peak <= b);
        // tiny budget: one at a time, flagged
        let p = decide(&e, G, true);
        assert_eq!((p.mode, p.par, p.over), (Mode::Spill, 1, true));
    }

    #[test]
    fn sizes_and_budget_parse() {
        assert_eq!(parse_size("12G", 0), Ok(12 * G));
        assert_eq!(parse_size("512m", 0), Ok(512 << 20));
        assert_eq!(parse_size("1.5T", 0), Ok(3 << 39));
        assert_eq!(parse_size("1000", 0), Ok(1000));
        assert_eq!(parse_size("50%", 10 * G), Ok(5 * G));
        assert!(parse_size("x", 0).is_err() && parse_size("-1G", 0).is_err());
        let m = Machine {
            total: 64 * G,
            avail: 50 * G,
            cgroup: None,
            cores: 10,
            p_cores: Some(8),
            e_cores: Some(2),
        };
        assert_eq!(budget_of(&m, None).unwrap().0, 40 * G);
        assert_eq!(budget_of(&m, Some("8G")).unwrap().0, 8 * G);
        assert_eq!(budget_of(&m, Some("60G")).unwrap().0, 60 * G); // explicit raise
        assert_eq!(budget_of(&m, Some("80G")).unwrap().0, 64 * G); // capped at physical
        let mc = Machine {
            cgroup: Some(20 * G),
            ..m
        };
        assert_eq!(budget_of(&mc, Some("30G")).unwrap().0, 20 * G); // capped at cgroup
        assert_eq!(max_jobs(10, 17), 4);
        assert_eq!(max_jobs(10, 2), 2);
        assert_eq!(max_jobs(1, 5), 1);
    }

    #[test]
    fn budget_leases_and_candidates() {
        let b = Budget::new(1000);
        let l1 = b.acquire(600);
        assert!(!b.try_take(400), "400 * 1.1 > 400 free");
        assert!(b.try_take(300));
        assert_eq!(b.used(), 930);
        b.give(300);
        drop(l1);
        assert_eq!(b.used(), 0);
        // an oversized job runs alone
        let l = b.acquire(5000);
        assert_eq!(b.used(), 5000);
        drop(l);
        // a waiting job is admitted when the other finishes
        let l1 = b.acquire(800);
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                let _l2 = b.acquire(500);
                b.used()
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert_eq!(b.used(), 800);
            drop(l1);
            assert_eq!(h.join().unwrap(), 500);
        });
        assert_eq!(b.used(), 0);
    }
}
