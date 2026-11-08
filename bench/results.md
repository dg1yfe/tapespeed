# Benchmarks

Wall time, CPU, memory and per-stage profile of `tapespeed`, measured on one machine. min/h = median wall time per
hour of audio. §9 describes the regression guard `bench/check.sh`.

## 1. Environment and load

| item | value |
|---|---|
| machine | Apple M1 Max, 8 P + 2 E cores, 64 GiB, macOS 15.7.4 |
| build | rustc 1.97.1, `--release` (thin LTO, cgu 1, `target-cpu=apple-m1`) |
| ffmpeg | 9.0.2 (MP3 decode; FLAC encode in §2–§6, see §7) |
| inputs | a 300 s mono f32 WAV excerpt; a 2601.2 s (43.35 min) stereo MP3 aircheck; 17 MP3 airchecks, 66 243 s (18.40 h) total |
| method | `/usr/bin/time -l` per run, N = 3, sequential; 1-min load average and the top CPU processes logged before each run |
| load, measured | 1-min load average 3.1–10.7 during all runs (batch runs: 10–22, from the 4 parallel jobs); ≈ 0.5–1 core busy in the background throughout |

Spread across N = 3: wall ≤ 4 % (max/min), most ≤ 2 %.

## 2. Macro (default threads = 10)

CPU = user + sys incl. ffmpeg children.

| run | wall min / median s | CPU s | min/h |
|---|---|---|---|
| 5 min WAV `--ref 9000 --abs line:15625` | 2.57 / 2.64 | 9.6 | 0.528 |
| 43 min `--ref 9000` | 18.67 / 19.36 | 65.0 | 0.447 |
| 43 min `--ref 9000 --abs line:15625 --lowpass` | 26.63 / 26.88 | 89.7 | 0.620 |
| 43 min `--ref 9000 --abs pair --auto-notch` | 29.35 / 30.01 | 130.1 | 0.692 |
| `--survey`, 17 files (18.4 h) | 19.05 / 19.59 | 107.0 | 0.0177 |
| batch 17 files `--ref 9000 --auto-notch` | 265.8 / 280.3 | 2206 | 0.254 |

Survey and batch are one multi-input process (`tapespeed F1 … F17 --survey -o DIR`, resp. `… --ref 9000 --auto-notch
-o DIR`); batch: planner 4 jobs in parallel, 10 threads; survey: files one after another, each survey parallel
inside (§8).

## 3. Thread scaling, 43 min `--ref 9000 --abs pair --auto-notch`

| `--threads` | wall min / median s | min/h | speedup vs 1 | efficiency | CPU s | max RSS GiB |
|---|---|---|---|---|---|---|
| 1 | 103.32 / 103.54 | 2.388 | 1.00 | 100 % | 109.6 | 3.45 |
| 2 | 59.50 / 59.80 | 1.379 | 1.73 | 87 % | 113.6 | 4.52 |
| 4 | 38.83 / 39.17 | 0.903 | 2.64 | 66 % | 117.4 | 5.52 |
| 8 | 30.08 / 30.13 | 0.695 | 3.44 | 43 % | 122.1 | 5.57 |
| 10 (default) | 28.97 / 29.20 | 0.674 | 3.55 | 35 % | 132.4 | 5.43 |

Also `--ref 9000`: threads 1 → 48.92 s (1.128 min/h), 8 → 18.86 s (0.435), 10 → 19.36 s (0.447; the 8/10 difference
is within noise). `line:` + lowpass: threads 1 → 71.25 s (1.643 min/h).
Bit identity: curve CSV and decoded output audio identical for threads 1/2/4/8/10.
E-core effect: 8 → 10 threads gains 3 % wall (pair) and nothing for `ref`, at +8 % CPU. The run is not limited by core
count above 4 threads (§4: ~5.5 s of the 29 s were single-threaded ffmpeg, and the stages that parallelise reach
2–6×). macOS gives no CPU affinity: `--threads 8` does not pin to P cores.
`--threads N` caps both the rayon pool and the number of concurrently running bw candidates, so `--threads 1` really
uses one core.

## 4. Per-stage profile, 43 min (`TAPESPEED_PROF=1`)

Wall/CPU: median of 3 profiled runs of the normal binary (profiler overhead: 18.83 vs 19.36 s unprofiled, within noise).
Live heap: one run of the `memprof` build (counting allocator, +1.5 % time). RSS peak: 2 ms sampler. ∥ = CPU/wall.
"pipe" = resampler + notch/lowpass chain + encoder. Measured with the ffmpeg FLAC encoder (§7 for the in-process one).

`--ref 9000 --abs pair --auto-notch` (incl. a `--threads 1` run):

| stage | wall s | % | CPU s | ∥ | 1 thread wall s | speedup | live heap peak GiB | RSS peak GiB |
|---|---|---|---|---|---|---|---|---|
| load (ffmpeg MP3 decode → f32) | 2.42 | 8 | 0.4 (+ffmpeg) | – | 2.43 | 1.0 | 0.86 | 0.86 |
| an.stage1 (het_decimate) | 0.30 | 1 | 2.8 | 9.2 | 2.02 | 6.7 | 0.94 | 0.94 |
| an.coarse (Viterbi) | 0.27 | 1 | 1.1 | 4.1 | 0.90 | 3.3 | 0.98 | 1.07 |
| an.tracks (5 bw candidates) | 3.45 | 12 | 14.6 | 4.2 | 12.81 | 3.7 | **4.28** | 5.32 |
| an.select | 0.04 | 0 | 0.1 | – | 0.04 | – | 1.08 | 3.73 |
| pair.measure | 1.30 | 4 | 11.0 | 8.5 | 8.55 | 6.6 | 1.26 | 4.08 |
| pair.switch_repair | 0.50 | 2 | 4.4 | 8.8 | 3.56 | 7.1 | 0.97 | 4.07 |
| pair.remeasure | 1.27 | 4 | 10.7 | 8.5 | 8.52 | 6.7 | 1.27 | 4.07 |
| model.fit (identify, L1–L3) | 1.51 | 5 | 5.3 | 3.5 | 4.09 | 2.7 | 1.00 | 4.09 |
| model.aided | 3.30 | 11 | 7.2 | 2.2 | 6.20 | 1.9 | 1.15 | 4.12 |
| evidence (fading, coherence) | 0.77 | 3 | 3.9 | 5.1 | 3.21 | 4.2 | 1.00 | 3.92 |
| L1 | 0.00 | 0 | 0.0 | – | 0.00 | – | 0.90 | 3.89 |
| curve CSV + plots | 0.32 | 1 | 0.3 | 1.0 | 0.31 | 1.0 | 0.95 | 3.89 |
| auto-notch pass 1: detect (correct → detector) | 3.20 | 11 | 20.4 | 6.4 | 17.43 | 5.4 | 1.05 | 3.92 |
| auto-notch pass 2: keying (correct → keying stats) | 3.66 | 13 | 20.6 | 5.6 | 16.41 | 4.5 | 1.11 | 3.93 |
| final correct + notches (pipe) | 6.42 | 22 | 17.8 | 2.8 | 17.52 | 2.7 | 1.00 | 3.93 |
| encode finish (ffmpeg flush) | 0.19 | 1 | 0.2 | – | 0.18 | – | 0.91 | 3.93 |
| **total** | **29.03** | 100 | 121.8 + ffmpeg 10.5 | 4.2 | 104.20 | 3.6 | **4.28** | **5.32** |

`--ref 9000` (total 18.75 s, CPU 52.2 + ffmpeg 10.5 s): load 2.44 · an.stage1 0.30 · an.coarse 0.28 · an.tracks 3.49 ·
model.fit 1.64 · model.aided 3.29 · evidence 0.77 · curve+plot 0.31 · final correct + notch (pipe) 5.94 · encode
flush 0.19. Live heap peak 4.25 GiB (tracks), RSS peak 5.23 GiB.
`--ref 9000 --abs line:15625 --lowpass` (total 26.04 s): as `ref` plus the second analyze (stage1 0.32, coarse 0.99,
tracks 5.96 s; line candidates at fs_p up to fs_d run longer) and lowpass in the final pipe (6.05 s). Live heap peak
**5.70 GiB** (second-pass tracks: the 15.6 kHz line uses wider basebands), RSS peak 7.83 GiB.

Where the final pipe goes (`--ref 9000`, ffmpeg FLAC encoder, 3 runs each):

| output | final pipe wall s | total wall median s | min/h |
|---|---|---|---|
| `.flac` | 5.94 | 19.36 | 0.447 |
| `.flac --keep-ref` (no notch) | 5.73 | 18.56 | 0.428 |
| `.wav` (f32, no encoding) | 2.70 | 15.50 | 0.358 |
| `.wav --keep-ref` | 2.31 | 15.09 | 0.348 |

With ffmpeg's single-threaded FLAC encoder the final pass was bounded by encoding (≈ 3.2 s of 5.9 s); the notch costs
≈ 0.4 s. This led to the in-process parallel encoder (§7).

## 5. Micro (criterion, median)

Benches: `crates/ts-core/benches/m3_micro.rs`, `crates/ts-dsp/benches/b_prims.rs` (deterministic synthetic inputs
generated in the benches). The table was measured with an earlier input set of the same sizes; the current inputs give
correct 49.1, het_decimate 46.1, PLL 79.6, welch 2.45, stft 23.5 ms at 1t (`bench/baseline.json`).
1t = 1-thread rayon pool, mt = 10 threads.

| kernel | size | 1t ms | mt ms | per hour 44.1 kHz stereo, 1t |
|---|---|---|---|---|
| `correct` (resampler, 64 taps, no post) | 10 s stereo | 48.8 | 11.1 | 17.6 s |
| `het_decimate` (stage 1: D 27, 731 taps, mid) | 60 s stereo | 46.2 | 8.6 | 2.8 s per pass |
| PLL (scalar by design) | 1 M samples | 96.1 | – | 96 ns/iteration; 34 M iterations ≈ 3.3 s |
| `fit_hier` + `explained` (mech preset, Ta 5 s) | 720 s curve (144 k pts) | 946.6 | 324.5 | – |
| welch, nperseg 4096 | 512 k | 2.47 | 0.51 | – |
| stft, nperseg 4096, hop 2048 (bin-major layout) | 60 s × 44.1 kHz | 24.1 | 10.6 | – |
| stft, frame-major layout (what the callers use) | same | 15.6 | 5.7 | – |

Notes: `correct` measures 1.2 G channel·tap MACs/s per core (gathers + linear interpolation of two table rows per
output). The PLL computes atan2 + cos/sin + hypot per iteration in a fixed order; it is inherently sequential. The
bin-major stft layout needs a transpose and is slower; the pipeline uses the frame-major variant.

## 6. Memory

| run | live heap peak GiB | max RSS GiB | peak footprint GiB |
|---|---|---|---|
| 5 min `line:` (mono) | – | 1.25 | 0.27 |
| 43 min `ref` | 4.25 | 5.37 | 2.30 |
| 43 min `line:` + lowpass | 5.70 | 7.77 | 2.80 |
| 43 min `pair` + auto-notch | 4.28 | 5.55 | 2.25 |
| 43 min `pair`, `--threads 1` | – | 3.45 | 1.80 |
| survey 17 files | – | 1.24 | 0.26 |
| batch 17 files (4 jobs ∥) | – | 24.10 | 11.20 |

Planner estimate (`memory:` line) for 43 min pair: 5.3 GiB vs 4.28 GiB live heap measured (+24 %, conservative).
Footprint under-reports (docs/MEMORY.md §1); live heap is the planning metric, RSS ≥ heap in every run. `--threads 1`
also runs the candidates one by one, hence the lower peak.

## 7. In-process parallel FLAC encoder

`.flac` outputs are encoded in-process (`ts_io::flac`: own frame coder, 16-block batches as rayon jobs, an output
thread for file + MD5, bounded in flight); `TAPESPEED_FLAC=ffmpeg` uses ffmpeg's encoder instead. Measured on the
43 min MP3 with `TAPESPEED_PROF=1`, default plots, ffmpeg encoder ("before") and in-process encoder ("after")
interleaved, 1-min load average 8–44 (pairs under comparable load are meaningful, single numbers are noisy):

| mode | threads | final pipe wall s, before → after | total CPU s (incl. ffmpeg), before → after |
|---|---|---|---|
| `ref` | 10 | 6.7–10.8 → 4.3–6.7 | 66.6–70.0 → 62.9–65.9 |
| `ref` | 8 | 7.1–11.1 → 6.2–6.3 | 65.0–68.8 → 62.4–65.4 |
| `ref` | 1 | 23.4 → 19.3 (comparable-load pair) | 65.1 → 61.9 |
| `pair` | 10 | 7.6–7.7 → 5.5 | 131.5–135.3 → 133.6 |
| `pair` | 8 | 6.8–7.3 → 5.4 | 130.1–131.0 → 128.2 |
| `pair` | 1 | 20.1–21.3 → 19.9 | 123.4–124.5 → 121.9 |

- Final pass at 8–10 threads −17…−53 % per pair; expected total gain ≈ 2.5–3 s per 43 min on an idle machine (the
  WAV-output bound of §4).
- Total CPU drops 3–4 s per run: the coder (2.9 s per 43 min on one core) is cheaper than ffmpeg's encoder plus the
  f32 pipe.
- Decoded output bit-identical to the ffmpeg path in every run; STREAMINFO MD5 equal to ffmpeg's; `flac -t` clean;
  ffprobe `flac, s32, 24 bit`. Size −0.06…−0.07 %. RSS unchanged (in flight: ≤ 2 × threads batches of 64 k frames).

## 8. Survey with several inputs: sequential vs parallel

`tapespeed F1 … F17 --survey -o DIR` (17 MP3s, 18.4 h), 10 threads, load average 13–18, wall / user+sys s:

| variant | run 1 | run 2 |
|---|---|---|
| sequential (current) | 11.16 / 54.9 | 10.93 / 54.9 |
| 4 surveys at once (`ceil(threads/2.5)`) | 14.64 / 94.8 | 11.57 / 71.0 |
| parallel variant with `--threads 2` (1 job at a time) | 23.73 / 50.6 | – |

Each survey already runs ~5× parallel inside (block decodes, Welch, trial tracks), so running several at once only
adds contention (+30–70 % CPU, no wall gain). Surveys of several inputs run one after another; §2's 19.05 s was
measured under more background load.

## 9. Regression guard

`bench/check.sh` (≈ 1–2 min): `tapespeed IN --ref 9000 --abs line:15625 --no-plot` × N (median wall) on a synthetic
5 min mono reel recording with a 15625 Hz line (generated once with `tapespeed synth` into `target/bench-check/`),
plus the criterion 1-thread subset `m3/correct_10s_stereo_1t`, `m3/het_decimate_60s_1t`, `m3/pll_1M_1t`,
`b/welch_512k_nper4096_1t`, `b/stft_60s_44k1_nper4096_1t` (median point estimate). It fails (exit 1) if any median
exceeds baseline × (1 + TOL), TOL = 0.10, and prints the load average (a 1-min load > 8 is flagged).
`bench/baseline.json` is machine-specific: on an idle machine run `bench/check.sh --update`, then `bench/check.sh` once
more to confirm, and commit `baseline.json` together with the change that moved the numbers. Knobs: `TOL`, `N`,
`TS_CHECK_DIR`, `TS_SKIP_MICRO=1` (macro only).

## 10. Knowledge gaps

- No CPU affinity on macOS: P-core-only runs could not be enforced; the E-core effect is inferred from 8 vs 10 threads.
  Not measured on Linux.
- Background load ≈ 0.5–1 core throughout; an idle-machine run was not possible. Expected effect < 5 % on the 10-thread
  numbers, less at 1 thread.
- ffmpeg's share is inferred from WAV vs FLAC output and child CPU, not profiled inside ffmpeg.
- Per-stage 1-thread times exist only for `pair`.
- Memory per stage is from one `memprof` run per configuration; batch memory per job was not split.
