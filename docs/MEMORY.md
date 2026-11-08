# Memory, chunking and machine adaptation (design + measurements)

Objective: **maximise throughput within the machine's memory budget** — RAM not used is RAM wasted.
Default: everything in RAM, all parallelism on. Spill and reduced concurrency are fallbacks, used only when the
estimate exceeds the budget. All fallbacks are **bit-identical** to the default run (§5).
Code: `crates/tapespeed/src/plan.rs` (planner, budget), `crates/ts-io/src/{machine,spill}.rs` (probe, spill file),
`crates/ts-core/src/analyze/api.rs` (candidate scheduler, per-sample coefficients).

## 1. Method
- Stage profile with `TAPESPEED_PROF=1` (per stage: wall time, CPU, RSS and footprint peak from a sampler) and the
  `memprof` build feature (counting global allocator → **live-heap peak** per stage).
- **macOS `peak memory footprint` under-reports.** Example, 43 min: live heap 4.04 GiB, footprint 2.54 GB, RSS
  5.50 GB. `TASK_VM_INFO`: internal ≈ footprint, "reusable" 4.7 GB. Likely cause: libmalloc reusable-page accounting on
  reused large blocks (unverified). Planning metric = **live heap**; cross-check = RSS (≥ heap in all runs).
- Inputs: a 43.35 min stereo MP3 aircheck (44.1 kHz), args `--ref 9000 --abs pair --auto-notch`; synthetic
  `synth --preset reel --duration 7200` (2 h, FLAC 44.1 kHz stereo), args `--ref 9000 --auto-notch [--search 0.05]`.
- Machine: Apple M1 Max (8 P + 2 E, 64 GiB), shared with other jobs (load average 35–85). Times are ±15 %; memory is
  unaffected.

## 2. Measured profile
Per-stage live-heap peak **excluding the input** (spill run, so the input is not on the heap), GiB, and wall time, s:

| stage | full-rate audio? | what the stage keeps / works on | 43 min GiB | 43 min s | 2 h GiB | 2 h s |
|---|---|---|---|---|---|---|
| load | writes | f32 interleaved, X = 4·sr·ch·T (0.86 / 2.37 GiB) | – | 2.60 | – | 1.86 |
| an.stage1 | reads | z c128 at fs_d = 1633 Hz (d1 = 27, 731 taps) | 0.08 | 0.34 | 0.19 | 0.94 |
| an.coarse | – | z + Viterbi tracks + fcd (≈ 32 B/z-sample) | 0.13 | 0.31 | 0.35 | 0.84 |
| **an.tracks (5 bw ∥)** | – | per bw: zp at fs_p = 181/408/816/1633/1633 Hz, PLL, 16 output arrays | **3.39** | 2.99 | **10.15** | 8.56 |
| an.stage6 | – | 5 track outputs (93 B/fs_p-sample, 1.0 / 3.7 GiB) + grids at 200 Hz | 1.52 | 0.73 | 4.08 | 2.11 |
| pair (+ re-measure) | reads ×2 | 2 basebands: fp0 ± 36–99 Hz (4.6–12.7 k taps); 0 ± 420 Hz (1093 taps); Welch | 0.42 | 1.43+1.47 | n/a | n/a |
| switch_repair | reads | baseband fc ± half, block PSD | 0.11 | 0.57 | n/a | n/a |
| model.fit (L1–L3) | – | 200 Hz curve; sparse normal equations (9.2 M fill) | 0.15 | 1.62 | 0.18 | 1.59 |
| model.aided | reads | baseband at ref·r̄ ± 1.15·dev, refine ×n | 0.29 | 3.71 | 0.69 | 6.32 |
| evidence | reads | line baseband ±(dev + 37.5 Hz) | 0.15 | 0.56 | 0.33 | 1.50 |
| l1, plot | – | curve | 0.09 | 0.32 | 0.24 | 0.92 |
| correct ×1–3 (detect, keying, final) | streams | O(chunk); encoder | 0.15–0.25 | 3.8+4.3+7.0 | 0.25–0.39 | 10.4+18.4 |

Whole run, GiB live heap / GB footprint / GB RSS / s wall, for the design variants that were measured:

| mode | 43 min | 2 h (search 0.03) | 2 h (search 0.05) |
|---|---|---|---|
| input `Vec` grown by doubling, 5 bw ∥ | 4.53 / 2.67 / 6.23 / 31.5 | – | 17.99 / 8.93 / 21.7 / 57.4 |
| exact reserve, 5 ∥ | **4.04** / 2.54 / 5.50 / 31.8 | **11.64** / 6.04 / 15.4 / 53.8 | 16.18 / 7.68 / 20.5 / 57.0 |
| exact, bw 2 ∥ | 3.34 / 2.16 / 5.05 / 35.0 | – | – |
| exact, bw 1 ∥ | 3.32 / 1.97 / 4.89 / 37.3 | 9.19 / 5.54 / 12.3 / 66.3 | 11.43 / 7.38 / 15.4 / 74.6 |
| exact + lean stage 6, bw 1 ∥ | 2.76 / 1.76 / 4.28 / 37.8 | – | 9.40 / 6.92 / 13.3 / 76.7 |
| spill (mmap), 5 ∥ | 3.39 / 1.42 / 5.63 / 31.9 | 10.15 / 4.22 / 15.8 / 53.5 | 12.79 / 5.01 / 17.7 / 59.1 |
| spill + lean, bw 1 ∥ | – | **5.26** / 2.76 / 10.4 / 68.6 | – |

Findings:
1. The peak comes from the **analyze bw candidates** (4.7–5.1 GiB/h with 5 in parallel). Everything else is
   ≤ 0.6 GiB/h above the input. Pair, aided, evidence, correct and the model are not an issue.
2. Growing the input buffer by doubling costs a transient 1.50 GiB / 43 min (6.0 GiB / 2 h); whole-run peak +0.5 /
   +1.8 GiB. Reserving from the ffprobe duration avoids it at no cost (implemented).
3. Spill (unlinked f32 temp file, mmap read-only): clean file pages count neither in footprint nor on the heap. Time ±0
   (31.9 vs 31.8 s; 53.5 vs 53.8 s). Writing the spill file costs nothing measurable.
4. Lower bw concurrency only helps a little (−0.7 GiB / 43 min, −2.4 GiB / 2 h) for +17–23 % time, because bw45 and
   bw80 (fs_p = fs_d) dominate. bw 2 ∥ = bw 1 ∥ in peak.
5. Lean stage 6 (grid each candidate right after tracking, keep only grid + C/N0 + valid): retained memory −1.0 /
   −3.7 GiB; peak −0.6 / −2.0 GiB at bw 1 ∥ (implemented).
6. Per-candidate scratch S ≈ 420 B per fs_p sample (both files).
7. Decode cost, ffmpeg to f32 pipe, wall per hour of audio: MP3 3.5 s, FLAC 0.8–2.0 s (multi-threaded, 4.7–6.5 s
   CPU), WAV 0.5 s.
8. Batch, 43 min × J, separate processes: J=1 31.7 s/file, J=2 21.8 s/file, J=4 18.0 s/file → 1.00 / 1.45 / 1.76×.
   One job averages 3.7 cores (117 s CPU / 32 s). Heap peak per job 4.2–4.6 GiB, RSS 5.5–5.8 GB.
9. `tapespeed synth` is a test-signal tool, not the pipeline; its memory is not designed for multi-hour programs.

## 3. Design: consumers of full-rate audio
Every consumer except `correct` is a heterodyne + decimating FIR (`het_decimate`). Its output depends only on
`x[k−(m−1)..=k]` and the absolute index i (NCO phase `mod(i·w, 1)`), so it **streams exactly** with m−1 frames of
history (`HetStream`, bitwise equal to `het_decimate`). Stage 1 of pass 1 and of the `line:` pass runs as a
`HetStream` while ffmpeg decodes (bounded queue of 32 × 64 Ki frames); the planner counts the later pass's baseband
(16 B per fs_d sample) during pass 1. `correct` streams too.

| consumer | default | fallback | chunk boundary / state |
|---|---|---|---|
| stage 1 (+ `line:` pass) | RAM, streamed during decode | spill mmap | m−1 = 730 frames; emit q once k = half + d·q is present; file end: zero-pad with n known at the end |
| pair spectrum + sub-audio band | RAM | spill mmap | m−1 = 4.6–12.7 k / 1.1 k frames |
| switch_repair, pair re-measure | RAM | spill mmap | same |
| model.aided | RAM | spill mmap | same; refine iterations reuse the decimated baseband |
| evidence line baseband | RAM | spill mmap | same |
| correct ×1–3 (detect / keying / final) | RAM | spill mmap (`&[f32]` over the mapping) | time map, ±m taps, notch 16 k look-ahead, lowpass overlap-save, STFT 60 s chunks |
| survey | own path (`-ss` blocks) | – | – |

Global steps stay in RAM on decimated or curve data (43 min / 2 h): z 68 / 188 MB; coarse Viterbi ≤ 0.13 / 0.35 GiB;
bw candidates 3.4 / 10.2 GiB (5 ∥); select on 200 Hz grids (520 k / 1.44 M points); model L1–L3 ≤ 0.18 GiB; pair
Welch on ≤ 2.5 kHz basebands ≤ 0.42 GiB.
PLL, Viterbi, fit and morphology never see a chunk boundary. Time-chunking the analyze tracks (PLL state carry,
backward loop in reverse chunk order, overlap ≥ the largest window: running median 10 s) is **not implemented**; it is
needed only beyond the limits in §6.

Fallback ladder per file, lowest cost first:

| level | input | bw candidates | time cost (measured) |
|---|---|---|---|
| ℓ0 | RAM (exact reserve) | all ∥ | 0 |
| ℓ1 | spill mmap (needs X free disk in `--tmp-dir`) | all ∥ | ±0 |
| ℓ2 | spill (or RAM without disk) | budgeted, largest first, at least one | +17–23 % at one candidate |

## 4. Planner
Inputs: probe (sr, ch, T = ffprobe duration + 1 s), args (ref, search, bw list, abs, second line), machine, CLI.
Machine probe (< 1 ms):
- macOS: `hw.memsize`, `hw.perflevel0/1.physicalcpu` (P/E), `host_statistics64` → A = (free + inactive + purgeable)·page.
- Linux: `/proc/meminfo` MemAvailable; cgroup v2 `memory.max` − `memory.current` along the cgroup path.

Estimates in bytes (coefficients from §2; error vs measured −7 … +24 %, conservative):
```
X        = 4·frames·ch,  frames = (T + 1 s)·sr                       (input)
N_d, N_k = stage-1 samples; samples of bw candidate k at its fs_p    (analyze::api::sizes)
fixed    = 24·N_d + 16·N_d(later pass) + 20·ΣN_k
own      = 420·N_max                                                  (largest candidate alone)
cand_k   = 250·N_k                                                    (each further concurrent candidate)
oth      = 180 kB·T                                                   (pair, aided, evidence, correct)
peak(x, an) = 1.1·(100 MiB + x + max(fixed + an, oth))
B        = --max-ram (≤ physical RAM / cgroup limit), else 0.8·A (A capped by the cgroup headroom)
jobs     = min(⌈threads / 2.5⌉, files)                                (one job ≈ 3.7 cores)
```
Algorithm:
1. Single file: ℓ0 if peak(X, all) ≤ B; else ℓ1 if the spill dir has > 1.05·X free and peak(0, all) ≤ B; else ℓ2:
   candidates start largest first while the budget holds, at least one runs (`WARNING: one bw candidate at a time
   already exceeds the budget` on the `memory:` line when even that is over budget).
2. Batch (several inputs, one process): a shared budget. A job reserves peak(x, own) before it starts (blocking; an
   oversized job runs alone); further bw candidates take budget only if it is free (non-blocking), so jobs cannot
   deadlock each other.
3. Reservations block only on job driver threads, never inside rayon workers. One global rayon pool, size =
   `--threads` (default: all cores); `--threads` also caps the concurrently running bw candidates.
4. The plan goes to stdout as the `memory:` line.

## 5. Quality
| check | result |
|---|---|
| 43 min: ratio (full f64) for doubling / exact / spill / bw 1 / bw 2 / lean | bitwise identical |
| 43 min: decoded output audio for doubling / bw 1 / exact / spill | bitwise identical |
| 2 h: ratio for doubling / exact / bw 1 / bw 2 / spill / lean (both search values) | bitwise identical per search value |
| `HetStream` vs `het_decimate`: 43 min MP3, 2 h FLAC, unit test (chunks 1…1 M frames) | bitwise identical |
| 2 h reel vs truth | err RMS 1.06e-4 all / 1.00e-4 valid, valid 98.7 % (same order as the 60 s presets, 1.6e-4) |

## 6. Expected footprint (live heap) and time
B assumes A = M − 5 GiB and the 80 % default → 16 GB: 8.8, 32 GB: 21.6, 64 GB: 47.2 GiB. Default args as in §1,
search 0.03; times from the loaded machine.

| case | 16 GB | 32 GB | 64 GB |
|---|---|---|---|
| 43 min | ℓ0: 4.0 GiB, 32 s | ℓ0: 4.0, 32 s | ℓ0: 4.0, 32 s |
| 2 h | ℓ2: ≈ 5.3 GiB (spill, 1–2 candidates), ~69 s | ℓ0: 11.6, 54 s | ℓ0: 11.6, 54 s |
| 4 × 43 min batch | J=2–4, ℓ0/ℓ1/ℓ2 by admission, 75–87 s (estimate) | J=4 ℓ0 (Σ ≤ 16.2), 72 s | J=4 ℓ0 (Σ ≤ 16.2), 72 s |
| max T at ℓ2 (2.65 GiB/h) | 3.0 h | 7.4 h | 16 h |

Sequential 4 × 43 min = 127 s.

## 7. CLI
| option | default | meaning |
|---|---|---|
| `--max-ram SIZE\|PCT` | 80 % of available | an explicit value sets B directly (it may also raise it), capped at physical RAM / cgroup headroom. `12G`, `512M`, `50%` (of available) |
| `--threads N` | all cores (P + E) | global rayon pool; also caps concurrent bw candidates |
| `--tmp-dir DIR` | `$TAPESPEED_TMPDIR`, else the system temp dir | spill file: unlinked right after creation (never left behind, even after SIGKILL), mapped read-only; a guard removes it on errors/panics before the unlink |
| `IN1 IN2 … -o DIR` | – | batch in one process, shared budget, ≤ ⌈threads/2.5⌉ jobs, largest reservation first |

`memory: mode ram|spill (dir), bw k/5 parallel, est peak, budget (source, available), jobs, threads` is the first
stdout line of a run and the last row of the `--survey` PLAN section. Output naming and batch stdout: MANUAL §5.3.

## 8. Limits and risks
| # | risk | mitigation |
|---|---|---|
| 1 | Footprint vs heap vs RSS diverge (macOS reusable accounting; a Linux cgroup counts page cache incl. the spill file) | plan on the live heap; spill pages are reclaimable |
| 2 | Coefficients from 2 files / 1 configuration (48 kHz, mono, other `--pair-dmax` untested) | estimates scale with the decimated sample counts, not per hour; 10 % slack |
| 3 | ffprobe duration wrong (VBR MP3 without Xing header) | reserve T + 1 s, the buffer grows past it; spill knows n exactly |
| 4 | Spill disk: 1.27 GB/h stereo 44.1 kHz; cold page-cache read rate not measured | disk-space check before spilling |
| 5 | E cores: a serial PLL candidate or the model fit on an E core gives a long tail | not measured separately |
| 6 | Batch speedups measured under load 35–85 | re-measure on an idle machine before tuning the job count |
| 7 | Files beyond the §6 limits | a time-chunked analyze would be needed (not implemented) |
