# tapespeed manual

Reference for the `tapespeed` binary. Facts come from the code and the measurements in `bench/results.md` and
`docs/MEMORY.md`. **[unverified]** marks claims that were not checked against code or a measurement.
The option tables are checked against `tapespeed --help` and the clap definition by `crates/tapespeed/tests/manual_drift.rs` (§17).

Contents: 1 Scope · 2 Build · 3 Quick start · 4 Pipeline · 5 CLI reference · 6 `--abs` modes · 7 Subcommands ·
8 Outputs · 9 Recipes · 10 Accuracy and limits · 11 Performance · 12 Memory planner · 13 Exit codes and messages ·
14 Environment · 15 Tests · 16 Glossary · 17 Doc drift check

## 1. Scope
Corrects speed, wow and flutter of tape transfers from a reference line recorded on the tape: AM adjacent-channel
beat (9/10 kHz), FM stereo pilot (19 kHz), TV line whistle (15625/15734 Hz), test tone. Output: corrected audio
(one resampling pass), speed curve CSV, report JSON, PNG plots. `--survey` finds candidate lines without writing audio.

## 2. Build and install
| item | value |
|---|---|
| toolchain | Rust `1.97.1` (pinned in `rust-toolchain.toml`, components rustfmt, clippy), edition 2024 |
| target flags | `.cargo/config.toml`: `-C target-cpu=apple-m1`. On other CPUs remove or change it, else the binary may not run [unverified on non-Apple hosts] |
| release profile | thin LTO, `codegen-units = 1`, `debug = 1` |
| runtime dependency | `ffmpeg` and `ffprobe` in `PATH` (decode of every input, encode of every output that is neither `.wav` nor `.flac`; `.flac` is encoded in-process, §8.1). Tested with ffmpeg 9.0.2 |
| OS | macOS (measured), Linux (machine probe incl. cgroup v2; not benchmarked) |

```
cargo build --release -p tapespeed          # -> target/release/tapespeed
cargo install --path crates/tapespeed       # optional, into ~/.cargo/bin
./ci.sh --fast | ./ci.sh | ./ci.sh --full   # fmt + tests; default also clippy; full also long and debug tests
cargo build --release --features memprof --target-dir target/memprof   # live-heap counter for TAPESPEED_PROF
```

## 3. Quick start
```
tapespeed rec.mp3 --survey                                  # candidates + recommended options, no audio
tapespeed rec.mp3 --ref 9000 -o rec.flac                    # correct; writes rec.flac, rec_curve.csv, rec_report.json, rec_speed.png
tapespeed rec.mp3 --ref 9000 --abs pair --check-line 15625 --lowpass --auto-notch -o rec.flac   # AM aircheck, full set
tapespeed *.mp3 --ref 9000 --auto-notch -o out/             # batch: out/<stem>.flac + side files
```

## 4. Processing pipeline
Order as executed by `pipeline::run_file` (single file). `[ ]` = conditional.

| # | stage (`TAPESPEED_PROF` mark) | input → output | key parameters | measures / decides |
|---|---|---|---|---|
| 0 | plan | ffprobe info → `memory:` line, `Plan` | `--max-ram`, `--tmp-dir`, `--threads` | RAM vs spill, bw candidates in parallel (§12) |
| 1 | load (decode, stage 1 overlapped); stage-1 tail (after decode) | file → f32 interleaved (RAM `Vec` or unlinked mmap spill file); each decoded chunk also goes to a stage-1 thread (queue 32 × 64 Ki frames) that streams row 2 for pass 1 and for row 6 (`HetStream`; 3 work items at a time while it keeps up, the whole pool when ≥ 8 chunks are queued and after the decode) | ffmpeg `-map 0:a:0 -f f32le` | – |
| 2 | an.stage1 (≈ 0 s: computed during row 1) | f32 → complex baseband z at fs_d; bit-identical to the in-RAM kernel | centre = `--ref` × play factor; `--channel` (mid=(L+R)/2, S=(L−R)/2, L, R); search ±`--search` | heterodyne + FIR decimation (9 kHz/44.1 kHz: d1 = 27, 731 taps, fs_d ≈ 1633 Hz) |
| 3 | an.coarse | z → coarse frequency track | `--search` | Viterbi track of the line |
| 4 | an.tracks | z → per-bw curve | `--bw auto` = candidates 6/12/25/45/80 Hz (passband 3.3/6.6/13.8/24.8/25 Hz) or one fixed bw | fwd/bwd PLL (ζ 0.707) + zero-phase refinement; slip/dip/interference/consistency guards → confidence, valid |
| 5 | an.select | candidates → curve on 200 Hz grid | fmax 25 Hz | bw choice (meta `bw=auto:N`); gaps = PCHIP trend + AR flutter prediction |
| 6 | [analyze #2] | f32 → second-line curve | `line:HZ` or `--check-line HZ` (searched at HZ × play) | second independent curve for L1 (`line:`) and coherence |
| 7 | [pair.measure] | f32 + pass-1 curve → pair table | `--abs pair[:HZ]`, `--pair-dmax`, cross-check line if 15625/15734/19000 | partner beat on the speed-normalised spectrum; f_true = 2·HZ·F_p/(F_lo+F_hi) (§6) |
| 8 | [pair.switch_repair, pair.remeasure] | pass-1 curve → repaired curve | ρ = F_hi/F_lo (or ghost), 5 s blocks / 2 s segments; off with `--no-switch-repair` | spans where pass 1 tracked the other beat; divided out; pair re-measured |
| 9 | model.fit | curve → L1/L2/L3 model, EV | `--model`, `--model-ta`, `--tape-speed`; skipped if file < 60 s or valid < 20 % | components (capstan, rollers, reels, clutch), jump test (≥ 0.1 % step within 1 s), EV |
| 10 | model.aided | f32 + model → refined curve | EV ≥ `--model-min-ev` or `--model on`; `--model-band` (capped at the pass-1 passband when bw was auto); 2 refinements | `aided` path; else `plain` (pass-1 curve, gap fill by hold-out: model vs analyze) |
| 11 | [long gaps] | curve → curve | ≥ 20 s with valid density < 2 % over 10 s | edge-linear inside, ramp to L1 at file ends, invalid; WARNING |
| 12 | evidence | f32 + curves → fading ρ, coherence, correction band | `--corr-limit`, `--weak-rf`, `--max-corr-hz` | bandwidth policy (§5.4); curve low-passed at `corr_max_hz` if limited, periodic model part kept |
| 13 | l1 | curve → final speed curve s | `--abs`, `--tape-speed`, `--play-speed` | absolute level L1; variations unchanged |
| 14 | curve+plot | → `_curve.csv`, `_speed.png`, [`_model.png`] | `--curve`, `--plot`, `--no-plot` | wow/flutter rms (report `plot`) |
| 15 | [autonotch.detect] | correct → reference-notched output → detector | `--auto-notch`, `--auto-notch-min` | lines ≥ 15 dB SNR, IQR ≤ 3 Hz, ≥ 20 s total, in 5 s blocks |
| 16 | [autonotch.keying] | second correct pass → keying statistics | – | keyed if contrast p90−p10 ≥ 25 dB and mid-third share ≤ 0.25 |
| 17 | correct+notch+lp (pipe) | f32 + curve → encoder (`.flac`: in-process, 16-block batches encoded in parallel on the pool, §8.1) | `--taps`, `--notch-q`, `--keep-ref`, `--notch-at`, `--notch-width`, `--lowpass`, `--lowpass-width` | one varying-rate resampling (Kaiser sinc, 1024 phases, f32), reference notch, extra notches (whole file or gated), lowpass; streamed, no output buffer |
| 18 | encode finish, report | → audio file, `_report.json`; curve CSV rewritten with `extra_notches` | – | – |

Correction math: r(t) = f_apparent/f_true; τ = T(t_in) = ∫₀^t_in r dt; output sample k at τ_k = k/sr is read at
t_in = T⁻¹(τ_k). Output length ≈ T(t_end)·sr, same sample rate and channel count as the input.
Never resample twice: each extra pass costs ~2 dB of resampling error.

## 5. CLI reference: `tapespeed [OPTIONS] <input>...`
Long options accept unique prefixes (`--model-b 5` = `--model-band 5`); an ambiguous prefix is an error (`--max`).
Negative numbers are accepted as values. `--help` shows each default as `[default: X]`; the table lists them too.
Optional-value options (`--lowpass`, `--auto-notch`) take the next token if it is not an option: put them last, before another option, or write `--lowpass=7000`.

### 5.1 Options
<!-- drift:main:begin -->
| option | type | default | meaning, interactions |
|---|---|---|---|
| `--max-ram` | SIZE | – | memory target B (a planning budget, not a hard cap: the measured peak can exceed it somewhat, e.g. 3.9 GB RSS in spill mode at `3G`, the page cache of the spill file included): bytes, `K/M/G/T` suffix (binary, case-insensitive) or `N%` of available RAM. Unset: 80 % of available. Explicit value may raise B; capped at physical RAM / cgroup limit. §12 |
| `--threads` | int | – | rayon pool size and cap on concurrently running bw candidates. Unset or 0: rayon default (`RAYON_NUM_THREADS`, else all logical cores). Batch jobs = ⌈threads/2.5⌉ |
| `--tmp-dir` | DIR | – | spill directory. Unset: `$TAPESPEED_TMPDIR`, else system temp dir. File is unlinked at creation. Spill needs the dir to exist and free space > 1.05 × input f32 size |
| `--ref` | float Hz | – | detection frequency = search centre. Required unless `--survey`. Also the nominal true frequency for `--abs ref` |
| `-o`, `--output` | path | – | corrected audio. `.wav` (case-insensitive) → f32 PCM; `.flac` → 24 bit (in-process encoder); lossy containers get sane encoder settings (§8.1) and a `note:` line; other extensions → ffmpeg's default codec for the container. Never an input file (§5.5). Required unless `--survey`. Batch rules §5.3 |
| `--overwrite` | flag | off | allow an output path (audio, curve, report, plots, survey files) to be the same file as an input (§5.5). Without it: `tapespeed: error: ...`, exit 2, before anything runs. Existing non-input outputs are always overwritten |
| `--survey` | flag | off | no audio: candidates, recommendation, plan (§7.1). `--ref`, `-o` optional |
| `--survey-blocks` | int | `12` | survey: blocks spread evenly over the file (negative → 0) |
| `--survey-block-s` | float s | `20` | survey: block length |
| `--curve` | path | – | curve CSV. Unset: `<output stem>_curve.csv` next to `-o`. Batch: needs `{stem}` |
| `--report` | path | – | report JSON. Unset: `<output stem>_report.json`. Batch: needs `{stem}` |
| `--plot` | path | – | speed plot path; also enables the model plot `<plot stem>_model.png`. Unset: `<output stem>_speed.png`, no model plot. Batch: needs `{stem}` |
| `--no-plot` | flag | off | no speed plot, no model plot, no report key `plot` |
| `--keep-ref` | flag | off | do not notch the reference tone. Default: IIR notch (Q `--notch-q`) at the corrected frequency `abs_ref_true_hz` |
| `--lowpass` | float Hz | off; bare: `8000` | linear-phase Kaiser FIR, −6 dB at HZ, 80 dB stopband, transition ±`--lowpass-width`/2, zero delay. Reference notch and extra notches at ≥ HZ + width/2 are skipped (in stopband). `0` = off |
| `--notch-at` | list Hz | – | extra whole-file notches at output frequencies, comma-separated (`8021,3361`; spaces ignored). Q = HZ/`--notch-width` |
| `--auto-notch` | `{keyed,all}` | off; bare: `keyed` | detect stationary narrowband lines in the corrected output, notch only in 5 s blocks where present (0.2 s crossfade). `keyed`: only on/off keyed lines (CW); steady carriers kept. `all`: every detected line. Costs 2 extra correct passes |
| `--auto-notch-min` | float Hz | `3000` | lowest frequency searched by `--auto-notch` |
| `--notch-width` | float Hz | `150` | −3 dB width of extra notches (manual and auto) |
| `--lowpass-width` | float Hz | `500` | lowpass transition width (full width) |
| `--notch-q` | float | `30` | Q of the reference notch (Q 30: line −26…−52 dB, program ±1 kHz −0.8…−1.1 dB, measured on Caroline) |
| `--search` | float | `0.03` | max \|speed offset\| as a fraction (0.03 = ±3 %) around the centre |
| `--bw` | `auto` or float Hz | `auto` | pass-1 PLL loop bandwidth; `auto` tries 6/12/25/45/80 Hz and selects |
| `--channel` | `{mid,L,R,S}` | `mid` | analysis channel; mid = (L+R)/2, S = (L−R)/2. Output keeps all channels. On a mono file `L`/`R`/`S` print `warning: --channel S on a mono file: analysing the mono signal (as --channel mid)` and run as `mid` |
| `--taps` | int | `64` | resampler kernel length (negative → `ValueError: taps must be >= 0`) |
| `--model` | `{auto,on,off}` | `auto` | model step. `auto`: aided if EV ≥ `--model-min-ev`; `on`: aided regardless of EV (reason `(forced)`); `off`: no model (path `off`). Files < 60 s or valid < 20 %: path `plain`, no model, whatever the mode is |
| `--model-min-ev` | float | `0.7` | EV threshold for the aided path |
| `--model-band` | float Hz | `20` | aided residual band B; capped at the pass-1 passband when `--bw auto` chose it |
| `--model-ta` | float s | `5` | L3 amplitude knot spacing |
| `--abs` | spec | `ref` | L1 source, §6 |
| `--pair-dmax` | float Hz | `300` | `--abs pair`: max \|station offset from the raster\| d (min 5 Hz, fixed) |
| `--no-switch-repair` | flag | off | `--abs pair`: keep pass-1 spans locked onto the other beat |
| `--check-line` | float Hz | – | second line for coherence evidence only (L1 unaffected). Also the pair cross-check if 15625, 15734 or 19000. Ignored when `--abs line:` (that line is used) |
| `--max-corr-hz` | float Hz | `6` | correction band limit when the policy limits |
| `--corr-limit` | `{auto,on,off}` | `auto` | bandwidth policy, §5.4 |
| `--weak-rf` | flag | off | = `--corr-limit on` (overrides `--corr-limit`) |
| `--tape-speed` | speed | – | nominal recording speed: `19.05cm`, `19.05 cm/s`, `7.5ips`, `7½ips`, `1 7/8 ips`, `15/16ips`, `0.1905m/s`; bare number = cm/s. Used by model physics |
| `--play-speed` | speed | – | transfer playback speed if different; needs `--tape-speed`. Play factor F = play/tape: search centre = ref·F, F part of L1 |
| `-h`, `--help` | flag | – | help text, exit 0 |
| `-V`, `--version` | flag | – | print `tapespeed <version>`, exit 0 |
<!-- drift:main:end -->

### 5.2 Interactions
| combination | effect |
|---|---|
| `--abs` vs `--tape-speed`/`--play-speed` | F = play/tape sets the gross level (search centre ref·F, curve × F). `ref`/`assume`/`line`/`pair` measure L1 with F included; `value`/`percent`/`none`/`curve` set L1 relative to F. Physics uses v_play if given, else v_rec |
| reference notch vs `--lowpass` | notch skipped if `abs_ref_true_hz` ≥ lowpass_hz + lowpass_width/2 (line `reference tone not needed: in lowpass stopband (>= 8250 Hz)`) |
| `--keep-ref` | no reference notch; meta `notch_hz=none` |
| extra notch vs reference notch | skipped if \|HZ − notch_hz\| < notch_width/2 (`covered by reference notch`) |
| extra/auto notch vs lowpass | skipped if HZ ≥ lowpass stop (`covered by lowpass`); auto `keyed` steady lines there print `removed by lowpass` |
| `--weak-rf` vs `--corr-limit` | `--weak-rf` wins (`on`) |
| `--check-line` vs `line:HZ` | `line:` defines the second line; `--check-line` ignored |
| `--model-band` vs `--bw` | B capped at the pass-1 passband only when bw was chosen automatically |
| `--threads` vs batch | jobs in parallel = min(⌈threads/2.5⌉, files), ≥ 1; each job runs its bw candidates within the shared budget |
| `--max-ram` vs `--tmp-dir` | over budget → spill if the tmp dir exists and has room, else RAM with fewer candidates (§12) |

### 5.3 Batch: several inputs
| case | `-o` meaning | outputs |
|---|---|---|
| 1 input, `-o FILE` | output file | `FILE`, `<stem>_curve.csv`, `<stem>_report.json`, `<stem>_speed.png` next to it |
| 1 input, `-o DIR` (existing dir, or `DIR/` with a trailing slash: created if missing) | directory | `DIR/<input stem>.flac` + side files |
| n inputs, `-o DIR` (existing dir, trailing `/`, or a name without an audio extension) | directory, created if missing | `DIR/<input stem>.flac` each |
| n inputs, `-o all.flac` (audio extension: wav, flac, mp3, ogg, opus, m4a, aac, mp4, aif(f), ...) | error `-o all.flac: with several inputs -o is a directory or contains {stem} (e.g. -o all/ or -o 'all/{stem}.flac')`, exit 2 | – |
| any, `-o out/{stem}.wav` | template | `{stem}` replaced by the input stem; missing directories are created |
| n inputs, `-o` existing file | error `-o X: with several inputs -o is a directory or contains {stem} ...`, exit 2 | – |
| n inputs, `--curve/--report/--plot` without `{stem}` | error `--curve with several inputs needs {stem} in the path`, exit 2 | – |
| two inputs with the same stem | error `<input>: output name collides with another input (same stem "x")`, exit 2 | – |

Scheduling: jobs sorted by reservation (largest first), ≤ ⌈threads/2.5⌉ in parallel, each admitted when its
reservation fits the shared budget (an oversized job runs alone). Stdout per file, printed when the file finishes:
`== <input>`, its `memory:` line, its lines (§8.6). Last line: `batch: N files, K ok, F failed; jobs J, budget X GiB (...), threads T`.
Failures: stderr `<input>: <message>`; exit 1 if any failed (`F of N files failed` on stderr). `--survey` with several
inputs runs the files one after another.

### 5.4 Bandwidth policy (`--corr-limit`)
| mode | rule | `corr_policy` text |
|---|---|---|
| `off` | full band | `off` |
| `on` (= `--weak-rf`) | curve low-passed at `--max-corr-hz` (Butterworth-4, zero-phase), periodic model components above it kept | `forced` |
| `auto`, second line present | correct up to the top of the contiguous bands with coherence ≥ 0.5 (bands 0.5–6, 6–25 Hz) | `coherent to 25 Hz` (full) / `coherent to 6 Hz` / `no coherent band above 0.5 Hz` (limit 0.5 Hz) |
| `auto`, no second line, path `plain`, `fade_rho` > 0.9 | limit at `--max-corr-hz` | `fading rho R > 0.9, no second line` |
| `auto`, otherwise | full band | `no evidence against tape origin` |

### 5.5 Inputs are never overwritten
Before anything runs (batch: before the first file starts), every output path of every file (audio, curve, report,
speed and model plot; survey JSON and PNG) is compared with every input (all batch inputs and an `--abs curve:CSV`
file): canonical paths (symlinks, `./`, `..` resolved; a missing output by its canonical parent + name) and, for
existing files, device + inode (hard links). A match is refused: `tapespeed: error: <input>: audio output X is the
input file Y: refusing to overwrite an input (choose another -o, e.g. <name>_corrected.flac, or pass --overwrite)`,
exit 2, nothing written. `--overwrite` disables the check (the input is decoded completely before the output is
opened, so this works, but the original is gone). Existing outputs that are not inputs are overwritten without asking
.

## 6. `--abs` modes
Curve ratio r = f_apparent/ref (× F); final s = r·ref/f_true; only L1 = mean(s) changes.

| spec | L1 / f_true | extra meta | fails with |
|---|---|---|---|
| `ref` | f_true = `--ref` | – | – |
| `none` | L1 = 1 × F (average speed kept) | – | – |
| `assume:HZ` | f_true = HZ | – | – |
| `line:HZ` | second line tracked over the whole file; q = Huber mean of r1/r2 on common valid samples; f_true = ref·q | `line_*` | `RuntimeError: line HZ Hz: too few samples valid in both lines (...)` (< 200 common valid samples) |
| `value:R` | L1 = R × F | – | – |
| `percent:P` | L1 = (1 + P/100) × F | – | – |
| `curve:CSV` | s = r·F·L1(t)/r̄₆₀ₛ(t), L1(t) = ratio column of CSV interpolated | `abs_curve` | CSV read error |
| `pair[:HZ]` | AM only. f_true = 2·HZ·F_p/(F_lo + F_hi), HZ = channel spacing (default 9000), from the partner beat on the pass-1-normalised spectrum; then as `assume:f_true` | `pair_*`, `sw_*` | `--abs pair: no partner line passing the criteria within |d| 5-300 Hz of the primary at F Hz (...)` or `no unambiguous partner`, followed by the candidate table; exit 1 |

Pair candidate criteria: peak ≥ 10 dB over ±10 Hz median; rejected if
< 15 dB or presence < 50 %; \|d\| outside 5…`--pair-dmax`; implied speed outside F × (1 ± 0.1); modulation match k̂ outside
0.6–1.4; mirror of the primary; co-channel (spacing = a 20–400 Hz sub-audio line ±0.4 Hz); comb with mains spacing
50/60/100/120 Hz × F ± 4 %. Runner-up within 6 dB = ambiguous, resolved only by a cross-check line agreeing within 0.01 %.
Sign of the station offset is not decidable from audio (`pair_offset_hz` valid if the primary is the lower neighbour's beat).

## 7. Subcommands
`--survey` is a mode of the main command (7.1). `synth`, `analyze`, `correct` (7.2–7.4) are dispatched when `argv[1]` names them; errors `tapespeed <name>: <message>`, exit 1; clap usage errors exit 2.

### 7.1 `--survey`
Decodes only `--survey-blocks` × `--survey-block-s` (ffmpeg `-ss/-t`), analyses L, R, mid, S. Stdout sections:
header, `CANDIDATES`, `COHERENCE`, `PRESENCE`, `LIMITS`, `RECOMMEND` (`--ref/--search/--channel/--abs/--check-line/--lowpass/--auto-notch` + `cmd:`),
`PAIR` (if a beat-pair check ran), `PLAN` (13 steps the run would take, resolved by the run's own parameter code, then
the `memory:` row), `WARNINGS`, final line `survey <json>, <png>; X s (Y min per hour of audio)`.
Writes `<stem>_survey.json` and `<stem>_survey.png` next to `-o`, else `<input stem>_*` in the cwd. No audio/curve/report.
The `memory:` row is estimated for the typed options (without `--ref`: for `--ref 9000`, suffix `(for --ref 9000)`).

RECOMMEND and `cmd:`:

| row | when | value |
|---|---|---|
| `--ref` | a usable primary | its nominal (FM/AM/TV), else its rounded frequency |
| `--search` | the primary's offset + 0.5 % or its p5–p95 spread + 0.25 % exceeds `--search` | that, rounded up to 0.01 (e.g. 9 kHz beat at −3.12 % → `0.04`; at +4.19 % → `0.05`) |
| `--channel`, `--abs`, `--check-line` | best channel within 3 dB of the strongest; `--abs` from the line class (`pair` when the PAIR check found the partner); a second usable line for coherence | – |
| `--lowpass` | primary is an AM beat | `8000`: nothing above ~8 kHz on medium wave is programme (the row shows the 8.25–9.25 kHz background level) |
| `--auto-notch` | primary is an AM beat | `keyed`: notches on/off keyed lines (CW) only where present, steady lines kept. The survey cannot detect keying itself (the uncorrected blocks smear CW lines by the wow), so this is a standing recommendation for AM |

`cmd:` contains every recommended option and `-o '<input stem>_corrected.<ext>'` in the cwd (`<ext>` = the input's for
lossless inputs, `flac` for lossy ones), so it never names the input. Paths are shell-quoted.

PLAN: without `--ref` it plans the recommended options, i.e. the `cmd:` line (header `PLAN (recommended options = the
cmd: line, typed options win; --ref 9000, from survey #1 (--ref not given))`): every RECOMMEND value is used where the
typed option is still at its default; typed non-default values and all other options (`-o`, `--weak-rf`,
`--tape-speed`, ...) are planned as typed. With `--ref` it plans exactly the typed options (header `PLAN (typed
options; the cmd: line is the recommendation; --ref 9000, given)`) and WARNINGS lists each recommended option that is
missing (`survey recommends --lowpass (see RECOMMEND)`). The "background ... content would be removed" lowpass warning
is not raised when the reference is an AM beat (there it is the interference the lowpass removes); the "near the
--search limit" warning uses the same rule as RECOMMEND `--search` and names the value.

Classes: an unknown line within 2 × `--search` of an AM beat nominal is classed `AM beat` (label `AM
adjacent-channel beat (EU 9 kHz at -3.12 %: outside --search 3 %, needs --search 0.04)`), so the survey recommends
`--ref 9000 --search 0.04` instead of `--ref 8719 --abs none`. An `FM pilot` candidate that is not the primary and whose
frame wow is uncorrelated with the primary's (wcorr < 0.2) is `unknown` (label `unknown line near the FM pilot: no FM
signature ...`): a pilot recorded on the tape wobbles with the tape.
PLAN step 13 is an estimate: time from the measured min/h of `bench/results.md` (43 min stereo MP3: `ref` 0.447, second line 0.620, pair +0.245 at 10 threads; one thread 1.128 / 1.643 / +1.260), memory = the planner's peak (`plan.rs`, same as the `memory:` row). Several inputs are surveyed one after another (each survey is parallel inside; 4 at once measured no faster, `bench/results.md` §10). Missing output directories are created.

| survey constant | value |
|---|---|
| line detection | Welch (Hann ~0.75 s) per block/channel, 0.05 % log grid, prominence ≥ 6 dB, persistence ≥ 50 % (0.09 s frames), grouped within 0.4 % |
| C/N0 scale | survey C/N0 = analyze C/N0 − ~4 dB (2.0–4.7 dB measured) |
| verdicts | no lock < 17, marginal 17–22, trackable ≥ 22 dB-Hz (survey scale); trial track (60 s analyze) overrides below 10 % valid |
| classes | `FM pilot` 19000, `AM beat` 9000/10000, `TV line` 15625/15734, `hum` 50/60 + harmonics, `codec`, `unknown` |
| pair check | primary AM beat: trial `--abs pair` on 3 × 60 s; found if ≥ min(2, trials) agree on d ±1 Hz |

### 7.2 `tapespeed analyze` (pass 1 only)
<!-- drift:analyze:begin -->
| option | type | default | meaning |
|---|---|---|---|
| `--ref` | float Hz | – | nominal reference frequency (required) |
| `--search` | float | `0.05` | max \|speed offset\| (fraction); note: the one-shot default is 0.03 |
| `--bw` | `auto` or Hz | `auto` | PLL bandwidth |
| `--fmax` | float Hz | `25` | max flutter frequency in the curve |
| `--channel` | `{mid,L,R,S}` | `mid` | analysis channel |
| `-o`, `--output` | path | – | curve CSV (required) |
| `--plot` | path | – | plot (pass-1 curve in the model plot layout) |
| `-h`, `--help` | flag | – | help |
<!-- drift:analyze:end -->
Usage: `tapespeed analyze IN --ref HZ -o c.csv`. Stdout: `bw auto:6, passband 3.3 Hz, 18001 points, valid 86.7 %, mean ratio 1.014122`.
Curve meta: `ref_hz, sr, channel, bw, passband_hz, search, source`. Ratio relative to `--ref`, no model step, no L1.

### 7.3 `tapespeed correct` (pass 2 only)
<!-- drift:correct:begin -->
| option | type | default | meaning |
|---|---|---|---|
| `-o`, `--output` | path | – | output audio (required) |
| `--notch` | flag | – | notch at meta `abs_ref_true_hz`, else `ref_hz` (missing → `KeyError: 'ref_hz'`) |
| `--notch-q` | float | `30` | notch Q |
| `--taps` | int | `64` | resampler kernel length |
| `-h`, `--help` | flag | – | help |
<!-- drift:correct:end -->
Usage: `tapespeed correct IN CURVE.csv -o OUT [--notch]`. Streams correct → notch → encoder. No stdout.

### 7.4 `tapespeed synth` (test signals)
`tapespeed synth [--preset P] -o OUT [--truth CSV] [--FIELD VALUE ...]` (also `--FIELD=VALUE`, `-` = `_` in names).
Writes OUT (WAV f32 / FLAC by extension) and the truth curve `<OUT without extension>_truth.csv` (or `--truth`).
`synth --help` / `-h`: usage, options and every field with its default, exit 0. Unknown field: `unknown parameter "x"`.
Deterministic for a given `--seed`. Presets: `fm19`, `am9w`, `am9`, `reel`, `clean`, `mech` (default `custom` = no preset).

<!-- drift:synth:begin -->
| field | default | meaning |
|---|---|---|
| `--preset` | `custom` | preset name (applied before the overrides) |
| `--duration` | `10.0` | input length [s] |
| `--sr` | `48000` | sample rate [Hz] |
| `--channels` | `2` | channels |
| `--seed` | `0` | RNG seed |
| `--ref_hz` | `9000.0` | reference tone [Hz] |
| `--offset` | `0.0` | constant speed offset [%] |
| `--drift` | `-0.2` | linear change over the file [%] |
| `--drift_sin` | `0.1` | very-low-frequency sinusoid amplitude [%] |
| `--drift_period` | `0.0` | its period [s]; 0 → 1.5 × duration |
| `--wow` | `0.55:0.12,1.7:0.06,4.3:0.03` | `hz:pct,...` sinusoids, random phases |
| `--ref_db` | `-20.0` | tone rms rel. program rms [dB] |
| `--fade_db` | `0.0` | slow fading depth, peak-to-peak [dB] |
| `--dropouts` | `0` | number of hard dropouts |
| `--dropout_len` | `2.0` | mean dropout length [s] |
| `--program` | `synth` | `synth`, `none` or an audio file path |
| `--prog_db` | `-20.0` | program rms [dBFS] |
| `--prog_lp` | `0.0` | program bandwidth [Hz]; 0 → 0.45 × sr |
| `--snr_db` | `60.0` | hiss rel. program rms (tone rms if no program); `inf` = off |
| `--mech` | `""` | mechanical components `fspec:a1/a2/a3,...`; fspec = Hz, `T` (take-up reel), `S` (supply), `<Hz>-T`; amplitudes in % |
| `--reels` | `0.1905:31.5e-6:0.0288:0.0815` | v [m/s] : tape thickness [m] : R take-up₀ : R supply₀ [m] |
| `--mech_var` | `0.0` | L3 relative amplitude variation (rms) |
| `--mech_phase` | `0.0` | L3 phase wander (rms rad) |
| `--mech_tc` | `60.0` | L3 time constant [s] |
| `--mech_fdrift` | `0.0` | relative frequency wander of fixed components (rms) |
| `--partials` | `""` | on-tape interfering partials `f0[~f1]:dB:t0:t1,...` (dB rel. ref tone) |
| `--pair_hz` | `0.0` | AM beat pair: channel spacing [Hz]; partner at 2·pair_hz − ref_hz; 0 = off |
| `--pair_db` | `-10.0` | partner level rel. ref tone [dB] |
| `--lines` | `""` | extra steady lines `f:dB[:post],...` (true Hz, dB rel. ref tone; `post` = added after the tape, not speed-modulated) |
<!-- drift:synth:end -->

| preset | settings |
|---|---|
| `fm19` | ref 19000, ref_db −20, offset 0.8, snr 50 |
| `am9w` | ref 9000, ref_db −30, offset −1.5, fade 10, dropouts 2, snr 40 |
| `am9` | ref 9000, ref_db −24, offset 1.5, fade 6, dropouts 1, snr 45 |
| `reel` | ref 9000, ref_db −20, offset −1.13, sr 44100, fade 4, wow `0.63:0.04,3.52:0.06,5.03:0.06,7.47:0.3,14.94:0.1`, snr 45 |
| `clean` | ref 9000, ref_db 0, offset 1.2, program none, snr inf |
| `mech` | 720 s, 44.1 kHz mono, capstan 7.46 Hz, roller 1.484 Hz, clutch 4.15 Hz − take-up, reels at 19.05 cm/s, L3 variation, 4 dropouts, partials |

## 8. Outputs
### 8.1 Files
| file | default path | written | content |
|---|---|---|---|
| audio | `-o` | always (non-survey) | same sr and channels as input; `.wav` f32, `.flac` 24 bit. FLAC is encoded in-process: f32 → 24 bit exactly as ffmpeg (`rint(x·2^31)`, clip, `>> 8`), so the decoded samples equal the ffmpeg path; STREAMINFO with total samples and MD5; vendor `tapespeed <version>`, no padding block; size −0.07 % vs ffmpeg. `TAPESPEED_FLAC=ffmpeg` uses the ffmpeg encoder again. Lossy containers without an explicit codec (ffmpeg's defaults are 128 kb/s MP3/AAC, 96 kb/s Opus): `.mp3` LAME VBR `-q:a 2` (~190 kb/s, LAME's recommended transparent preset; VBR spends fewer bits on band-limited AM material), `.m4a/.aac/.mp4/.m4b` AAC 128 kb/s per channel, `.opus` Opus 80 kb/s per channel (48 kHz: ffmpeg resamples), `.ogg/.oga` Vorbis `-q:a 6` (~192 kb/s) if ffmpeg has libvorbis, else ffmpeg's FLAC-in-Ogg (lossless). Lossy outputs print `note: X is lossy (MP3, LAME VBR -q:a 2, ~190 kb/s); .flac or .wav give a lossless master` |
| curve | `<out stem>_curve.csv` | twice: before correction, then rewritten with `extra_notches` | §8.2 |
| report | `<out stem>_report.json` | at the end | §8.3 |
| speed plot | `<out stem>_speed.png` | unless `--no-plot` | §8.4 |
| model plot | `<plot stem>_model.png` | only with explicit `--plot` | §8.4 |
| survey | `<stem>_survey.json`, `<stem>_survey.png` | `--survey` | §8.5 |
| synth truth | `<out>_truth.csv` | `synth` | curve format, confidence 1, valid 1, meta = synth fields |
| spill | in `--tmp-dir` | spill mode | unlinked at creation; never left behind |

### 8.2 Curve CSV (`tapespeed-curve v1`)
```
# tapespeed-curve v1
# ref_hz=9000.0
# key=value                      (one line per meta key, insertion order; floats in shortest round-trip form: 30.0, 1e-05, nan, True)
t_s,ratio,confidence,valid
0.000000,1.016689233,1.0000,1
```
| column | format | meaning |
|---|---|---|
| `t_s` | `%.6f` | input-file time [s]; uniform grid, dt = 0.005 s (200 Hz), from 0 over the whole file |
| `ratio` | `%.9f` | final speed s(t): > 1 = played too fast. Always the best estimate (also in gaps) |
| `confidence` | `%.4f` | 0…1 lock quality (set to 0 on invalid samples by the plain path and the long-gap fill; aided path: its own confidence) |
| `valid` | `0`/`1` | 1 = trusted measurement; 0 = gap fill or low confidence |

Full-precision ratio: `TAPESPEED_DUMP_RATIO=path` (§14). Meta keys of the one-shot pipeline, in file order:

| key | present | meaning |
|---|---|---|
| `ref_hz` | always | `--ref` |
| `sr` | always | input sample rate |
| `channel` | always | `--channel` |
| `bw` | always | pass-1 bw (`auto:N` or N) |
| `passband_hz` | always | flat passband of the pass-1 curve |
| `search` | always | `--search` |
| `source` | always | input path |
| `search_centre_hz` | always | ref × play factor |
| `sw_rho`, `sw_spans`, `sw_s`, `sw_major` | `pair`, ρ found, repair on | ρ = F_hi/F_lo; repaired spans; their total [s]; beat tracked after repair (`lo`/`hi`) |
| `aided_B`, `refine` | aided path | residual band [Hz]; refinement passes (2) |
| `model_holdout_rms` | plain path with hold-out | `model E analyze E pchip E n N` gap-fill hold-out RMS |
| `model_path` | always | `aided`, `plain`, `off` |
| `model_ev` | always | EV of L3 on valid, non-rejected samples (4 dp); `nan` if skipped |
| `model_ev_l2`, `model_min_ev`, `model_comps`, `model_rejected_pct`, `model_gapfill`, `model_Ta` | model fitted | EV of L2; threshold; component count; jump-test rejected [%]; `model`/`analyze`; L3 knot spacing [s] |
| `ref_gaps`, `ref_gap_s`, `ref_gap_fill`, `ref_gap_L1`, `ref_coverage_pct` | gaps ≥ 20 s | spans `a-b ...`; total [s]; `edge-linear / L1 ramp`; L1 used; share with reference [%] |
| `fade_rho`, `fade_fm_ratio` | always | Spearman(line dip, residual FM); deep-dip/flat residual-FM ratio (`nan` if not computable) |
| `corr_max_hz` | always | band limit [Hz] or `none` |
| `corr_policy` | always | reason (§5.4) |
| `corr_keep_model_hf` | always | 1 = periodic model part kept above the limit |
| `coh_line_hz`, `coh_bands` | second line | line [Hz]; `0-0.5:c 0.5-6:c 6-25:c` coherence of the two pass-1 curves |
| `abs_mode` | always | `ref`, `none`, `assume:HZ`, `line:HZ`, `value:R`, `percent:P`, `curve:PATH`, `pair:HZ` |
| `play_factor` | always | F = play/tape |
| `line_line_valid`, `line_common`, `line_q`, `line_q_std_pct`, `line_q_drift_pct`, `line_n_blocks`, `line_hz`, `line_disagree_pct` | `line:` | second-line valid share; common valid share; q = r1/r2 (Huber); spread [%]; 60 s-block drift ptp [%]; blocks; line; primary offset from nominal [%] |
| `pair_hz`, `pair_lo_hz`, `pair_hi_hz`, `pair_delta_hz`, `pair_side`, `pair_offset_hz`, `pair_ambiguity` | `pair` | spacing; true lower/upper beat; d = (hi − lo)/2; `primary is the lower/upper beat`; station offset if the primary is the lower neighbour's beat; ambiguity verdict |
| `pair_prom_db`, `pair_pres`, `pair_k`, `pair_n_cand`, `pair_n_rejected`, `pair_n_blocks` | `pair` | `primary/partner` prominence [dB]; partner presence; k̂; candidates; rejected; blocks with both lines |
| `pair_blk_ftrue_std_ppm`, `pair_blk_ftrue_ptp_ppm`, `pair_blk_delta_ptp` | `pair`, ≥ 2 blocks | per-block f_true spread rms/ptp [ppm]; d ptp [Hz] |
| `pair_xcheck_ppm` | `pair` + cross-check line | `HZ:±ppm` |
| `abs_curve` | `curve:` | CSV path |
| `abs_l1`, `abs_l1_pct`, `abs_ref_true_hz` | always | mean of s; (L1 − 1)·100; implied true frequency of the tracked line = ref·r̄/L1 (abs floats rounded to 7 dp) |
| `tape_speed_given`, `play_speed`, `tape_speed_inferred` | always | `19.05 cm/s` or `none`; speeds consistent with reel/capstan physics (comma list) or `none` |
| `notch_hz`, `notch_q` | always | reference notch [Hz] (2 dp) and Q, or `none` |
| `lowpass_hz` | always | lowpass [Hz] or `none` |
| `extra_notches` | always (2nd write) | `HZ@all` or `HZ@t0-t1,...` per notch, or `none` |

### 8.3 Report JSON (`_report.json`)
One value per line (indent 1), keys in this order, `NaN`/`Infinity` tokens, `null` for none.

| key | present | content |
|---|---|---|
| `analyze` | always | `{bw, valid, s}` |
| `switch` | pair repair ran | `{sw_rho, sw_spans, sw_s, sw_major, sw_sigma_pct, sw_list}` |
| `pair`, `pair_s` | `pair` | `{hz, f_primary, prom_primary, r0, candidates[{f, prom, s, l1_pct, delta, side, why, pres, drift, k_hat, gain_db, k_contrast, mirror_db, cochan, comb, sub_d}], xcheck, n_blocks, sub_lines, ambiguity, best, s, f_lo, f_hi, delta, side, f_true, offset_hz, blocks}`; seconds |
| `model` | always | `{mode, min_ev, [constrain], [physics{speeds, consistent, has_reel, v_given, v_used, geometry}], ev, ev_l2, r0, rejected, n_comps, comps[{name, kind, h, f_start, f_end, law, amp_l2_pct, amp_l3_rms_pct}], path, reason, gapfill, [aided_B, refine], [holdout], [aided]}` |
| `ref_gaps` | gaps ≥ 20 s | `{spans, total_s, L1}` |
| `coherence`, `coherence_final` | second line | `{n_seg, coh{band: value}, rms1, rms2, rms_diff}` (pass-1 curves; final vs line) |
| `fading` | always | `{rho, fm_ratio, deep_frac, rho_curve, curve_ratio, valid_in_deep, rho_final, ratio_final, rho_model, ratio_model, seg_rho[], seg_rho_median, seg_high_frac}` or `{error}` |
| `corr_max_hz`, `corr_policy`, [`corr_keep_model_hf`] | always / if true | §5.4 |
| `abs` | always | the `abs_*`, `play_factor`, `line_*`, `pair_*` values unrounded |
| `runtime_s` | always | seconds up to L1 |
| `notch` | always | `{hz, q}` |
| `plot` | speed plot drawn | `{wow_pct, flutter_pct, l1_pct}` |
| `auto_notch_s` | `--auto-notch` | seconds |
| `extra_notches` | always | `[{hz, spans[[t0, t1]], src: manual|auto}]` |
| `correct_s`, `total_s`, `duration_s` | always | seconds; input duration |

### 8.4 Plots (PNG, embedded DejaVu Sans)
| plot | size | panels |
|---|---|---|
| `_speed.png` | 1430 × 975 | timeline: speed deviation [%] pass 1 (L1-aligned) vs final, 2 s trend, L1 level, invalid / no-reference spans shaded; confidence strip (filled area from 0 to the mean confidence per pixel column, dropouts read as empty); 5 s detail; Welch spectrum of the deviation with model component frequencies; wow rms (0.5–6 Hz), flutter rms (6–25 Hz) |
| `_model.png` | 1320 × 880 | pass 1 vs refined curve, model part, component frequency laws |
| `_survey.png` | – | spectrum overview, line-prominence map, presence map |

### 8.5 Survey JSON (`_survey.json`)
Top-level keys (measured run): `file, info, blocks, channels, search, play, coherence, candidates, primary, bandwidth,
runtime_s, recommend, plan`, plus `pair` when the pair check ran.

| key | content |
|---|---|
| `info` | codec, format, sr, channels, duration, bit_rate, lossy |
| `blocks` | `[t0, len]` s |
| `candidates[]` | `f, cls, label, nominal, codec_risk, ch, low, n, nb, presence, lock_blocks, prom, cn0, pers, wow_pct, spread_hz, spread_pct, blocks{i: f, cn0, prom}, channels{ch: n, lock, cn0}, ratio, offset_pct, sigma25_pct, wow_corr, tape, primary, verdict, trial_valid, id` |
| `primary` | candidate id |
| `coherence[]` | `[a, b, t0, dur, common, n_seg, coh{band: value}, tape]` |
| `bandwidth` | nyquist, cliff_hz, levels_db |
| `pair` | hz, trials[], found, n_agree, n_trials, delta, f_true, f_true_spread_ppm, f_lo, f_hi, side, prom, xcheck_ppm, why |
| `recommend` | `{ref|channel|abs|check_line: [value, reason], cmd}` |
| `plan` | `{steps[[name, text]], warnings[], resolved{ref, ref_src, centre, play, search_hz, bw, channel, line_hz, corr_limit, model, abs_mode, abs_l1, true_hz, notch_hz, lowpass, extra_notches, out_len_s, outputs}}` |

### 8.6 Stdout lines (one-shot run, in order)
| line | condition |
|---|---|
| `memory: mode ram\|spill (DIR), bw k/5 parallel, est peak X GiB, budget Y GiB (SRC, Z GiB available), jobs J, threads T[; WARNING: one bw candidate at a time already exceeds the budget]` | always first. SRC = `80 %`, `--max-ram`, `--max-ram capped at physical RAM / cgroup`, `RAM unknown` |
| `warning: --channel S on a mono file: analysing the mono signal (as --channel mid)` | `--channel L/R/S` on a mono input |
| `note: x.mp3 is lossy (MP3, LAME VBR -q:a 2, ~190 kb/s); .flac or .wav give a lossless master` | lossy `-o` |
| `analyze 9000 Hz (centre 9000): bw auto:6, valid 86.6 %` | always |
| `analyze 15625 Hz: bw auto:25, valid 87.9 %` | second line |
| `pair (9000 Hz spacing, \|d\| <= 300 Hz): <ambiguity>` + candidate table | `pair` |
| `beat switching (rho R[, from the ghost line pair]): N span(s), S s on the other beat, repaired to the lo\|hi beat` | `pair`, ρ found |
| `pair on the repaired curve: ...` + table | spans > 0 |
| `model: path P (reason)` + EV, component table, `tape speed: ...`, capstan/reel geometry, `constrained: {...}` | always |
| `WARNING: no usable reference for S s (P % of the file: spans); there only the slow level is corrected ...` | gaps ≥ 20 s |
| `evidence: fading rho R (deep-dip/flat residual FM Xx)[, coherence with HZ Hz: bands] -> correction band full\|<= F Hz (policy)` | always |
| `L1 (mode): mean speed L (±P %), implied true line F Hz[; line ...][; pair ...]; tape speed given G, inferred I` | always |
| `plot PATH` | plot drawn |
| `curve PATH: path P, EV E, valid V %, ratio a..b` | always |
| `reference tone notched at F Hz (Q 30)` / `kept (--keep-ref)` / `not needed: in lowpass stopband (>= F Hz)` | always |
| `line F Hz: steady (contrast C dB, mid M), kept\|removed by lowpass` | `--auto-notch keyed`, steady line |
| `notch F Hz (manual\|auto, width W Hz): whole file\|S s in N span(s)[, present S s, SNR D dB, keying contrast C dB]` | per extra notch |
| `notch F Hz (src): skipped, covered by lowpass\|reference notch` | skipped notch |
| `lowpass F Hz (transition ±W Hz, 80 dB)` | `--lowpass` |
| `audio PATH: D s -> D' s @ SR Hz; T s total (M min per hour of audio)` | always last |

`TAPESPEED_PROF=1` adds stderr lines `prof <stage> wall ... cpu ... (∥x) heap_pk ... rss_pk ... fp_pk ... t ...`.

## 9. Recipes
| material | command | notes |
|---|---|---|
| first look at any file | `tapespeed rec.mp3 --survey` | read RECOMMEND `cmd:` and run it |
| AM aircheck, 9 kHz beat (Radio Caroline) | `tapespeed rec.mp3 --ref 9000 --abs pair --check-line 15625 --lowpass --auto-notch -o rec.flac` | pair midpoint = offset-independent L1; TV whistle cross-checks it and drives coherence; lowpass 8 kHz removes beat + neighbour sidebands (Caroline program −20…−30 dB at 7–8 kHz, so 8 kHz keeps it); reference notch then skipped (in stopband) |
| AM, no partner beat found | `--ref 9000 --abs line:15625 -o rec.flac` | L1 from the PAL TV whistle (receiving site, independent of RF path). Caroline 790415: beat 8934.1 Hz = −0.73 % off 9000, constant (drift 0.009 % over 44 min) |
| AM, beat frequency known | `--ref 9000 --abs assume:8934 -o rec.flac` | e.g. from an earlier `line:` or `pair` run of the same station/date |
| AM, absolute pitch unknown | `--ref 9000 --abs none -o rec.flac` | average speed kept, only variations corrected |
| AM, US raster | `--ref 10000 --abs pair:10000` | channel spacing 10 kHz [unverified on real data] |
| weak AM: fading, co-channel | add `--weak-rf` | curve ≤ 6 Hz, periodic model components above kept. Caroline c: 6–25 Hz 0.182 → 0.029 %; b: 0.171 → 0.109 % |
| keyed CW interferer | `--auto-notch` (keyed) or `--notch-at 8021` | Caroline 790415: CW 8021.1 Hz (Scheveningen Radio PCH, minutes 27–31) −34 dB, notched 220 s; CW contrast 36.5 dB / mid 0.12 vs steady carriers 8.6–15.2 dB / 0.32–0.34 |
| FM recording, 19 kHz pilot | `tapespeed rec.flac --ref 19000 -o out.flac` (`--channel L` if only one channel carries it) | `--abs ref` is exact: the pilot is a broadcast standard. Pilot notched at its corrected frequency. Synthetic fm19: err 8.3e-6 RMS |
| TV sound / PAL line whistle as primary | `--ref 15625` (NTSC `15734`) | standard frequency, `--abs ref` exact; survey class `TV line` |
| reel-to-reel, heavy 7.5 Hz capstan flutter | `--ref 9000 --tape-speed 19.05cm -o out.flac` | keep `--bw auto` (picks 25/45 Hz) and `--model-band 20` (15–25 Hz flutter is real, coherent on two lines); physics infers reel/capstan geometry. Synthetic `reel` (7.47 Hz 0.3 %): valid 69 → 99 %, err 8.3e-4 → 1.6e-4 |
| wrong-speed transfer | `--tape-speed 9.53cm --play-speed 19.05cm` | F = 2; search at ref·F |
| batch | `tapespeed /in/*.mp3 --ref 9000 --auto-notch -o /out/` | 17 Caroline files, 18.4 h: 280 s wall, 4 jobs. Options apply to every file: `--abs pair` fails files without a partner (batch exit 1); per-file `--abs` needs separate runs (survey first) |
| batch, custom names | `tapespeed a.mp3 b.mp3 --ref 9000 -o out/{stem}.wav --curve curves/{stem}.csv` | directories must exist |
| long file, small machine | `tapespeed 2h.flac --ref 9000 -o o.flac --max-ram 6G --tmp-dir /fast/ssd` | input spills (≈ 1.27 GB/h stereo 44.1 k on disk), candidates serialised; results bit-identical |
| deterministic logs | `TAPESPEED_FIXED_CLOCK=1 tapespeed ...` | timings 0 for diffing |

## 10. Accuracy and limits
### 10.1 Measured accuracy
Synthetic, 60 s, vs the truth curve written by `tapespeed synth` (default pipeline, presets of §7.4):

| preset | valid % | err RMS present | err RMS gaps | length err [samples] |
|---|---|---|---|---|
| am9w (−30 dB, fades, dropouts) | 87 | 1.6e-4 | 2.4e-4 | +14 |
| am9 | 91 | 1.1e-4 | 1.6e-4 | +3 |
| reel | 99 | 1.6e-4 | 3.4e-4 | −1 |
| fm19 | 99.5 | 8.3e-6 | 1.3e-5 | 0 |

False-valid: 0.09 % (am9w), 0.42 % (reel). `mech` 720 s: valid 93 %, err 2.2e-4, false-lock 0.7 %, speed inferred 19.05 cm/s,
joint reel fit h 31.46 µm (true 31.5). 2 h `reel` synth: err RMS 1.06e-4 all / 1.00e-4 valid, valid 98.7 %.

Real data, Caroline 790415_1930 (43 min, `--abs pair --check-line 15625`): L1 0.998122 vs `line:15625` 0.998113 (+8.4 ppm);
TV cross-check +7.0 ppm (batch: +7.2); block spread f_true 14.9 ppm rms; coherence with the TV line 0.996 / 0.998 / 0.995
(0–0.5 / 0.5–6 / 6–25 Hz); valid 75.1 → 88.8 %. Batch of 17 Caroline airchecks: pair found in 4,
5 files flagged DISCARD (reference in 0–21 % of the file).

Results do not depend on the thread count: curves and audio are bit-identical for `--threads` 1…10.

Signal processing figures: resampler residual −109…−117 dB (tests assert < −90/−80 dB); reference notch
Q 30: line −26…−52 dB, program ±1 kHz −0.8…−1.1 dB; lowpass 8 kHz: 7.5 kHz unchanged, 8.5–11 kHz −88…−105 dB.

### 10.2 Limits
| limit | value / behaviour |
|---|---|
| lock threshold | no lock below ~21 dB-Hz (analyze scale) |
| file edges | ~0.15 s held |
| model step | needs ≥ 60 s and pass-1 valid ≥ 20 % |
| long gaps | ≥ 20 s without reference: only the slow level is corrected, no wow/flutter correction there |
| speed inference | needs both reel laws over ≥ ~10 min; 5-min files ambiguous |
| tape origin of FM | verified only with a second independent line; without it the aided path's 20 Hz residual band rests on EV |
| fading calibration | two tape-proven files; threshold ρ 0.9 above every tape-proven value (max 0.86) |
| beat-switch repair | `--abs pair` only; switch placement ±0.25 s |
| `pair` | assumes both neighbours on the exact raster; side/sign of station offset undecidable; thresholds from one real file + synthetic |
| survey | sees only sampled blocks; frame wow resolves < ~5 Hz |
| analyze memory | bw candidates dominate (4.7–5.1 GiB/h with 5 in parallel); no time-chunked analyze, see §12 limits |

## 11. Performance
Apple M1 Max (8 P + 2 E), 64 GiB, rustc 1.97.1, ffmpeg 9.0.2, light background load (`bench/results.md`).

| run | wall s | min/h |
|---|---|---|
| 5 min mono WAV `--ref 9000 --abs line:15625` | 2.64 | 0.528 |
| 43 min `--ref 9000` | 19.36 | 0.447 |
| 43 min `--abs line:15625 --lowpass` | 26.88 | 0.620 |
| 43 min `--abs pair --auto-notch` | 30.01 | 0.692 |
| `--survey`, 17 files (18.4 h) | 19.59 | 0.0177 |
| batch 17 files `--auto-notch` (4 jobs) | 280.3 | 0.254 |

| `--threads` (43 min pair) | 1 | 2 | 4 | 8 | 10 |
|---|---|---|---|---|---|
| wall s | 103.5 | 59.8 | 39.2 | 30.1 | 29.2 |
| max RSS GiB | 3.45 | 4.52 | 5.52 | 5.57 | 5.43 |

Outputs bit-identical for threads 1…10. Largest stages (43 min pair, 10 threads): final correct + notches into the
encoder 6.4 s, auto-notch keying 3.7 s, an.tracks 3.5 s, model.aided 3.3 s, auto-notch detect 3.2 s, MP3 decode 2.4 s.
FLAC: the in-process encoder (2.9 s CPU per 43 min, parallel) took the final pass from 6.7–11.1 to 4.3–6.7 s (`ref`) and 7.3–7.6 to 5.4–5.5 s (`pair`) at 8–10 threads, under loadavg 8–44 (`bench/results.md` §9); before, ffmpeg's single-threaded encoder cost ≈ 3.2 s of the 19 s `ref` run (`.wav` output: 15.5 s). The table above predates it.
Peak memory, 43 min stereo: live heap 4.25 GiB (`ref`), 5.70 (`line:` + lowpass), 4.28 (`pair`).
Regression guard: `bench/check.sh` (fails at > 1.10 × `bench/baseline.json`).

## 12. Memory planner
Code: `crates/tapespeed/src/plan.rs`; design and measurements: `docs/MEMORY.md`.

| symbol | formula (bytes) |
|---|---|
| frames | (T + 1 s)·sr, T = ffprobe duration (else size/bit rate, else size/(4·sr·ch)) |
| X (input) | 4·frames·ch |
| N_d, N_k | stage-1 samples; samples of bw candidate k at its fs_p (`analyze::api::sizes`); with a second line the larger analyze pass counts |
| fixed | 24·N_d + 20·ΣN_k (+ 16·N_d of the second line during pass 1: both basebands are streamed during the decode) |
| own | 420·N_max (largest candidate alone) |
| cand_k | 250·N_k per further concurrent candidate |
| oth | 180 kB·T (pair, aided, evidence, correct) |
| peak(x, an) | 1.1·(100 MiB + x + max(fixed + an, oth)) |
| B | `--max-ram` value (≤ physical RAM / cgroup limit), else 0.8 × available (available capped by cgroup headroom); RAM unknown → unlimited |
| jobs | min(⌈threads/2.5⌉, files), ≥ 1 |

| step | condition | mode | bw parallel |
|---|---|---|---|
| 1 | peak(X, all) ≤ B | `ram` | 5/5 |
| 2 | spill possible and peak(0, all) ≤ B | `spill` | 5/5 |
| 3 | else | `spill` if possible, else `ram` | 1 + as many further candidates (largest first) as fit; `WARNING` if even 1 exceeds B (runs anyway) |

Spill possible = tmp dir exists and free space > 1.05·X. Spill file: unlinked f32 file, mmap read-only; time ±0 measured.
Batch: one shared budget; a job reserves peak(x, own) before it starts (blocking; an oversized job runs alone); further
candidates take budget only if free (non-blocking). Measured estimate vs live heap: 5.3 vs 4.28 GiB (43 min pair, +24 %).
All modes give bit-identical curves and audio. Not implemented: re-decode mode, time-chunked analyze.
Expected limit with spill + 1 candidate: ~2.65 GiB per hour of audio → 16 GB machine ≈ 3.0 h per file (MEMORY.md §6, estimate).

## 13. Exit codes and messages
| exit | cause |
|---|---|
| 0 | success; `--help`; `--version` |
| 1 | runtime error (stderr `Class: message` or a plain message), any batch file failed, every `synth` error, `--max-ram` parse error |
| 2 | usage error from clap (unknown/ambiguous option, invalid value, missing `<input>`), `--ref and -o are required (except with --survey)`, an output that is an input file (§5.5), batch naming errors (§5.3) |

| stderr message | cause |
|---|---|
| `error: unexpected argument '--max' found` | ambiguous prefix (exit 2) |
| `error: invalid value 'x.wav' for '--auto-notch [<AUTO_NOTCH>]'` | optional-value option before the input (exit 2) |
| `tapespeed: error: --ref and -o are required (except with --survey)` | (exit 2) |
| `tapespeed: error: x.flac: audio output x.flac is the input file x.flac: refusing to overwrite an input ...` | output = input (exit 2, §5.5) |
| `tapespeed: error: -o all.flac: with several inputs -o is a directory or contains {stem} ...` | batch `-o` (exit 2, §5.3) |
| `OSError: ffprobe failed (exit Some(1)): PATH: No such file or directory` | unreadable input / no ffprobe |
| `OSError: io: No such file or directory (os error 2)` | output directory missing |
| `--max-ram: cannot parse "12Q" (e.g. 12G, 512M, 50%)` | bad SIZE |
| `--play-speed needs --tape-speed (the recording speed)` | – |
| `ValueError: tape speed 'x': use e.g. 19.05cm or 7.5ips` | bad speed |
| `ValueError: --abs 'x': ref \| none \| assume:HZ \| line:HZ \| value:R \| percent:P \| curve:PATH \| pair[:HZ]` | bad `--abs` |
| `RuntimeError: line HZ Hz: too few samples valid in both lines (line valid V %, common C %)` | `line:` |
| `--abs pair: no partner line passing the criteria within \|d\| 5-300 Hz ...` / `no unambiguous partner` + table | `pair` |
| `ValueError: taps must be >= 0` | – |
| `AssertionError: non-finite ratio from analyze` / `non-finite speed curve` | internal check |
| `KeyError: 'ref_hz'` | `correct --notch` on a curve without `ref_hz` |
| `N of M files failed` | batch summary on stderr |

Warnings (stdout): `WARNING: no usable reference ...` (§8.6), `; WARNING: one bw candidate at a time already exceeds the budget` (memory line), survey `WARNINGS` section.

## 14. Environment variables
| variable | effect |
|---|---|
| `TAPESPEED_TMPDIR` | spill dir when `--tmp-dir` is not given |
| `TAPESPEED_PROF` | non-empty and ≠ `0`: per-stage wall/CPU/RSS/footprint (and live heap with feature `memprof`) on stderr; single-file runs (batch interleaves) |
| `TAPESPEED_FIXED_CLOCK` | set: all timings in stdout/report/survey are 0.0 (byte-comparable output) |
| `TAPESPEED_DUMP_RATIO` | path: final ratio as raw little-endian f64 (no CSV rounding) |
| `TAPESPEED_FLAC` | `ffmpeg`: encode `.flac` outputs with ffmpeg instead of the in-process encoder |
| `RAYON_NUM_THREADS` | thread count when `--threads` is not given (rayon default) |
| `TOL`, `N`, `TS_CHECK_DIR`, `TS_SKIP_MICRO` | `bench/check.sh` knobs |
| `TS_BLESS` | tests: rewrite the stored survey reference results in `crates/ts-core/tests/data/k` from the current code (review the diff before committing) |

## 15. Tests
`./ci.sh` (= `tools/fetch-testdata.sh`, `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --release`) needs Rust, ffmpeg, and for the test data `curl`, `zstd`, `tar`, `sha256sum`/`shasum` (POSIX sh; macOS: `brew install zstd`, Debian/Ubuntu: `apt install curl zstd`).

| kind | where | checks |
|---|---|---|
| unit tests | `crates/*/src` | primitives, parsers, formats, planner, CLI naming |
| synthetic signals | `crates/*/tests` | `tapespeed synth` / generated tones with a known speed curve: tracking error, model, correction, notches, CLI behaviour |
| stored reference results | `crates/*/tests/data` (fetched, below) | inputs plus expected outputs (`.npy`, `.json`, `.csv`, survey text) of DSP primitives, analyze stages, model fits, interference detection, survey cases; compared within stated tolerances. Numerical conventions follow the SciPy/NumPy definitions where noted (e.g. `welch`, `filtfilt`, `percentile`) |
| thread invariance | several | results bit-identical for 1 and N threads |
| doc drift | `crates/tapespeed/tests/manual_drift.rs` | §17 |

The survey cases in `crates/ts-core/tests/data/k` are regenerated with `TS_BLESS=1 cargo test --release -p ts-core --test survey_cases` (then pack a new data version, below).

**Stored test data.** `crates/*/tests/data` (807 files, 90 MB; 56 MB as `.tar.zst`) is git-ignored and
comes from a versioned archive described by `testdata.lock` (committed):

| key | meaning |
|---|---|
| `version` | data version N |
| `file` | `tapespeed-testdata.tar.zst` (the same name in every version directory) |
| `size`, `sha256` | of the archive; checked before extracting |
| `url` | mirrors, tried in order: `https://ib-erckenbrecht.de/files/tapespeed/v<N>/tapespeed-testdata.tar.zst`; more `url=` lines may be added |

- **Fetch**: `tools/fetch-testdata.sh` does nothing when `.testdata.stamp` holds the lock's sha256. Otherwise it takes the archive from `$TAPESPEED_TESTDATA` (local file, offline override), else from `dist/v<N>/` (left by a pack or an earlier download), else downloads it from the `url=` mirrors in order; verifies size and sha256 (a mismatch is rejected); extracts to a scratch directory, replaces each `crates/*/tests/data` from the archive and writes the stamp. If no mirror delivers a valid archive it stops with the list of URLs tried and the `TAPESPEED_TESTDATA` hint.
- **Missing data fail, they do not skip**: every test that reads stored data goes through `crates/testdata.rs` (`testdata::dir`), which panics with `stored test data missing ... run tools/fetch-testdata.sh` (or `outdated`, when the stamp differs from the lock). Tests on synthetic signals do not need the archive. (Checked: the only stored arrays that are plain formulas are the 39 time axes `t = k · 5 ms`, 3.6 MB raw but 65 KB compressed; everything else is reference output or noise input recorded from the reference implementation and cannot be regenerated bit for bit, so nothing was moved to generation.)
- **New data version**: change the data in `crates/*/tests/data`, run `tools/pack-testdata.sh <N+1> --lock` (reproducible: sorted list, fixed mtime/owner/modes, ustar, `zstd -19 --long`; two runs give the same sha256), upload `dist/v<N+1>/tapespeed-testdata.tar.zst` to `/files/tapespeed/v<N+1>/` on the server (and any other mirror), then commit the updated `testdata.lock` (version, size, sha256 and the `/v<N+1>/` URLs change together). Old version directories stay on the server for old commits.
- **CI**: `.github/workflows/ci.yml` caches `dist/` under the key `testdata-<sha256 from testdata.lock>` and runs `tools/fetch-testdata.sh` before the tests.
Long tests are `#[ignore]` and run with `./ci.sh --full`.

## 16. Glossary
| term | definition |
|---|---|
| ratio r | f_apparent/f_true of the reference line; r > 1 = tape plays fast (pitch high). Curve column `ratio` = final s |
| τ, T(t) | original (true-speed) time, T = ∫ r dt |
| L1 | absolute speed level, mean of s (`abs_l1`) |
| L2 | drift + periodic components (capstan, rollers, reels, clutch) with fixed parameters |
| L3 | slow variation of the L2 amplitudes/phases, knots every `--model-ta` s |
| EV | share of var(r − r0) explained by L3 (`model_ev`; L2: `model_ev_l2`) |
| aided / plain / off | model path: model-aided re-tracking in ±B Hz / pass-1 curve / no model |
| wow | speed-variation rms in 0.5–6 Hz [%] (`plot.wow_pct`) |
| flutter | speed-variation rms in 6–25 Hz [%] (`plot.flutter_pct`) |
| coherence bands | 0–0.5, 0.5–6, 6–25 Hz; ≥ 0.5 = tape-caused |
| C/N0 | carrier-to-noise density ratio of the line [dB-Hz]; survey scale ≈ analyze − 4 dB |
| d, δ | station offset from the raster in `pair`: half the beat difference, (F_hi − F_lo)/2 true (`pair_delta_hz`) |
| ρ | F_hi/F_lo of the beat pair, factor of a beat switch |
| F, play factor | playback/recording tape speed |
| valid | 1 = trusted measurement; 0 = gap fill or low confidence |
| confidence | 0…1 lock quality per sample |
| bw | pass-1 PLL loop bandwidth (6/12/25/45/80 Hz candidates) |
| fs_d, fs_p | stage-1 baseband rate; per-candidate PLL rate |
| ram / spill | input decoded into RAM / into an unlinked mmap temp file |
| jump test | rejection of ≥ 0.1 % level steps within 1 s (false locks) |

## 17. Doc drift check
`cargo test -p tapespeed --release --test manual_drift` (part of `./ci.sh`). It fails when:
- the option set in §5.1 differs from the long options printed by `tapespeed --help`, or from the clap definition;
- a §5.1 default cell differs from the clap default (`` `X` ``, `` off; bare: `X` `` for optional values, `off` for flags, `–` otherwise), or an enum type cell differs from the possible values;
- the §7.2/§7.3 tables differ from `tapespeed analyze --help` / `tapespeed correct --help` (names and `[default: X]`);
- the §7.4 table differs from the synth fields and defaults in the truth-CSV header of `tapespeed synth -o x.wav`, or in `tapespeed synth --help`;
- `tapespeed --help` does not show a clap default as `[default: X]`.
Rows are read between the `<!-- drift:NAME:begin/end -->` markers. Prose and the other tables are not checked.
