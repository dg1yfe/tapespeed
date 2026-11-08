#!/bin/bash
# Performance regression guard (~1-2 min).
#
#   bench/check.sh            run the subset, compare with bench/baseline.json, exit 1 on a
#                             median > baseline * (1 + TOL) (TOL default 0.10)
#   bench/check.sh --update   run the subset and write the medians as the new baseline
#
# Subset: macro = `tapespeed IN --ref 9000 --abs line:15625` on a synthetic 5 min mono reel
# recording with a 15625 Hz line (generated once with `tapespeed synth`, N runs, median wall);
# micro = criterion 1-thread kernels (m3_micro: correct 10 s stereo, het_decimate 60 s,
# PLL 1 M; b_prims: welch 512k, stft 60 s; criterion median point estimate).
# Knobs: TOL, N (macro runs, default 5), TS_CHECK_DIR (outputs, default target/bench-check),
# TS_SKIP_MICRO=1 (macro only). Baselines are machine-specific: re-run --update on the machine
# that does the checking. Only meaningful on an otherwise idle machine: the 1-min load average
# is printed and a load > 8 is flagged in the verdict.
set -euo pipefail
export LC_ALL=C
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TOL=${TOL:-0.10}
N=${N:-5}
OUT=${TS_CHECK_DIR:-$ROOT/target/bench-check}
BASE=$ROOT/bench/baseline.json
mkdir -p "$OUT"
cd "$ROOT"
cargo build --release -q -p tapespeed
BIN=./target/release/tapespeed

load1() {
  if [ -r /proc/loadavg ]; then cut -d' ' -f1 /proc/loadavg
  else sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}' | tr , .; fi
}

IN=$OUT/reel5min.flac
if [ ! -f "$IN" ]; then
  $BIN synth --preset reel --duration 300 --channels 1 --lines 15625:-6 -o "$IN" > /dev/null
fi
echo "load before: $(load1)"
: > "$OUT/macro.txt"
for _ in $(seq 1 "$N"); do
  /usr/bin/time -p $BIN "$IN" --ref 9000 --abs line:15625 --no-plot \
    -o "$OUT/line.flac" > /dev/null 2> "$OUT/time.txt"
  awk '/^real/ {print $2}' "$OUT/time.txt" | tr , . >> "$OUT/macro.txt"
done
median() { sort -g | awk '{v[NR]=$1} END {if (NR%2) print v[(NR+1)/2]; else print (v[NR/2]+v[NR/2+1])/2}'; }
declare -a KEYS VALS
KEYS+=("macro/synth_line_wall_s"); VALS+=("$(median < "$OUT/macro.txt")")

if [ "${TS_SKIP_MICRO:-0}" != 1 ]; then
  CA="--warm-up-time 1 --measurement-time 3"
  cargo bench -q -p ts-core --bench m3_micro -- 'm3/(correct_10s_stereo|het_decimate_60s|pll_1M)_1t' $CA > "$OUT/micro.txt" 2>&1
  cargo bench -q -p ts-dsp --bench b_prims -- 'b/(welch_512k_nper4096|stft_60s_44k1_nper4096)_1t' $CA >> "$OUT/micro.txt" 2>&1
  for id in m3/correct_10s_stereo_1t m3/het_decimate_60s_1t m3/pll_1M_1t \
            b/welch_512k_nper4096_1t b/stft_60s_44k1_nper4096_1t; do
    est=$ROOT/target/criterion/$id/new/estimates.json
    ns=$(tr -d ' \n' < "$est" | sed -E 's/.*"median":\{"confidence_interval":\{[^}]*\},"point_estimate":([-0-9.eE+]+).*/\1/')
    KEYS+=("${id}_ms"); VALS+=("$(awk -v v="$ns" 'BEGIN {printf "%.4f", v / 1e6}')")
  done
fi
L1=$(load1)
echo "load after:  $L1"

if [ "${1:-}" = "--update" ]; then
  {
    echo "{"
    echo " \"note\": \"bench/check.sh --update; medians (s / ms); see bench/results.md\","
    echo " \"date\": \"$(date '+%Y-%m-%d %H:%M')\","
    echo " \"rev\": \"$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)\","
    echo " \"loadavg\": \"$L1\","
    echo " \"tol\": $TOL,"
    echo " \"values\": {"
    for i in "${!KEYS[@]}"; do
      sep=","; [ "$i" -eq $((${#KEYS[@]} - 1)) ] && sep=""
      echo "  \"${KEYS[$i]}\": ${VALS[$i]}$sep"
    done
    echo " }"
    echo "}"
  } > "$BASE"
  echo "baseline written: $BASE"
  for i in "${!KEYS[@]}"; do printf '  %-40s %10s\n' "${KEYS[$i]}" "${VALS[$i]}"; done
  exit 0
fi

bad=()
printf '%-40s %10s %10s %7s\n' metric base now ratio
for i in "${!KEYS[@]}"; do
  k=${KEYS[$i]}; v=${VALS[$i]}
  b=$(sed -nE "s|^ *\"$k\": *([-0-9.eE+]+),?$|\1|p" "$BASE")
  if [ -z "$b" ]; then
    printf '%-40s %10s %10s %7s\n' "$k" "-" "$v" "-"
    continue
  fi
  r=$(awk -v v="$v" -v b="$b" 'BEGIN {printf "%.3f", v / b}')
  flag=$(awk -v r="$r" -v t="$TOL" 'BEGIN {if (r > 1 + t) print "FAIL"}')
  printf '%-40s %10s %10s %7s %s\n' "$k" "$b" "$v" "$r" "$flag"
  [ -n "$flag" ] && bad+=("$k")
done
note=$(awk -v l="$L1" 'BEGIN {if (l > 8) printf " (load %.1f > 8: re-run on an idle machine before trusting this)", l}')
if [ ${#bad[@]} -gt 0 ]; then
  echo "REGRESSION: ${bad[*]}$note"
  exit 1
fi
echo "OK (all within +$(awk -v t="$TOL" 'BEGIN {printf "%.0f", t * 100}') %)$note"
