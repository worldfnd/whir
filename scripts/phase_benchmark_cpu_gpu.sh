#!/usr/bin/env bash
set -euo pipefail

out="${WHIR_PHASE_BENCH_OUTPUT:-outputs/phase_benchmark_cpu_gpu.jsonl}"
min_log_size=16
max_log_size=28
folds="1,2,3,4,6"
rates="1,2,3"
phases="commit,sumcheck,e2e"
article_grid=false
common_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      out="$2"
      shift 2
      ;;
    --output=*)
      out="${1#--output=}"
      shift
      ;;
    --min-log-size)
      min_log_size="$2"
      shift 2
      ;;
    --min-log-size=*)
      min_log_size="${1#--min-log-size=}"
      shift
      ;;
    --max-log-size)
      max_log_size="$2"
      shift 2
      ;;
    --max-log-size=*)
      max_log_size="${1#--max-log-size=}"
      shift
      ;;
    --folds)
      folds="$2"
      shift 2
      ;;
    --folds=*)
      folds="${1#--folds=}"
      shift
      ;;
    --rates)
      rates="$2"
      shift 2
      ;;
    --rates=*)
      rates="${1#--rates=}"
      shift
      ;;
    --phases)
      phases="$2"
      shift 2
      ;;
    --phases=*)
      phases="${1#--phases=}"
      shift
      ;;
    --article-grid)
      article_grid=true
      shift
      ;;
    *)
      common_args+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$(dirname "$out")"
: > "$out"

cargo build --release --bin phase_benchmark
CARGO_TARGET_DIR=target/metal cargo build --release --features metal --bin phase_benchmark
cargo build --release --bin phase_benchmark_report

IFS=, read -r -a fold_values <<< "$folds"
IFS=, read -r -a rate_values <<< "$rates"
IFS=, read -r -a phase_values <<< "$phases"

cpu_bin="target/release/phase_benchmark"
gpu_bin="target/metal/release/phase_benchmark"

for log_size in $(seq "$min_log_size" "$max_log_size"); do
  for fold in "${fold_values[@]}"; do
    if [[ "$fold" -gt "$log_size" ]]; then
      printf 'skip n=%s fold=%s: fold must be <= n\n' "$log_size" "$fold" >&2
      continue
    fi
    for rate in "${rate_values[@]}"; do
      if [[ "$article_grid" == true && "$log_size" -ge 24 && "$rate" -gt 1 ]]; then
        continue
      fi

      for phase in "${phase_values[@]}"; do
        case_args=(
          --output "$out"
          --min-log-size "$log_size"
          --max-log-size "$log_size"
          --folds "$fold"
          --rates "$rate"
          --phases "$phase"
        )

        "$cpu_bin" "${case_args[@]}" "${common_args[@]}"
        "$gpu_bin" "${case_args[@]}" "${common_args[@]}"
      done
    done
  done
done

target/release/phase_benchmark_report "$out" > "${out%.jsonl}.csv"

printf 'wrote %s\n' "$out"
printf 'wrote %s\n' "${out%.jsonl}.csv"
