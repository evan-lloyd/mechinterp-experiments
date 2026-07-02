#!/usr/bin/env bash

set -euo pipefail

DEFAULT_MODEL_NAME="gemma_2_2b"

usage() {
  cat <<'EOF'
Usage:
  bucket.sh <upload|download> <checkpoint|validation|benchmark> -n <method_name> [-m <model_name>] [-b <benchmark>] [--all]

Arguments:
  command                upload or download
  data_type              checkpoint, validation, or benchmark

Options:
  -n <method_name>       Required. Training method name.
  -m <model_name>        Optional. Defaults to gemma_2_2b.
  -b <benchmark>         For benchmarks only. Specific benchmark name (e.g., mmlu, arc-e, etc.).
  --all                  For benchmarks only. Sync all benchmarks for this method/model.
  -h                     Show this help message.

Examples:
  bucket.sh upload checkpoint -n standard
  bucket.sh upload validation -n standard -m gemma_2_2b
  bucket.sh download checkpoint -n sparse_autoencoder
  bucket.sh upload benchmark -n standard -m gemma_2_2b -b mmlu
  bucket.sh download benchmark -n mymethod -m gemma_2_2b --all

Tab completion:
  source bucket-completion.sh (zsh: run `autoload -U +X bashcompinit && bashcompinit` first)
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

command_name="$1"
data_type="$2"
shift 2

model_name="$DEFAULT_MODEL_NAME"
method_name=""
benchmark_name=""
sync_all=0

# handle --all before getopts, so getopts doesn't get confused
# Manually filter out --all from the arguments array, since Bash arrays and parameter substitution can be tricky.
new_args=()
for arg in "$@"; do
  if [[ "$arg" == "--all" ]]; then
    sync_all=1
  else
    new_args+=("$arg")
  fi
done
set -- "${new_args[@]}"

while getopts ":m:n:b:h" opt; do
  case "$opt" in
    m) model_name="$OPTARG" ;;
    n) method_name="$OPTARG" ;;
    b) benchmark_name="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Error: option -$OPTARG requires an argument." >&2
      usage
      exit 1
      ;;
    \?)
      echo "Error: invalid option -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${HF_BUCKET_REMOTE:-}" ]]; then
  echo "Error: HF_BUCKET_REMOTE is not set." >&2
  exit 1
fi

if [[ -z "$method_name" ]]; then
  echo "Error: method_name is required (use -n <method_name>)." >&2
  usage
  exit 1
fi

case "$data_type" in
  checkpoint)
    local_dir="${HF_BUCKET_LOCAL}/${model_name}/${method_name}"
    remote_dir="${HF_BUCKET_REMOTE}/${model_name}/${method_name}"
    cmd="sync"
    extra_args=()
    ;;
  validation)
    local_dir="${HF_BUCKET_LOCAL}/validations/${model_name}/${method_name}"
    remote_dir="${HF_BUCKET_REMOTE}/validations/${model_name}/${method_name}"
    cmd="sync"
    extra_args=()
    ;;
  benchmark)
    # Flat layout: {BUCKET}/{model}/benchmarks/{method_name}_{benchmark_name}.parquet
    local_dir="${HF_BUCKET_LOCAL}/benchmarks/${model_name}"
    remote_dir="${HF_BUCKET_REMOTE}/benchmarks/${model_name}"
    cmd="sync"
    # Determine include glob
    if [[ $sync_all -eq 1 ]]; then
      include_pattern="${method_name}_*.parquet"
    else
      if [[ -z "$benchmark_name" ]]; then
        echo "Error: must specify either --all or -b <benchmark> for benchmarks." >&2
        usage
        exit 1
      fi
      include_pattern="${method_name}_${benchmark_name}.parquet"
    fi
    extra_args=(--include "$include_pattern")
    ;;
  *)
    echo "Error: data_type must be 'checkpoint', 'validation', or 'benchmark'." >&2
    usage
    exit 1
    ;;
esac

case "$command_name" in
  upload)
    from_dir="$local_dir"
    to_dir="$remote_dir"
    ;;
  download)
    from_dir="$remote_dir"
    to_dir="$local_dir"
    ;;
  *)
    echo "Error: command must be 'upload' or 'download'." >&2
    usage
    exit 1
    ;;
esac

echo "Syncing:"
echo "  from: $from_dir"
echo "  to:   $to_dir"
if [[ "$data_type" == "benchmark" ]]; then
  echo "  include: $include_pattern"
fi

hf buckets $cmd "${extra_args[@]}" "$from_dir" "$to_dir"
