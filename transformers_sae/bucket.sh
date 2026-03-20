#!/usr/bin/env bash

set -euo pipefail

LOCAL_ROOT="/workspace/sae_checkpoints"
DEFAULT_MODEL_NAME="gemma_2_2b"

usage() {
  cat <<'EOF'
Usage:
  bucket.sh <upload|download> <checkpoint|validation> -n <method_name> [-m <model_name>]

Arguments:
  command                upload or download
  data_type              checkpoint or validation

Options:
  -n <method_name>       Required. Training method name.
  -m <model_name>        Optional. Defaults to gemma_2_2b.
  -h                     Show this help message.

Examples:
  bucket.sh upload checkpoint -n standard
  bucket.sh upload validation -n standard -m gemma_2_2b
  bucket.sh download checkpoint -n sparse_autoencoder
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

while getopts ":m:n:h" opt; do
  case "$opt" in
    m) model_name="$OPTARG" ;;
    n) method_name="$OPTARG" ;;
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
    local_dir="${LOCAL_ROOT}/${model_name}/${method_name}"
    remote_dir="${HF_BUCKET_REMOTE}/${model_name}/${method_name}"
    ;;
  validation)
    local_dir="${LOCAL_ROOT}/validations/${model_name}/${method_name}"
    remote_dir="${HF_BUCKET_REMOTE}/validations/${model_name}/${method_name}"
    ;;
  *)
    echo "Error: data_type must be 'checkpoint' or 'validation'." >&2
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

uv run hf buckets sync "$from_dir" "$to_dir"
