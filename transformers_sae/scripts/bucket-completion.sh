#!/usr/bin/env bash
# Tab completion for bucket.sh.
#
# bash:  source transformers_sae/scripts/bucket-completion.sh
# zsh:   autoload -U +X bashcompinit && bashcompinit
#        source transformers_sae/scripts/bucket-completion.sh
#
# Completes commands, data types, and option flags, plus values for
# -m / -n / -b by listing what already exists:
#   upload   -> local files under $HF_BUCKET_LOCAL
#   download -> remote files via `hf buckets ls $HF_BUCKET_REMOTE`

_bucket_sh_default_model="gemma_2_2b"

# Cache the last remote listing so repeated TABs don't re-hit the network.
_bucket_sh_ls_remote() {
  # NB: don't name this local "path" -- in zsh that's tied to PATH
  local remote_path="$1" now
  now=$(date +%s)
  if [[ "${_bucket_sh_cache_path:-}" == "$remote_path" && $((now - ${_bucket_sh_cache_time:-0})) -lt 60 ]]; then
    printf '%s\n' "$_bucket_sh_cache_out"
    return
  fi
  _bucket_sh_cache_out=$(hf buckets ls "$remote_path" 2>/dev/null) || _bucket_sh_cache_out=""
  _bucket_sh_cache_path="$remote_path"
  _bucket_sh_cache_time="$now"
  printf '%s\n' "$_bucket_sh_cache_out"
}

# List entry basenames at a subpath of the bucket.
# $1: upload|download   $2: subpath relative to the bucket root ("" for root)
_bucket_sh_list() {
  local cmd="$1" sub="$2" root line name
  if [[ "$cmd" == "upload" ]]; then
    [[ -n "${HF_BUCKET_LOCAL:-}" ]] || return 0
    root="${HF_BUCKET_LOCAL%/}${sub:+/$sub}"
    [[ -d "$root" ]] || return 0
    ls -1 "$root" 2>/dev/null
  else
    [[ -n "${HF_BUCKET_REMOTE:-}" ]] || return 0
    root="${HF_BUCKET_REMOTE%/}${sub:+/$sub}"
    _bucket_sh_ls_remote "$root" | while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      name="${line##* }"   # last column, in case of size/date columns
      name="${name%/}"
      name="${name##*/}"
      [[ -n "$name" ]] && printf '%s\n' "$name"
    done
  fi
}

# Model names for a data type.
_bucket_sh_models() {
  local cmd="$1" dtype="$2"
  case "$dtype" in
    checkpoint) _bucket_sh_list "$cmd" "" | grep -vx -e validations -e benchmarks ;;
    validation) _bucket_sh_list "$cmd" "validations" ;;
    benchmark)  _bucket_sh_list "$cmd" "benchmarks" ;;
  esac
}

# Method names for a data type + model.
_bucket_sh_methods() {
  local cmd="$1" dtype="$2" model="$3" f
  case "$dtype" in
    checkpoint) _bucket_sh_list "$cmd" "$model" ;;
    validation) _bucket_sh_list "$cmd" "validations/$model" ;;
    benchmark)
      # Files are {method}_{benchmark}.parquet; benchmark has no underscores.
      _bucket_sh_list "$cmd" "benchmarks/$model" | while IFS= read -r f; do
        [[ "$f" == *.parquet ]] || continue
        f="${f%.parquet}"
        [[ "$f" == *_* ]] && printf '%s\n' "${f%_*}"
      done | sort -u
      ;;
  esac
}

# Benchmark names for a model, optionally filtered by method.
_bucket_sh_benchmarks() {
  local cmd="$1" model="$2" method="$3" f
  _bucket_sh_list "$cmd" "benchmarks/$model" | while IFS= read -r f; do
    [[ "$f" == *.parquet ]] || continue
    f="${f%.parquet}"
    if [[ -n "$method" ]]; then
      [[ "$f" == "${method}_"* ]] && printf '%s\n' "${f#"${method}_"}"
    else
      [[ "$f" == *_* ]] && printf '%s\n' "${f##*_}"
    fi
  done | sort -u
}

_bucket_sh() {
  local cur prev cmd dtype opts i
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  cmd="${COMP_WORDS[1]:-}"
  dtype="${COMP_WORDS[2]:-}"

  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=($(compgen -W "upload download" -- "$cur"))
    return
  fi
  if [[ $COMP_CWORD -eq 2 ]]; then
    COMPREPLY=($(compgen -W "checkpoint validation benchmark" -- "$cur"))
    return
  fi

  # Pick up any -m / -n already on the line.
  local model="$_bucket_sh_default_model" method=""
  for ((i = 3; i < COMP_CWORD; i++)); do
    case "${COMP_WORDS[i]}" in
      -m) model="${COMP_WORDS[i+1]:-$model}" ;;
      -n) method="${COMP_WORDS[i+1]:-}" ;;
    esac
  done

  case "$prev" in
    -m)
      COMPREPLY=($(compgen -W "$(_bucket_sh_models "$cmd" "$dtype")" -- "$cur"))
      ;;
    -n)
      COMPREPLY=($(compgen -W "$(_bucket_sh_methods "$cmd" "$dtype" "$model")" -- "$cur"))
      ;;
    -b)
      COMPREPLY=($(compgen -W "$(_bucket_sh_benchmarks "$cmd" "$model" "$method")" -- "$cur"))
      ;;
    *)
      opts="-n -m -h"
      [[ "$dtype" == "benchmark" ]] && opts="-n -m -b --all -h"
      COMPREPLY=($(compgen -W "$opts" -- "$cur"))
      ;;
  esac
}

complete -F _bucket_sh bucket.sh
