#!/bin/bash

# Script runner that executes Python scripts with uv or shell scripts directly
# Usage: ./run.sh <script_name> [args...]
# Tab completion: source this file or add to .bashrc

SCRIPTS_DIR="scripts"

_run_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local scripts=""

    # Past the script name, delegate `run.sh bucket ...` to bucket.sh's
    # own completion (sourced from scripts/bucket-completion.sh below)
    if [[ $COMP_CWORD -ge 2 ]]; then
        if [[ "${COMP_WORDS[1]}" == "bucket" ]] && command -v _bucket_sh >/dev/null; then
            COMP_WORDS=("${COMP_WORDS[@]:1}")
            COMP_CWORD=$((COMP_CWORD - 1))
            _bucket_sh
        fi
        return
    fi

    if [[ -d "$SCRIPTS_DIR" ]]; then
        for f in "$SCRIPTS_DIR"/*.py "$SCRIPTS_DIR"/*.sh; do
            if [[ -f "$f" && "$f" != *-completion.sh ]]; then
                local basename=$(basename "$f")
                local name="${basename%.*}"
                scripts="$scripts $name"
            fi
        done
    fi

    COMPREPLY=($(compgen -W "$scripts" -- "$cur"))
}

# Register completion if being sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    complete -F _run_completions run.sh
    complete -F _run_completions ./run.sh
    source "$(dirname "${BASH_SOURCE[0]:-.}")/scripts/bucket-completion.sh"
    source .venv/bin/activate
    return 0
fi

# Main execution
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script_name> [args...]"
    echo "Available scripts:"
    if [[ -d "$SCRIPTS_DIR" ]]; then
        for f in "$SCRIPTS_DIR"/*.py "$SCRIPTS_DIR"/*.sh; do
            if [[ -f "$f" && "$f" != *-completion.sh ]]; then
                echo "  $(basename "${f%.*}")"
            fi
        done
    fi
    exit 1
fi

SCRIPT_NAME="$1"
shift

USE_PDB=0

if [[ "$1" == "--pdb" ]]; then
    USE_PDB=1
    shift
fi

# Check for Python script
if [[ -f "$SCRIPTS_DIR/$SCRIPT_NAME.py" ]]; then
    if [[ $USE_PDB -eq 1 ]]; then
        exec python -m "scripts.debug" $SCRIPT_NAME.py "$@"
    else
        exec python -m "scripts.$SCRIPT_NAME" "$@"
    fi
# Check for shell script
elif [[ -f "$SCRIPTS_DIR/$SCRIPT_NAME.sh" ]]; then
    exec bash "$SCRIPTS_DIR/$SCRIPT_NAME.sh" "$@"
else
    echo "Error: Script '$SCRIPT_NAME' not found in $SCRIPTS_DIR"
    echo "Looking for: $SCRIPTS_DIR/$SCRIPT_NAME.py or $SCRIPTS_DIR/$SCRIPT_NAME.sh"
    exit 1
fi
