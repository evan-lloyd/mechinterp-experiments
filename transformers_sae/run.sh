#!/bin/bash

# Script runner that executes Python scripts with uv or shell scripts directly
# Usage: ./run.sh <script_name> [args...]
# Tab completion: source this file or add to .bashrc

SCRIPTS_DIR="scripts"

_run_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local scripts=""
    
    if [[ -d "$SCRIPTS_DIR" ]]; then
        for f in "$SCRIPTS_DIR"/*.py "$SCRIPTS_DIR"/*.sh; do
            if [[ -f "$f" ]]; then
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
    return 0
fi

# Main execution
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script_name> [args...]"
    echo "Available scripts:"
    if [[ -d "$SCRIPTS_DIR" ]]; then
        for f in "$SCRIPTS_DIR"/*.py "$SCRIPTS_DIR"/*.sh; do
            if [[ -f "$f" ]]; then
                echo "  $(basename "${f%.*}")"
            fi
        done
    fi
    exit 1
fi

SCRIPT_NAME="$1"
shift

# Check for Python script
if [[ -f "$SCRIPTS_DIR/$SCRIPT_NAME.py" ]]; then
    exec uv run -m "scripts.$SCRIPT_NAME" "$@"
# Check for shell script
elif [[ -f "$SCRIPTS_DIR/$SCRIPT_NAME.sh" ]]; then
    exec bash "$SCRIPTS_DIR/$SCRIPT_NAME.sh" "$@"
else
    echo "Error: Script '$SCRIPT_NAME' not found in $SCRIPTS_DIR"
    echo "Looking for: $SCRIPTS_DIR/$SCRIPT_NAME.py or $SCRIPTS_DIR/$SCRIPT_NAME.sh"
    exit 1
fi
