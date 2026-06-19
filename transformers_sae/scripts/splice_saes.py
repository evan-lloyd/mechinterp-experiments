"""
Splice the latest SAE checkpoints from two training methods into a new method folder.

Given two training methods (subfolders under $HF_BUCKET_LOCAL/{model_name}) and a
splice layer L, this copies the latest checkpoint (most tokens trained) for each layer
into a new sibling folder:

  * layers <= L are taken from the "lower" method
  * layers  > L are taken from the "upper" method

The result is itself a normal method folder under $HF_BUCKET_LOCAL/{model_name}, so it
can be downloaded / validated / benchmarked like any other.

Expects:
  HF_BUCKET_LOCAL: local bucket root (see setup_for_validations.py)

Layout: {model_name}/{training_method}/layer_{layer}_tokens_{n_tokens}.checkpoint
Threshold (and any other non-checkpoint) files are ignored; only SAE checkpoints are copied.

Example:
  ./run.sh splice_saes gemma_2_2b foo bar 6
    -> creates gemma_2_2b/foo_le6_bar_gt6 with foo's checkpoints for layers <= 6
       and bar's checkpoints for layers > 6.
"""

import argparse
import os
import shutil
import sys
from typing import Dict

# Reuse checkpoint discovery/parsing from ops so we stay in sync with the save format.
from transformers_sae.ops import _parse_checkpoint_filename, find_latest_checkpoint


def _latest_checkpoint_per_layer(method_dir: str) -> Dict[int, str]:
    """Map each layer present in method_dir to its latest checkpoint path (most tokens)."""
    layers = set()
    for filename in os.listdir(method_dir):
        parsed = _parse_checkpoint_filename(filename)
        if parsed is not None:
            layers.add(parsed[0])

    latest: Dict[int, str] = {}
    for layer in layers:
        path = find_latest_checkpoint(method_dir, layer)
        if path is not None:
            latest[layer] = path
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Splice latest SAE checkpoints from two training methods at a given layer."
        )
    )
    parser.add_argument(
        "model_name",
        help="Model subfolder under $HF_BUCKET_LOCAL (e.g. gemma_2_2b).",
    )
    parser.add_argument(
        "lower_method",
        help="Training method whose checkpoints are used for layers <= splice_layer.",
    )
    parser.add_argument(
        "upper_method",
        help="Training method whose checkpoints are used for layers > splice_layer.",
    )
    parser.add_argument(
        "splice_layer",
        type=int,
        help="Boundary layer: <= goes to lower_method, > goes to upper_method.",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help=(
            "Name of the resulting method folder. "
            "Defaults to {lower_method}_le{L}_{upper_method}_gt{L}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing (non-empty) output folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned copies without writing anything.",
    )
    args = parser.parse_args()

    local_root = os.environ.get("HF_BUCKET_LOCAL")
    if not local_root:
        print(
            "Set HF_BUCKET_LOCAL to the local bucket root.",
            file=sys.stderr,
        )
        return 1

    splice_layer = args.splice_layer
    lower_dir = os.path.join(local_root, args.model_name, args.lower_method)
    upper_dir = os.path.join(local_root, args.model_name, args.upper_method)

    for method, method_dir in (
        (args.lower_method, lower_dir),
        (args.upper_method, upper_dir),
    ):
        if not os.path.isdir(method_dir):
            print(f"Method folder not found: {method_dir}", file=sys.stderr)
            return 1

    lower_latest = _latest_checkpoint_per_layer(lower_dir)
    upper_latest = _latest_checkpoint_per_layer(upper_dir)

    # (layer, src_path, source_method) for layers on each side of the splice.
    selected = [
        (layer, path, args.lower_method)
        for layer, path in lower_latest.items()
        if layer <= splice_layer
    ] + [
        (layer, path, args.upper_method)
        for layer, path in upper_latest.items()
        if layer > splice_layer
    ]
    selected.sort(key=lambda item: item[0])

    if not selected:
        print(
            f"No checkpoints to splice for {args.lower_method} (layers <= {splice_layer}) "
            f"or {args.upper_method} (layers > {splice_layer}).",
            file=sys.stderr,
        )
        return 1

    out_name = args.out_name or (
        f"{args.lower_method}_le{splice_layer}_{args.upper_method}_gt{splice_layer}"
    )
    out_dir = os.path.join(local_root, args.model_name, out_name)

    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.overwrite:
        print(
            f"Output folder already exists and is non-empty: {out_dir}\n"
            "Pass --overwrite to write into it anyway.",
            file=sys.stderr,
        )
        return 1

    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    for layer, src_path, source_method in selected:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(out_dir, filename)
        action = "Would copy" if args.dry_run else "Copied"
        if not args.dry_run:
            shutil.copy2(src_path, dst_path)
        print(f"{action} layer {layer:>3} from {source_method}: {filename}")

    print(
        f"\nSpliced {len(selected)} checkpoints into {out_dir}"
        + (" (dry run, nothing written)" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
