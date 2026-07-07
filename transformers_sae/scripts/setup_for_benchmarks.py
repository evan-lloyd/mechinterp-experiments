"""
Download the latest checkpoint (most tokens trained) per model/training_method/layer
from a Hugging Face bucket to a local directory.

Expects:
  HF_BUCKET_REMOTE: bucket root (e.g. username/my-bucket or hf://buckets/username/my-bucket)
  HF_BUCKET_LOCAL:  local directory to download into

Bucket layout: {model_name}/{training_method}/layer_{layer}_tokens_{n_tokens}.checkpoint
Checkpoint filename format is from transformers_sae.ops (_checkpoint_filename / _parse_checkpoint_filename).
"""

import os
import sys
from itertools import chain

from huggingface_hub import download_bucket_files, list_bucket_tree

from scripts.methods_for_blog_post import TUNED_ENCODER_METHODS

# Reuse checkpoint filename parsing from ops so we stay in sync with save format
from transformers_sae.ops import _parse_checkpoint_filename


def _normalize_bucket_id(remote: str) -> str:
    """Convert HF_BUCKET_REMOTE to bucket_id (namespace/bucket_name)."""
    s = remote.strip()
    if s.startswith("hf://buckets/"):
        s = s[len("hf://buckets/") :].strip("/")
    return s


def _discover_latest_checkpoints_and_extra_files(bucket_id: str, prefix: str):
    """
    List bucket recursively. For TUNED_ENCODER_METHODS, yield highest-token checkpoint per model/layer/method.
    Also, yield any other files (not ending in .checkpoint) under the same model/method directory.
    Yields (remote_path, local_path_rel, is_checkpoint, extra_info)
    """
    # Use TUNED_ENCODER_METHODS only for checkpoint selection
    checkpoints: dict[tuple[str, str, int], list[tuple[int, str, str]]] = {}

    # Store non-checkpoint files: (remote_path, model_name, training_method, filename)
    extra_files: list[tuple[str, str, str, str]] = []

    for item in list_bucket_tree(bucket_id, prefix=prefix, recursive=True):
        if getattr(item, "type", None) == "directory":
            continue
        path = getattr(item, "path", None) or str(item)
        parts = path.split("/")
        if len(parts) < 3:
            continue
        filename = parts[-1]
        training_method = parts[-2]
        model_name = parts[-3]

        if filename.endswith(".checkpoint"):
            parsed = _parse_checkpoint_filename(filename)
            if parsed is None:
                continue
            layer, tokens = parsed
            key = (model_name, training_method, layer)
            checkpoints.setdefault(key, []).append((tokens, path, filename))
        else:
            # Copy all other files under the same model/method (ignore subfolders)
            extra_files.append((path, model_name, training_method, filename))

    # Yield highest-token checkpoint per (model, method, layer)
    yielded_ckpt = set()
    for (model_name, training_method, layer), token_path_list in checkpoints.items():
        top = sorted(token_path_list, key=lambda x: -x[0])[:1]
        for tokens, remote_path, filename in top:
            yield (
                remote_path,
                os.path.join(model_name, training_method, filename),
                True,
                (model_name, training_method, layer, tokens),
            )
            yielded_ckpt.add((model_name, training_method, filename))

    # Yield extra files (not a checkpoint)
    for path, model_name, training_method, filename in extra_files:
        # Avoid duplicate copies (could match both .checkpoint and not)
        if (model_name, training_method, filename) in yielded_ckpt:
            continue
        yield path, os.path.join(model_name, training_method, filename), False, None


def main() -> int:
    remote = os.environ.get("HF_BUCKET_REMOTE")
    local_root = os.environ.get("HF_BUCKET_LOCAL")
    if not remote or not local_root:
        print(
            "Set HF_BUCKET_REMOTE and HF_BUCKET_LOCAL to the bucket root and local download path.",
            file=sys.stderr,
        )
        return 1

    bucket_id = _normalize_bucket_id(remote)
    downloads = [
        f
        for prefix in TUNED_ENCODER_METHODS
        for f in _discover_latest_checkpoints_and_extra_files(
            bucket_id, f"gemma_2_2b/{prefix}"
        )
    ]

    if not downloads:
        print("No files found in bucket.", file=sys.stderr)
        return 0

    # (remote_path, local_path) for download_bucket_files
    files_to_download = []
    for remote_path, local_path_rel, is_checkpoint, extra_info in downloads:
        local_path = os.path.join(local_root, local_path_rel)
        if not os.path.exists(local_path):
            files_to_download.append((remote_path, local_path))

    for _, local_path in files_to_download:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    download_bucket_files(bucket_id, files=files_to_download)
    for remote_path, local_path in files_to_download:
        print(f"Downloaded {remote_path} -> {local_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
