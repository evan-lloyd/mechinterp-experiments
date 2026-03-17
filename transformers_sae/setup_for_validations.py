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

from huggingface_hub import download_bucket_files, list_bucket_tree

# Reuse checkpoint filename parsing from ops so we stay in sync with save format
from transformers_sae.ops import _parse_checkpoint_filename


def _normalize_bucket_id(remote: str) -> str:
    """Convert HF_BUCKET_REMOTE to bucket_id (namespace/bucket_name)."""
    s = remote.strip()
    if s.startswith("hf://buckets/"):
        s = s[len("hf://buckets/") :].strip("/")
    return s


def _discover_latest_checkpoints(bucket_id: str):
    """
    List bucket recursively, parse checkpoint paths, and return for each
    (model_name, training_method, layer) the single checkpoint path with the most tokens.

    Yields (remote_path, model_name, training_method, layer, tokens).
    """
    # (model_name, training_method, layer) -> (tokens, remote_path)
    best: dict[tuple[str, str, int], tuple[int, str]] = {}

    for item in list_bucket_tree(bucket_id, recursive=True):
        if getattr(item, "type", None) == "directory":
            continue
        path = getattr(item, "path", None) or str(item)
        if not path.endswith(".checkpoint"):
            continue
        parts = path.split("/")
        if len(parts) < 3:
            continue
        filename = parts[-1]
        training_method = parts[-2]
        model_name = parts[-3]
        parsed = _parse_checkpoint_filename(filename)
        if parsed is None:
            continue
        layer, tokens = parsed
        key = (model_name, training_method, layer)
        if key not in best or tokens > best[key][0]:
            best[key] = (tokens, path)

    for (model_name, training_method, layer), (tokens, remote_path) in best.items():
        yield remote_path, model_name, training_method, layer, tokens


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
    downloads = list(_discover_latest_checkpoints(bucket_id))
    if not downloads:
        print("No checkpoints found in bucket.", file=sys.stderr)
        return 0

    # (remote_path, local_path) for download_bucket_files
    files_to_download = []
    for remote_path, model_name, training_method, layer, tokens in downloads:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(local_root, model_name, training_method, filename)
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
