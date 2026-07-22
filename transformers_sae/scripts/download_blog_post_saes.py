"""
Synchronize the SAE checkpoints used in the blog post from a Hugging Face bucket
to a local directory.

Expects:
  HF_BUCKET_REMOTE: bucket root (e.g. username/my-bucket or hf://buckets/username/my-bucket)
  HF_BUCKET_LOCAL:  local directory root to sync into

Remote layout: {HF_BUCKET_REMOTE}/gemma_2_2b/{method}/...
Local  layout: {HF_BUCKET_LOCAL}/gemma_2_2b/sae_checkpoints/{method}/...

(The remote and local subdirectory structures differ intentionally; the local
side is being refactored to nest checkpoints under a `sae_checkpoints` folder.)

Only checkpoints whose method sub-folder name is in ALL_BUCKET_METHODS are synced,
and each method sub-folder is synced individually.
"""

import os
import sys

from huggingface_hub import sync_bucket

from scripts.methods_for_blog_post import ALL_BUCKET_METHODS


def _normalize_bucket_url(remote: str) -> str:
    """Convert HF_BUCKET_REMOTE to an hf://buckets/... URL."""
    s = remote.strip().rstrip("/")
    if s.startswith("hf://buckets/"):
        return s
    return f"hf://buckets/{s}"


def main() -> int:
    remote = os.environ.get("HF_BUCKET_REMOTE")
    local_root = os.environ.get("HF_BUCKET_LOCAL")
    if not remote or not local_root:
        print(
            "Set HF_BUCKET_REMOTE and HF_BUCKET_LOCAL to the bucket root and local sync path.",
            file=sys.stderr,
        )
        return 1

    bucket_url = _normalize_bucket_url(remote)

    failures = []
    for method in ALL_BUCKET_METHODS:
        source = f"{bucket_url}/gemma_2_2b/{method}"
        dest = os.path.join(local_root, "gemma_2_2b", "sae_checkpoints", method)
        os.makedirs(dest, exist_ok=True)

        print(f"Syncing {source} -> {dest}")
        try:
            plan = sync_bucket(source, dest, verbose=True)
            print(f"  {method}: {plan.summary()}")
        except Exception as e:  # keep going if a method folder is missing/errors
            print(f"  {method}: FAILED ({e})", file=sys.stderr)
            failures.append(method)

    if failures:
        print(f"Finished with {len(failures)} failed method(s): {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
