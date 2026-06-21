#!/bin/bash
uv run hf buckets sync $HF_BUCKET_REMOTE/validations $HF_BUCKET_LOCAL/validations
uv run hf buckets sync $HF_BUCKET_REMOTE/benchmarks $HF_BUCKET_LOCAL/benchmarks
