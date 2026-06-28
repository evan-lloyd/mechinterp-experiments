#!/bin/bash
hf buckets sync $HF_BUCKET_REMOTE/validations $HF_BUCKET_LOCAL/validations
hf buckets sync $HF_BUCKET_REMOTE/benchmarks $HF_BUCKET_LOCAL/benchmarks
