# transformers_sae

## Overview

Code and experiments scaling up [replacement-aware SAE training](https://elloworld.net/posts/replacement-aware-sae-training/) to more realistic models. Currently supports Gemma-2-2B, but should work on any `transformers` model, with the caveat that most models will require a custom wrapper. Documentation is TODO, but see GemmaReplacement for an example. See the write-up [here](https://elloworld.net/posts/you-dont-need-error-nodes-you-need-better-features).

## Data

All data associated with the blog post are available in a [huggingface bucket](https://huggingface.co/buckets/evan-lloyd/replacement-aware-saes). The directory structure is as follows, and should be largely self-explanatory (entries in square brackets are not present in all subfolders):

```
figures_and_tables/
gemma_2_2b/
├── benchmarks/
├── sae_checkpoints/
│   ├── {training_method}/
│   │   ├── layer_{x}_tokens_{y}.checkpoint
│   │   ├── [tuned_thresholds_{x}]
│   │   └── [train_thresholds_{x}]
├── validations/
│   ├── {training_method}/
│   │   ├── 0.validation.cloudpickle
│   │   └── [single_layer_rre.validation.cloudpickle]
└── benchmarks/
    └── {training_method}_{arc-e|cqa|mmlu}_{0|25}.parquet
```

Naming conventions: training_methods are somewhat arbitrarily named, depending on how I was thinking about the variant I was trying at the time it was run, but are consistent across sae_checkpoints/validations/benchmarks. "standard" is self-explanatory, "next_layer" refers to replacement-aware SAEs. See [Table 3](https://elloworld.net/posts/you-dont-need-error-nodes-you-need-better-features/#tbl-sae-suite) from the blog post (particularly the links) for a more comprehensive mapping.

The .checkpoint files are zip files containing multiple safetensors files for the SAE weights at the given layer and number of training tokens, as well as training metrics and SAE configuration objects. `cloudpickle` and `safetensors` are required to open them; see `ops.py#load_checkpoint`. `tuned_thresholds` and `train_thresholds` are cloudpickles of updated BatchTopK thresholds (dict of tuples), independent of any updated weights. This means that they have to applied separately after loading the checkpoint (this is handled by `ops.py`, but something to keep in mind if writing your own deserialization).

Validations are simply a cloudpickled `LayerEval` (validation.py). The number represents the start layer of the replacement model (in retrospect this should have been included in their metadata), or "single_layer_rre" (for runs of "standard" SAE metrics).

Benchmarks are `pandas` dataframes serialized as parquet, for the corresponding method, benchmark, and replacement model start layer. (I only use layer 0 in the blog post, but a comparison I didn't end up running could use the final start layer as a rough comparison to the performance that could be achieved by a replacement model with error nodes).

## Usage

This project uses the `uv` package manager. To set up the virtual environment, you'll want to pass the "extra" corresponding to whether you want a version of torch with "cuda", "xpu", or "rocm" (if not specified, you'll get CPU/MPS only). eg: `uv sync --extra cuda`. I recommend activating the environment in shell rather than using `uv run`, since the latter will overwrite the venv unless you pass `--extra` every time.

The recommended way to run these SAEs is to load a full suite of them via ops.py#load_saes, which takes in a directory, iterable of layers to load (generally, you'll want to use range(num_layers)), and optional token count of the desired checkpoint (defaulting to latest checkpoint). The return value is a dict of (int, SAE), mapping layers to their corresponding SAE. To construct a replacement model from these, use replacement_model.py#make_replacement_model. This is a fully-functional torch.nn.Module, but for any analysis not supported by existing scripts I'd recommend using tokenization.py#make_dataloader and activation_data.py#make_activation_batch, since this will be much more efficient (eg, automatically supports early stopping). For kv cache-accelerated rollouts, use ops.py#generate, as this will ensure that appropriate arguments are added to the SAEs--results may be incorrect or this may raise if using the built-in `.generate()`.

See [notebooks/example_usage.ipynb](./notebooks/example_usage.ipynb) for a simple example.

## Reproducing results

Some utility scripts are included under the scripts directory. There is an incomplete "frontend" to them offering some basic features like tab-complete that can be accessed by sourcing `./run.sh` in shell, which will also activate the project venv. Scripts can then be invoked with `./run.sh name_of_script [script_args]`. These scripts assume the existence of an environment variable `HF_BUCKET_LOCAL` that points to a directory to which SAE checkpoints have been generated / downloaded from the bucket. NB: the `bucket.sh` helper script assumes a bucket layout that applied to the private bucket I used during development, which is slightly different from the public bucket, so you may have to move files to use this. (I may follow up by cleaning this up).

I don't have a unified script for reproducing my main training runs, and I'm not sure how useful one would be (it would take at least several days on a 5090 to reproduce every run used in the post). See `scripts/train_gemma_*.py` for some examples of how these SAEs were trained. For example, you could replicate the "standard training, LISTA encoder" training run with `./run.sh train_gemma_standard_lista`.

KL fine-tuning can be applied to a replacement model with `./run.sh fine_tune_gemma -m {main_phase_prefix} -f {finetuned|next_layer_finetuned}`. eg: `./run.sh fine_tune_gemma -m standard -f finetuned`. This assumes that the main phase checkpoints already exist in the $HF_BUCKET_LOCAL directory. `fine_tune_gemma_cqa` is similar, while interleaving in questions/answers from CommonSenseQA, as was used for some of the benchmark results in the blog post.

Encoder tuning is applied with `./run.sh tune_encoders -m {checkpoint_prefix}`.

Validation data are generated with `./run.sh validate_gemma -m {checkpoint_prefix}`, and benchmarks with `./run.sh benchmark_gemma -m {checkpoint_prefix}`.

Figures and tables used in the blog post were all generated using [notebooks/gemma_plots.ipynb](./notebooks/gemma_plots.ipynb) and [qualitative_results.ipynb](./notebooks/qualitative_results.ipynb).
