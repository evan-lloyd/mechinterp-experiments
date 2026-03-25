# transformers_sae

Code and experiments scaling up [replacement-aware SAE training](https://elloworld.net/posts/replacement-aware-sae-training/) to more realistic models. Currently supports Gemma-2-2B, but should work on any `transformers` model, with the caveat that most models will require a custom wrapper. Documentation is TODO, but see GemmaReplacement for an example. I will write additional documentation here once I am finished with the accompanying blog post. I also plan to release the trained SAEs and validation data backing the plots in the plotting notebooks. See some preliminary results in the notebooks under [./notebooks](./notebooks), which show a massive reduction in KL divergence between the full replacement model and the base model using my training methods.

