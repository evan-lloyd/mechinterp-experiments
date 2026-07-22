# Methods that have been tuned to run as full replacement models
TUNED_ENCODER_METHODS = [
    "gemma_scope_100_l0_tuned_encoder_0",
    "standard_tuned_encoder_0",
    "standard_finetuned_tuned_encoder_0",
    "standard_lista_unit_scale_tuned_encoder_0",
    "standard_finetuned_lista_unit_scale_tuned_encoder_0",
    "next_layer_tuned_encoder_0",
    "next_layer_finetuned_tuned_encoder_0",
    "next_layer_lista_unit_scale_tuned_encoder_0",
    "next_layer_finetuned_lista_unit_scale_tuned_encoder_0",
    "next_layer_in_place_finetuned_lista_unit_scale",
]

CQA_TUNED_METHODS = [
    "standard_finetuned_cqa_tuned_encoder_0",
    "standard_lista_unit_scale_finetuned_cqa_tuned_encoder_0",
    "next_layer_finetuned_cqa_tuned_encoder_0",
    "next_layer_lista_unit_scale_finetuned_cqa_tuned_encoder_0",
]

PRE_TUNED_ENCODER_METHODS = [
    "standard",
    "standard_finetuned",
    "standard_lista_unit_scale",
    "standard_finetuned_lista_unit_scale",
    "next_layer",
    "next_layer_finetuned",
    "next_layer_lista_unit_scale",
    "next_layer_finetuned_lista_unit_scale",
]

# Methods that don't correspond to an actual method you can download from a bucket, but are parsed
# by validation scripts in a particular way (currently: gemma_scope, train_activations).
VALIDATION_ONLY_METHODS = [
    "gemma_scope",
    "gemma_scope_100_l0_train_activations",
    "standard_train_activations",
    "standard_finetuned_train_activations",
    "standard_lista_unit_scale_train_activations",
    "standard_finetuned_lista_unit_scale_train_activations",
    "next_layer_train_activations",
    "next_layer_finetuned_train_activations",
    "next_layer_lista_unit_scale_train_activations",
    "next_layer_finetuned_lista_unit_scale_train_activations",
]

# Used in appendix
MISC_METHODS = [
    "gemma_scope_canonical_l0_tuned_encoder_0",
    "standard_fresh_init",
    "next_layer_5e7",
    "next_layer_5e7_tuned_encoder_0",
]

ALL_BUCKET_METHODS = TUNED_ENCODER_METHODS + PRE_TUNED_ENCODER_METHODS + MISC_METHODS
