# Run Log Template

## Identity

- model_name:
- run_owner:
- run_date:
- machine_info:

## Source

- starting_checkpoint:
- tokenizer_path:
- training_data_path:
- data_filtering_version:

## Objective

- masked_alpha:
- masked_ratio_sampling:
- masked_loss_positions_only:
- mask replacement: 80% mask / 10% random / 10% keep

## Training Hyperparameters

- max_sequence_length:
- batch_size_per_device:
- gradient_accumulation_steps:
- optimizer:
- learning_rate:
- scheduler:
- warmup_ratio:
- weight_decay:
- gradient_clipping:
- mixed_precision:
- seed:

## Budget and Checkpoints

- training_budget_mode (steps or tokens):
- total_training_steps:
- total_training_tokens:
- checkpoint_save_every_steps:
- checkpoint_selection_rule:
- final_selected_checkpoint:

## Training Notes

- startup checks:
- anomalies:
- loss trend:
- crash/restart history:

## Evaluation Notes

- evaluation_script_version:
- evaluation_data_version:
- scoring_method:
- final_status:
