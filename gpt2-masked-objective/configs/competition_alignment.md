# Competition Alignment (Shared With Teammate)

Use these values for both `run_mask5.json` and `run_mask10.json`:

- `starting_checkpoint`: same absolute path for both runs
- `tokenizer_path`: same absolute path for both runs
- `training_data_path`: same dataset path and version
- `max_sequence_length`: `256`
- `batch_size_per_device`: `4`
- `gradient_accumulation_steps`: `16`
- `optimizer`: `AdamW`
- `learning_rate`: `3e-4`
- `weight_decay`: `0.1`
- `scheduler`: `cosine`
- `warmup_ratio`: `0.06`
- `gradient_clipping`: `1.0`
- `seed`: `42`
- `num_train_epochs`: `10`
- `max_words_seen`: `1000000000`
- `checkpoint_words`: `[100M, 200M, 400M, 600M, 800M, 1000M]`
- `final_checkpoint_selection_rule`: `last_checkpoint`

Only keep these different:

- `masked_alpha`: `0.05` vs `0.10`
- `masked_ratio_sampling`: `0.05` vs `0.10`
