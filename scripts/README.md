# City-specific scripts

Copy the applicable reference scripts from the repository root into this
directory, rename them with the numbered names described in
[`instructions.md`](../../instructions.md), and update their input/output
paths and explicit column lists. Keep the original reference scripts unchanged
so a future comparison remains possible.

## Guwahati experiment helpers

`prepare_guwahati_vertical_visibility_mlready.py` prepares a standalone
vertical-visibility experiment dataset from
`08_unified_dataset/guwahati_unified_lstm_dataset_v3_physics_corrected.csv`.
Generated output is written to
`07_lstm_ready/vertical_visibility_experiment/`.

This helper is separate from the numbered Stage 4 pipeline, which prepares the
canonical daily LSTM-ready files.

## Model benchmarking

`benchmark_guwahati_models.py` benchmarks classical ML and deep sequence models
with the same chronological split, lookback, target masking, and measured-label
evaluation. Generated summaries and checkpoints are written to
`10_models/guwahati_model_benchmark/`.
