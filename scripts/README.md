# Processing and experiment scripts

The numbered scripts implement the canonical Guwahati data pipeline. Read
[`instructions.md`](../instructions.md) before running them.

## Guwahati experiment helpers

`10_prepare_guwahati_vertical_visibility_mlready.py` prepares a standalone
vertical-visibility experiment dataset from
`08_unified_dataset/guwahati_unified_lstm_dataset_v3_physics_corrected.csv`.
Generated output is written to
`07_lstm_ready/vertical_visibility_experiment/`.

This helper is separate from the numbered Stage 4 pipeline, which prepares the
canonical daily LSTM-ready files.

`07a_prepare_guwahati_target_filtered_mlready.py` is a separate Stage 4
comparison branch after Stage 6. It drops rows missing `VIS_Visibility_MOR`,
removes long wind-gap intervals, and writes to
`07_lstm_ready/target_filtered_experiment/`. It does not replace
`07_stage4_mlreadydata.py`.

## Vertical-visibility filling experiments

- `11_impute_cloud_vertical_visibility.py` trains a gradient-boosting model from
  measured vertical-visibility rows and writes model estimates to a separate
  column in a new unified dataset.
- `12_rule_based_fill_cloud_visibility.py` preserves measured values, then fills
  missing values using conditional medians and deterministic weather rules.
  It writes fill method, reason, and confidence columns.

Both helpers default to the physics-corrected v3 dataset in
`08_unified_dataset/`; their outputs are separate generated CSV files in the
same ignored dataset directory. They are alternative approaches; run either
one independently, not as sequential stages. Imputed estimates are not
ground-truth measurements and must not be used as test labels.

Model benchmarking is performed in
[`../notebooks/benchmark_guwahati_models.ipynb`](../notebooks/benchmark_guwahati_models.ipynb).
