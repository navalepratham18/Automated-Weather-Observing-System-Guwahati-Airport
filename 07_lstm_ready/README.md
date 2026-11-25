# LSTM-Ready Data

This folder contains daily files formatted for machine-learning workflows.
The files contain the selected features, target columns, consistent timestamps,
and data layout expected by the training notebooks.

Preparation is performed by `scripts/07_stage4_mlreadydata.py`.

Optional comparison workflows write to separate subfolders here, including
`vertical_visibility_experiment/` and `target_filtered_experiment/`. These
outputs are not replacements for the canonical daily Stage 4 data.
