# Unified Datasets

This folder contains chronological datasets created by combining the daily
LSTM-ready files into a single training and evaluation input.

Merging and physics-based cloud corrections are handled by
`scripts/08_merge_lstm_ready_dataset.py` and
`scripts/09_physics_cloud_correction.py`. Large datasets are ignored by Git.
