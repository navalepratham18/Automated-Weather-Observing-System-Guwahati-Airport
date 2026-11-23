# Notebooks

This folder contains exploratory analysis, data-distribution, training, and
evaluation notebooks for the Guwahati weather and visibility models.

Notebooks should use the configured pipeline paths and document important
assumptions, target columns, and evaluation results.

## Training notebooks

- `train_guwahati_lstm_with_physics_correction_V2.ipynb` is the primary
  physics-corrected workflow. It predicts `CLOUD_Cloud_Base_1` and
  `VIS_Visibility_MOR`.
- `train_guwahati_lstm_vertical_visibility_experiment.ipynb` is a separate
  experiment that predicts `CLOUD_Vertical_Visibility` and
  `VIS_Visibility_MOR`. Its generated data and artifacts are isolated under
  `07_lstm_ready/vertical_visibility_experiment/` and
  `10_models/guwahati_vertical_visibility_experiment/`.
