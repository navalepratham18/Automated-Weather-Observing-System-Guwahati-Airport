"""Benchmark classical and deep-learning forecasters for Guwahati.

The script is intentionally separate from the canonical Stage 4 pipeline.
It uses a chronological 70/15/15 split and reports metrics only for measured
target values.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42

TARGET_COLUMNS = ["CLOUD_Cloud_Base_1", "VIS_Visibility_MOR"]
TARGET_IMPUTED_FLAGS = {"CLOUD_Cloud_Base_1": "CLOUD_Cloud_Base_1_IMPUTED"}

FEATURE_COLUMNS = [
    "CLOUD_Cloud_Base_1", "CLOUD_Cloud_Base_2", "CLOUD_Cloud_Base_3",
    "CLOUD_OCTA1", "CLOUD_OCTA2", "CLOUD_OCTA3", "CLOUD_Vertical_Visibility",
    "PTU_Pressure_QNH", "PTU_Temperature", "PTU_DewPoint", "PTU_Humidity",
    "PW_Rain_Intensity", "PW_Rain_Sum", "VIS_Visibility_MOR", "VIS_RVR",
    "WIND_Wind_Speed", "WIND_Wind_Gust", "Hour_Sin", "Hour_Cos",
    "CLOUD_SENSOR_ON", "PTU_SENSOR_ON", "PW_SENSOR_ON", "VIS_SENSOR_ON",
    "WIND_SENSOR_ON", "CLOUD_Cloud_Base_1_IMPUTED", "CLOUD_Cloud_Base_2_IMPUTED",
    "CLOUD_Cloud_Base_3_IMPUTED", "CLOUD_Vertical_Visibility_IMPUTED",
    "CLOUD_LONG_GAP", "CLOUD_VERTICAL_LONG_GAP", "WIND_Wind_Dir_Sin", "WIND_Wind_Dir_Cos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Guwahati forecasting models.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "08_unified_dataset" / "guwahati_unified_lstm_dataset_v3_physics_corrected.csv",
    )
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--classic-train-samples", type=int, default=50000)
    parser.add_argument("--classic-test-samples", type=int, default=100000)
    parser.add_argument("--skip-classic", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "10_models" / "guwahati_model_benchmark",
    )
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def prepare_data(input_file: Path, lookback: int, horizon: int):
    df = pd.read_csv(input_file, parse_dates=["Datetime"], dayfirst=True)
    df = df.sort_values("Datetime").drop_duplicates("Datetime")

    missing = sorted(set(FEATURE_COLUMNS + TARGET_COLUMNS) - set(df.columns))
    raw_missing = sorted(set(missing) - {
        "WIND_Wind_Dir_Sin",
        "WIND_Wind_Dir_Cos",
    })
    if "WIND_Wind_Dir" not in df.columns:
        raw_missing.append("WIND_Wind_Dir")
    if raw_missing:
        raise ValueError(f"Missing required columns: {sorted(set(raw_missing))}")

    numeric_columns = sorted(
        set(FEATURE_COLUMNS + TARGET_COLUMNS)
        - {"WIND_Wind_Dir_Sin", "WIND_Wind_Dir_Cos"}
    )
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df.loc[df["WIND_Wind_Speed"] < 0, "WIND_Wind_Speed"] = np.nan
    df.loc[df["WIND_Wind_Gust"] < 0, "WIND_Wind_Gust"] = np.nan
    df["WIND_Wind_Dir_Sin"] = np.sin(np.deg2rad(pd.to_numeric(df["WIND_Wind_Dir"], errors="coerce")))
    df["WIND_Wind_Dir_Cos"] = np.cos(np.deg2rad(pd.to_numeric(df["WIND_Wind_Dir"], errors="coerce")))
    df = df.set_index("Datetime")
    df["ROW_GAP"] = df.index.to_series().diff().dt.total_seconds().gt(15).fillna(False)

    train_rows = int(len(df) * 0.70)
    feature_medians = df.iloc[:train_rows][FEATURE_COLUMNS].median().fillna(0.0)
    feature_scaler = StandardScaler()
    feature_scaler.fit(df.iloc[:train_rows][FEATURE_COLUMNS].fillna(feature_medians))
    prepared = pd.DataFrame(
        feature_scaler.transform(df[FEATURE_COLUMNS].fillna(feature_medians)),
        index=df.index,
        columns=FEATURE_COLUMNS,
    ).fillna(0).astype("float32")

    target_training = df.iloc[:train_rows][TARGET_COLUMNS].copy()
    for target, flag in TARGET_IMPUTED_FLAGS.items():
        target_training.loc[df.iloc[:train_rows][flag].fillna(0).eq(1), target] = np.nan
    target_medians = target_training.median().fillna(0.0)

    target_scaler = StandardScaler()
    target_scaler.fit(target_training.dropna(how="all").fillna(target_medians))

    measured = df[TARGET_COLUMNS].notna().to_numpy(dtype=bool)
    for index, target in enumerate(TARGET_COLUMNS):
        flag = TARGET_IMPUTED_FLAGS.get(target)
        if flag:
            measured[:, index] &= df[flag].fillna(0).eq(0).to_numpy(dtype=bool)

    values = prepared.to_numpy(dtype="float32")
    targets = df[TARGET_COLUMNS].replace([np.inf, -np.inf], np.nan).to_numpy(dtype="float32")
    gaps = df["ROW_GAP"].to_numpy(dtype=bool)

    window_ends, labels, label_masks, timestamps = [], [], [], []
    for end in range(lookback, len(df) - horizon + 1):
        horizon_index = end + horizon - 1
        if gaps[end - lookback + 1:horizon_index + 1].any():
            continue
        if not measured[horizon_index].any():
            continue
        label = targets[horizon_index].copy()
        label_mask = measured[horizon_index].copy()
        label[~label_mask] = target_medians.to_numpy()[~label_mask]
        window_ends.append(end)
        labels.append(label)
        label_masks.append(label_mask.astype("float32"))
        timestamps.append(df.index[horizon_index])

    window_ends = np.asarray(window_ends, dtype="int64")
    labels = np.asarray(labels, dtype="float32")
    label_masks = np.asarray(label_masks, dtype="float32")
    timestamps = pd.DatetimeIndex(timestamps)
    scaled_labels = target_scaler.transform(labels).astype("float32")

    train_cutoff = timestamps[int(len(timestamps) * 0.70)]
    validation_cutoff = timestamps[int(len(timestamps) * 0.85)]
    train_mask = timestamps < train_cutoff
    validation_mask = (timestamps >= train_cutoff) & (timestamps < validation_cutoff)
    test_mask = timestamps >= validation_cutoff

    state = {
        "df": df,
        "values": values,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "target_medians": target_medians,
        "window_ends": window_ends,
        "labels": labels,
        "scaled_labels": scaled_labels,
        "label_masks": label_masks,
        "timestamps": timestamps,
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }
    return state


def flatten_windows(values: np.ndarray, window_ends: np.ndarray, indices: np.ndarray, lookback: int) -> np.ndarray:
    rows = []
    for window_end in window_ends[indices]:
        rows.append(values[window_end - lookback:window_end].reshape(-1))
    return np.asarray(rows, dtype="float32")


def measured_metrics(actual: np.ndarray, predicted: np.ndarray, masks: np.ndarray, model_name: str) -> list[dict]:
    rows = []
    for index, target in enumerate(TARGET_COLUMNS):
        valid = masks[:, index].astype(bool)
        actual_target = actual[valid, index]
        predicted_target = predicted[valid, index]
        rows.append({
            "model": model_name,
            "target": target,
            "samples": int(valid.sum()),
            "MAE": mean_absolute_error(actual_target, predicted_target),
            "RMSE": mean_squared_error(actual_target, predicted_target) ** 0.5,
            "R2": r2_score(actual_target, predicted_target) if len(actual_target) > 1 else np.nan,
        })
    return rows


def even_indices(mask: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[positions]


def classic_models() -> dict:
    return {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, values, window_ends, labels, masks, lookback):
        self.values = values
        self.window_ends = window_ends
        self.labels = labels
        self.masks = masks
        self.lookback = lookback

    def __len__(self):
        return len(self.window_ends)

    def __getitem__(self, index):
        end = self.window_ends[index]
        return (
            torch.from_numpy(self.values[end - self.lookback:end]),
            torch.from_numpy(self.labels[index]),
            torch.from_numpy(self.masks[index]),
        )


class MLPForecaster(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, values):
        return self.net(values)


class CNNForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_size),
        )

    def forward(self, values):
        return self.net(values.transpose(1, 2))


class GRUForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.gru = nn.GRU(feature_size, 64, batch_first=True)
        self.head = nn.Linear(64, output_size)

    def forward(self, values):
        output, _ = self.gru(values)
        return self.head(output[:, -1])


class LSTMForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, 64, batch_first=True)
        self.head = nn.Linear(64, output_size)

    def forward(self, values):
        output, _ = self.lstm(values)
        return self.head(output[:, -1])


class BiLSTMForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, output_size)

    def forward(self, values):
        output, _ = self.lstm(values)
        return self.head(output[:, -1])


class ResidualTemporalBlock(nn.Module):
    def __init__(self, feature_size, channels, kernel_size):
        super().__init__()
        padding = kernel_size - 1
        self.conv1 = nn.Conv1d(feature_size, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.project = nn.Conv1d(feature_size, channels, 1) if feature_size != channels else nn.Identity()

    def forward(self, values):
        residual = self.project(values)
        output = self.relu(self.conv1(values)[..., :values.shape[-1]])
        output = self.conv2(output)[..., :values.shape[-1]]
        return self.relu(output + residual)


class TCNForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.block1 = ResidualTemporalBlock(feature_size, 64, 3)
        self.block2 = ResidualTemporalBlock(64, 64, 3)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, output_size))

    def forward(self, values):
        output = self.block1(values.transpose(1, 2))
        output = self.block2(output)
        return self.head(output)


class AttentionLSTMForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, 64, batch_first=True)
        self.attention = nn.Linear(64, 1)
        self.head = nn.Linear(64, output_size)

    def forward(self, values):
        output, _ = self.lstm(values)
        weights = torch.softmax(self.attention(output), dim=1)
        context = (output * weights).sum(dim=1)
        return self.head(context)


class TransformerForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.projection = nn.Linear(feature_size, 64)
        encoder = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=2)
        self.head = nn.Linear(64, output_size)

    def forward(self, values):
        output = self.encoder(self.projection(values))
        return self.head(output[:, -1])


class PatchTSTForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.patch = nn.Conv1d(feature_size, 64, kernel_size=16, stride=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, output_size)

    def forward(self, values):
        output = torch.relu(self.patch(values.transpose(1, 2)))
        return self.head(self.pool(output).flatten(1))


class TimesNetForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.branch_short = nn.Conv1d(feature_size, 64, 3, padding=1)
        self.branch_long = nn.Conv1d(feature_size, 64, 15, padding=7)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, output_size)

    def forward(self, values):
        sequence = values.transpose(1, 2)
        short = torch.relu(self.branch_short(sequence))
        long = torch.relu(self.branch_long(sequence))
        combined = torch.relu(short + long)
        return self.head(self.pool(combined).flatten(1))


class TFTStyleForecaster(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.input_gate = nn.Linear(feature_size, 64)
        self.grn = nn.Sequential(nn.Linear(feature_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.GLU())
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.attention = nn.Linear(64, 1)
        self.head = nn.Linear(64, output_size)

    def forward(self, values):
        gated = self.input_gate(values)
        grn_output = gated * torch.sigmoid(gated)
        output, _ = self.gru(grn_output)
        weights = torch.softmax(self.attention(output), dim=1)
        context = (output * weights).sum(dim=1)
        return self.head(context)


def deep_models(feature_size: int, lookback: int, output_size: int) -> dict:
    return {
        "MLP": MLPForecaster(feature_size * lookback, output_size),
        "1D_CNN": CNNForecaster(feature_size, output_size),
        "GRU": GRUForecaster(feature_size, output_size),
        "LSTM_trained": LSTMForecaster(feature_size, output_size),
        "BiLSTM": BiLSTMForecaster(feature_size, output_size),
        "TCN": TCNForecaster(feature_size, output_size),
        "Attention_LSTM": AttentionLSTMForecaster(feature_size, output_size),
        "Transformer": TransformerForecaster(feature_size, output_size),
        "PatchTST_style": PatchTSTForecaster(feature_size, output_size),
        "TimesNet_style": TimesNetForecaster(feature_size, output_size),
        "TFT_style": TFTStyleForecaster(feature_size, output_size),
    }


def train_deep_model(model, loader, validation_loader, device, epochs, patience):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")
    best_loss = float("inf")
    best_state = None
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        for inputs, labels, label_masks in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            label_masks = label_masks.to(device)
            optimizer.zero_grad()
            loss = (criterion(model(inputs), labels) * label_masks).sum() / label_masks.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for inputs, labels, label_masks in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                label_masks = label_masks.to(device)
                loss = (criterion(model(inputs), labels) * label_masks).sum() / label_masks.sum().clamp_min(1)
                losses.append(loss.item())
        validation_loss = float(np.mean(losses)) if losses else float("inf")
        if validation_loss < best_loss:
            best_loss = validation_loss
            no_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict_deep_model(model, loader, device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _, _ in loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    set_seed()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    print(f"Loading benchmark dataset: {args.input}")
    state = prepare_data(args.input, args.lookback, args.horizon)
    values = state["values"]
    window_ends = state["window_ends"]
    labels = state["labels"]
    scaled_labels = state["scaled_labels"]
    label_masks = state["label_masks"]
    train_mask = state["train"]
    validation_mask = state["validation"]
    test_mask = state["test"]

    print(
        f"Windows: {len(window_ends)} "
        f"train={int(train_mask.sum())} "
        f"validation={int(validation_mask.sum())} "
        f"test={int(test_mask.sum())}"
    )

    test_indices = even_indices(test_mask, args.classic_test_samples)
    test_actual = labels[test_indices]
    test_masks = label_masks[test_indices].astype(bool)
    all_rows = []

    if not args.skip_classic:
        train_indices = even_indices(train_mask, args.classic_train_samples)
        X_train = flatten_windows(values, window_ends, train_indices, args.lookback)
        X_test = flatten_windows(values, window_ends, test_indices, args.lookback)

        print(f"Classic training samples: {len(train_indices)}")
        for name, model in classic_models().items():
            print(f"Training classic model: {name}")
            started = time.time()
            predicted = np.empty((len(test_indices), len(TARGET_COLUMNS)), dtype="float32")
            for target_index, target in enumerate(TARGET_COLUMNS):
                target_train_valid = label_masks[train_indices, target_index].astype(bool)
                target_model = classic_models()[name]
                target_model.fit(
                    X_train[target_train_valid],
                    labels[train_indices[target_train_valid], target_index],
                )
                predicted[:, target_index] = target_model.predict(X_test).reshape(-1)
                joblib.dump(
                    target_model,
                    args.output_dir / f"{name}__{target}.joblib",
                )
            elapsed = time.time() - started
            all_rows.extend(measured_metrics(test_actual, predicted, test_masks, name))
            print(f"Finished {name} in {elapsed:.1f}s")

    if not args.skip_deep:
        dataset = SequenceDataset(
            values,
            window_ends,
            scaled_labels,
            label_masks,
            args.lookback,
        )

        def sequence_loader(mask, shuffle=False):
            indices = np.flatnonzero(mask).tolist()
            subset = Subset(dataset, indices)
            return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle)

        train_loader = sequence_loader(train_mask, shuffle=True)
        validation_loader = sequence_loader(validation_mask)

        test_subset = test_indices.tolist()
        test_loader = DataLoader(Subset(dataset, test_subset), batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        for name, model in deep_models(len(FEATURE_COLUMNS), args.lookback, len(TARGET_COLUMNS)).items():
            print(f"Training deep model: {name}")
            started = time.time()
            model, validation_loss = train_deep_model(
                model,
                train_loader,
                validation_loader,
                device,
                args.epochs,
                args.patience,
            )
            predicted_scaled = predict_deep_model(model, test_loader, device)
            predicted = state["target_scaler"].inverse_transform(predicted_scaled)
            elapsed = time.time() - started
            all_rows.extend(measured_metrics(test_actual, predicted, test_masks, name))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": FEATURE_COLUMNS,
                    "targets": TARGET_COLUMNS,
                    "lookback": args.lookback,
                    "validation_loss": validation_loss,
                },
                args.output_dir / f"{name}.pt",
            )
            print(f"Finished {name} in {elapsed:.1f}s")

    results = pd.DataFrame(all_rows)
    results = results.sort_values(["target", "RMSE"], ascending=[True, True])
    results.to_csv(args.output_dir / "benchmark_results.csv", index=False)

    metadata = {
        "seed": SEED,
        "input": str(args.input),
        "targets": TARGET_COLUMNS,
        "features": FEATURE_COLUMNS,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "split": [0.70, 0.15, 0.15],
        "classic_train_samples": args.classic_train_samples,
        "classic_test_samples": args.classic_test_samples,
        "epochs": args.epochs,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }
    with (args.output_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nBenchmark results")
    print(results.to_string(index=False))
    print(f"\nSaved benchmark artifacts to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
