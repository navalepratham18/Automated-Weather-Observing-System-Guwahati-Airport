"""Physics and conditional-median based filling for Guwahati cloud visibility.

This script uses no machine-learning or deep-learning model.

Priority:
1. Preserve original measured CLOUD_Vertical_Visibility values.
2. Build deterministic lookup medians from valid Guwahati observations.
3. Apply explicit weather/cloud physics rules.
4. Use broad conditional fallbacks and the Guwahati dataset median.

The output is a new CSV with original values retained and two new columns:
- CLOUD_Vertical_Visibility_FILLED
- CLOUD_Vertical_Visibility_FILL_METHOD

Optional:
- CLOUD_Vertical_Visibility_FILL_REASON
- CLOUD_Vertical_Visibility_FILL_CONFIDENCE

Paths default to the repository's physics-corrected unified dataset and an
experiment output under 08_unified_dataset. Absolute paths may also be passed
with --guwahati and --output.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


TARGET = "CLOUD_Vertical_Visibility"
OUTPUT_TARGET = "CLOUD_Vertical_Visibility_FILLED"
OUTPUT_METHOD = "CLOUD_Vertical_Visibility_FILL_METHOD"
OUTPUT_REASON = "CLOUD_Vertical_Visibility_FILL_REASON"
OUTPUT_CONFIDENCE = "CLOUD_Vertical_Visibility_FILL_CONFIDENCE"


def number(value: object) -> float:
    if value is None:
        return math.nan
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return math.nan


def bucket(value: float, boundaries: Iterable[float], fallback: int = -1) -> int:
    """Return a stable ordinal bucket for a numeric value."""
    if not math.isfinite(value):
        return fallback
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return index
    return len(tuple(boundaries))


def cloud_base_bucket(value: float) -> int:
    return bucket(value, (100, 300, 500, 1000, 2000, 3000, 5000, 8000, 15000, 25000), -1)


def octa_bucket(value: float) -> int:
    return bucket(value, (0, 1, 2, 3, 4, 5, 6, 7, 8), -1)


def humidity_bucket(value: float) -> int:
    return bucket(value, (50, 70, 85, 92, 95, 98, 100), -1)


def rain_bucket(value: float) -> int:
    return bucket(value, (0, 0.1, 1, 5, 10, 25, 50), -1)


def mor_bucket(value: float) -> int:
    return bucket(value, (100, 500, 1000, 2000, 5000, 9000, 10000), -1)


def rvr_bucket(value: float) -> int:
    return bucket(value, (300, 600, 1000, 1500, 3000, 6000, 10000), -1)


def temperature_bucket(value: float) -> int:
    return bucket(value, (-5, 5, 15, 20, 25, 30, 35, 40), -1)


def dewpoint_bucket(value: float) -> int:
    return bucket(value, (-5, 5, 10, 15, 20, 25, 30), -1)


def pressure_bucket(value: float) -> int:
    return bucket(value, (970, 990, 1000, 1005, 1010, 1015, 1020, 1035), -1)


def wind_bucket(value: float) -> int:
    return bucket(value, (0.5, 2, 5, 10, 15, 25, 40), -1)


def median(values: list[float]) -> float:
    return statistics.median(values)


class ConditionalLookup:
    """Stores values for several progressively broader lookup keys."""

    def __init__(self) -> None:
        self.levels: dict[str, dict[tuple[int, ...], list[float]]] = {
            "weather": defaultdict(list),
            "cloud": defaultdict(list),
            "cloud_rain": defaultdict(list),
            "base_rain": defaultdict(list),
            "base": defaultdict(list),
            "rain": defaultdict(list),
            "global": defaultdict(list),
        }
        self.values: list[float] = []

    def add(self, row: dict[str, str], target: float) -> None:
        if not math.isfinite(target):
            return

        base1 = number(row.get("CLOUD_Cloud_Base_1"))
        octa1 = number(row.get("CLOUD_OCTA1"))
        rain = number(row.get("PW_Rain_Intensity"))
        humidity = number(row.get("PTU_Humidity"))
        mor = number(row.get("VIS_Visibility_MOR"))
        rvr = number(row.get("VIS_RVR"))
        temperature = number(row.get("PTU_Temperature"))
        dewpoint = number(row.get("PTU_DewPoint"))
        pressure = number(row.get("PTU_Pressure_QNH"))
        wind = number(row.get("WIND_Wind_Speed"))
        gust = number(row.get("WIND_Wind_Gust"))

        self.values.append(target)

        weather_key = (
            cloud_base_bucket(base1),
            rain_bucket(rain),
            humidity_bucket(humidity),
            octa_bucket(octa1),
            mor_bucket(mor),
            rvr_bucket(rvr),
        )
        self.levels["weather"][weather_key].append(target)

        cloud_key = (
            cloud_base_bucket(base1),
            octa_bucket(octa1),
            rain_bucket(rain),
            humidity_bucket(humidity),
            temperature_bucket(temperature),
            dewpoint_bucket(dewpoint),
        )
        self.levels["cloud"][cloud_key].append(target)

        cloud_rain_key = (
            octa_bucket(octa1),
            rain_bucket(rain),
            humidity_bucket(humidity),
            mor_bucket(mor),
            rvr_bucket(rvr),
        )
        self.levels["cloud_rain"][cloud_rain_key].append(target)

        base_rain_key = (
            cloud_base_bucket(base1),
            rain_bucket(rain),
            humidity_bucket(humidity),
        )
        self.levels["base_rain"][base_rain_key].append(target)

        base_key = (cloud_base_bucket(base1),)
        self.levels["base"][base_key].append(target)

        rain_key = (rain_bucket(rain), humidity_bucket(humidity))
        self.levels["rain"][rain_key].append(target)

        # Pressure and wind contribute to the final key through this retained
        # broad fallback rather than being ignored in the lookup construction.
        global_key = (
            pressure_bucket(pressure),
            wind_bucket(wind),
            wind_bucket(gust),
            temperature_bucket(temperature),
        )
        self.levels["global"][global_key].append(target)
        self.levels["global"][(-1, -1, -1, -1)].append(target)

    def get_median(self, row: dict[str, str], level: str) -> tuple[float | None, tuple[int, ...] | None]:
        if level == "global":
            if self.values:
                return median(self.values), (-1, -1, -1, -1)
            return None, (-1, -1, -1, -1)

        base1 = number(row.get("CLOUD_Cloud_Base_1"))
        octa1 = number(row.get("CLOUD_OCTA1"))
        rain = number(row.get("PW_Rain_Intensity"))
        humidity = number(row.get("PTU_Humidity"))
        mor = number(row.get("VIS_Visibility_MOR"))
        rvr = number(row.get("VIS_RVR"))
        temperature = number(row.get("PTU_Temperature"))
        dewpoint = number(row.get("PTU_DewPoint"))
        pressure = number(row.get("PTU_Pressure_QNH"))
        wind = number(row.get("WIND_Wind_Speed"))
        gust = number(row.get("WIND_Wind_Gust"))

        if level == "weather":
            key = (
                cloud_base_bucket(base1),
                rain_bucket(rain),
                humidity_bucket(humidity),
                octa_bucket(octa1),
                mor_bucket(mor),
                rvr_bucket(rvr),
            )
        elif level == "cloud":
            key = (
                cloud_base_bucket(base1),
                octa_bucket(octa1),
                rain_bucket(rain),
                humidity_bucket(humidity),
                temperature_bucket(temperature),
                dewpoint_bucket(dewpoint),
            )
        elif level == "cloud_rain":
            key = (
                octa_bucket(octa1),
                rain_bucket(rain),
                humidity_bucket(humidity),
                mor_bucket(mor),
                rvr_bucket(rvr),
            )
        elif level == "base_rain":
            key = (cloud_base_bucket(base1), rain_bucket(rain), humidity_bucket(humidity))
        elif level == "base":
            key = (cloud_base_bucket(base1),)
        elif level == "rain":
            key = (rain_bucket(rain), humidity_bucket(humidity))
        else:
            key = (
                pressure_bucket(pressure),
                wind_bucket(wind),
                wind_bucket(gust),
                temperature_bucket(temperature),
            )

        values = self.levels[level].get(key)
        if values:
            return median(values), key
        return None, key

    def global_median(self) -> float:
        if not self.values:
            raise ValueError("No valid target values are available for median fallback")
        return median(self.values)


def iter_csv(path: Path) -> tuple[csv.DictReader, object]:
    handle = path.open("r", newline="", encoding="utf-8-sig")
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"CSV has no header: {path}")
    return reader, handle


def build_lookup(path: Path, target_column: str, lookup: ConditionalLookup, source_name: str) -> int:
    reader, handle = iter_csv(path)
    if target_column not in (reader.fieldnames or []):
        handle.close()
        raise ValueError(f"{path}: missing required column {target_column}")

    count = 0
    try:
        for row in reader:
            target = number(row.get(target_column))
            if math.isfinite(target):
                lookup.add(row, target)
                count += 1
    finally:
        handle.close()

    print(f"{source_name}: loaded {count:,} valid {target_column} observations")
    return count


def octa_sum(row: dict[str, str]) -> float:
    values = [
        number(row.get("CLOUD_OCTA1")),
        number(row.get("CLOUD_OCTA2")),
        number(row.get("CLOUD_OCTA3")),
        number(row.get("CLOUD_OCTA4")),
    ]
    valid = [value for value in values if math.isfinite(value)]
    return sum(valid) if valid else math.nan


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def physics_estimate(row: dict[str, str]) -> tuple[float, str, float]:
    """Explicit deterministic clear/cloudy/rainy rule estimate."""
    base1 = number(row.get("CLOUD_Cloud_Base_1"))
    base2 = number(row.get("CLOUD_Cloud_Base_2"))
    base3 = number(row.get("CLOUD_Cloud_Base_3"))
    octa1 = number(row.get("CLOUD_OCTA1"))
    octas = octa_sum(row)
    humidity = number(row.get("PTU_Humidity"))
    dewpoint = number(row.get("PTU_DewPoint"))
    temperature = number(row.get("PTU_Temperature"))
    rain = number(row.get("PW_Rain_Intensity"))
    rain_sum = number(row.get("PW_Rain_Sum"))
    mor = number(row.get("VIS_Visibility_MOR"))
    rvr = number(row.get("VIS_RVR"))
    wind = number(row.get("WIND_Wind_Speed"))
    gust = number(row.get("WIND_Wind_Gust"))

    clear_score = 0
    severe_score = 0
    reasons: list[str] = []

    if math.isfinite(base1) and base1 >= 5000:
        clear_score += 4
    elif math.isfinite(base1) and base1 < 500:
        severe_score += 4
        reasons.append("very low cloud base")

    if math.isfinite(octa1) and octa1 <= 2:
        clear_score += 3
    elif math.isfinite(octa1) and octa1 >= 7:
        severe_score += 4
        reasons.append("overcast octa")

    if math.isfinite(octas) and octas >= 24:
        severe_score += 2

    if math.isfinite(rain) and rain <= 0.1:
        clear_score += 3
    elif math.isfinite(rain) and rain >= 5:
        severe_score += 5
        reasons.append("heavy rain intensity")
    elif math.isfinite(rain) and rain > 0.1:
        severe_score += 2
        reasons.append("light rain or rain onset")

    if math.isfinite(rain_sum) and rain_sum >= 10:
        severe_score += 3
        reasons.append("substantial rain accumulation")

    if math.isfinite(humidity) and humidity <= 85:
        clear_score += 2
    elif math.isfinite(humidity) and humidity >= 98:
        severe_score += 3
        reasons.append("near-saturated humidity")
    elif math.isfinite(humidity) and humidity >= 92:
        severe_score += 1

    if math.isfinite(mor) and mor >= 5000:
        clear_score += 3
    elif math.isfinite(mor) and mor < 1000:
        severe_score += 4
        reasons.append("low MOR visibility")

    if math.isfinite(rvr) and rvr >= 3000:
        clear_score += 2
    elif math.isfinite(rvr) and rvr < 600:
        severe_score += 4
        reasons.append("low RVR")

    if math.isfinite(base2) and base2 < 1000:
        severe_score += 1
    if math.isfinite(base3) and base3 < 1500:
        severe_score += 1

    if math.isfinite(temperature) and math.isfinite(dewpoint):
        spread = temperature - dewpoint
        if spread >= 5:
            clear_score += 1
        elif spread <= 1:
            severe_score += 2
            reasons.append("temperature-dewpoint spread near zero")

    if math.isfinite(wind) and math.isfinite(gust) and gust >= 20:
        severe_score += 1
        reasons.append("strong gust")

    if severe_score >= 7:
        estimate = 600
        reason = "severe: " + (", ".join(reasons) if reasons else "combined cloud/weather indicators")
        confidence = 0.78
    elif severe_score >= 4:
        estimate = 1500
        reason = "cloudy/rainy: " + (", ".join(reasons) if reasons else "combined adverse indicators")
        confidence = 0.68
    elif clear_score >= 10:
        estimate = 15000
        reason = "clear: high cloud base, low octa/rain, low humidity, and good visibility"
        confidence = 0.82
    elif clear_score >= 6:
        estimate = 8000
        reason = "mostly clear: several positive clear-weather indicators"
        confidence = 0.62
    else:
        estimate = 3000
        reason = "mixed: insufficient evidence for clear or severe conditions"
        confidence = 0.45

    return clamp(estimate, 100, 15000), reason, confidence


def find_lookup_value(
    row: dict[str, str],
    guwahati_lookup: ConditionalLookup,
) -> tuple[float | None, str, float]:
    for level in ("weather", "cloud", "cloud_rain", "base_rain"):
        value, _ = guwahati_lookup.get_median(row, level)
        if value is not None:
            return value, f"guwahati_{level}_conditional_median", 0.88 if level in ("weather", "cloud") else 0.78

    return None, "", 0.0


def write_output(
    guwahati_path: Path,
    output_path: Path,
    guwahati_lookup: ConditionalLookup,
    progress_every: int,
) -> dict[str, int | float]:
    reader, handle = iter_csv(guwahati_path)
    missing_required = {
        "Datetime",
        TARGET,
        "CLOUD_Cloud_Base_1",
        "CLOUD_OCTA1",
        "PTU_Humidity",
        "PW_Rain_Intensity",
        "VIS_Visibility_MOR",
        "VIS_RVR",
        "WIND_Wind_Speed",
        "WIND_Wind_Gust",
    } - set(reader.fieldnames or [])
    if missing_required:
        handle.close()
        raise ValueError(f"{guwahati_path}: missing required columns: {sorted(missing_required)}")

    guwahati_global = guwahati_lookup.global_median()

    counts: dict[str, int] = defaultdict(int)
    original_count = 0
    filled_count = 0
    total_rows = 0

    fieldnames = list(reader.fieldnames or [])
    for new_column in (OUTPUT_TARGET, OUTPUT_METHOD, OUTPUT_REASON, OUTPUT_CONFIDENCE):
        if new_column not in fieldnames:
            fieldnames.append(new_column)

    try:
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()

            for row_index, row in enumerate(reader, start=1):
                total_rows = row_index
                original = number(row.get(TARGET))

                if math.isfinite(original):
                    filled = original
                    method = "original_measured"
                    reason = "preserved original non-missing sensor value"
                    confidence = 1.0
                    original_count += 1
                else:
                    lookup_value, lookup_method, lookup_confidence = find_lookup_value(
                        row, guwahati_lookup
                    )
                    if lookup_value is not None:
                        filled = clamp(lookup_value, 100, 15000)
                        method = lookup_method
                        reason = f"conditional median from {lookup_method}"
                        confidence = lookup_confidence
                    else:
                        physics_value, reason, confidence = physics_estimate(row)
                        filled = physics_value
                        method = "physics_rules"
                        if not math.isfinite(filled):
                            filled = guwahati_global
                            method = "guwahati_global_median_fallback"
                            confidence = 0.25

                    filled_count += 1

                counts[method] += 1
                row[OUTPUT_TARGET] = f"{filled:.3f}"
                row[OUTPUT_METHOD] = method
                row[OUTPUT_REASON] = reason
                row[OUTPUT_CONFIDENCE] = f"{confidence:.2f}"
                writer.writerow(row)

                if row_index % progress_every == 0:
                    print(f"Processed {row_index:,} rows")
    finally:
        handle.close()

    return {
        "total_rows": total_rows,
        "original_count": original_count,
        "filled_count": filled_count,
        "output_rows": total_rows,
        "method_counts": counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based physics and conditional-median filling for Guwahati cloud vertical visibility"
    )
    parser.add_argument(
        "--guwahati",
        type=Path,
        default=PROJECT_ROOT
        / "08_unified_dataset"
        / "guwahati_unified_lstm_dataset_v3_physics_corrected.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "08_unified_dataset"
        / "guwahati_unified_lstm_dataset_v3_vertical_visibility_rule_filled.csv",
    )
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.guwahati.exists():
        print(f"ERROR: Guwahati file not found: {args.guwahati}", file=sys.stderr)
        return 1
    if args.progress_every <= 0:
        print("ERROR: --progress-every must be greater than zero", file=sys.stderr)
        return 1

    guwahati_lookup = ConditionalLookup()
    print("Building Guwahati conditional lookup from original measured targets...")
    guwahati_valid = build_lookup(args.guwahati, TARGET, guwahati_lookup, "Guwahati")
    if guwahati_valid == 0:
        print("WARNING: Guwahati lookup has no valid original targets", file=sys.stderr)

    print(f"Writing output to {args.output}...")
    summary = write_output(
        args.guwahati,
        args.output,
        guwahati_lookup,
        args.progress_every,
    )

    print("\nSummary")
    print(f"Total rows: {summary['total_rows']:,}")
    print(f"Original measured rows preserved: {summary['original_count']:,}")
    print(f"Missing rows filled: {summary['filled_count']:,}")
    print("Fill method counts:")
    for method, count in sorted(summary["method_counts"].items(), key=lambda item: (-item[1], item[0])):
        print(f"  {method}: {count:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
