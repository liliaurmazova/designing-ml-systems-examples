"""Topic 5, Step 2: Time-series feature engineering without future leakage.

NOTE: the requested dataset `nyu-mll/nyctaxi` does not exist on the Hugging
Face Hub; this script uses `vmozzon/yellow_tripdata` instead — a mirror of
the official NYC TLC yellow-cab data with real pickup AND dropoff
timestamps, so trip duration is computable exactly.

Two leakage-sensitive techniques demonstrated:

* Chronological split — with time series, a random split lets the model
  train on the future and predict the past. We train on the FIRST 80% of
  trips and test on the LAST 20%.

* Cyclical encoding — hour 23 and hour 0 are 1 apart, not 23 apart.
  Mapping t onto a circle fixes the wrap-around:
      sin(2*pi*t/T), cos(2*pi*t/T)   (T=24 for hours, T=7 for weekdays)
  Both are needed: one alone maps two different times to the same value.

* Rolling stats with .shift(1) — the rolling mean/std of recent trip
  durations is a strong signal (traffic regime), but rolling().mean() at
  row i INCLUDES row i's own duration — the target itself. .shift(1)
  moves the window one row back, so row i's feature is computed strictly
  from earlier trips. Test-row windows may include train-period trips:
  that is fine (the past is legitimately known in production) — only the
  FUTURE must never leak.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Load, clean, and build the target ---------------------------------------
# The dataset holds Apr-Jun 2024 (~10.7M trips) plus a handful of rows with
# corrupt timestamps (2002, 2009, 2026...). Load only the needed columns,
# filter to the valid range, and keep every 10th trip (~1M) for speed —
# rolling windows still see ~400 trips/hour after downsampling.
ds = load_dataset("vmozzon/yellow_tripdata", split="train").select_columns(
    ["tpep_pickup_datetime", "tpep_dropoff_datetime",
     "trip_distance", "passenger_count"])
df = ds.to_pandas()
df = df[(df["tpep_pickup_datetime"] >= "2024-04-01")
        & (df["tpep_pickup_datetime"] < "2024-07-01")]
df["duration_min"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

# Basic sanity filter: keep plausible trips (1-120 min, positive distance)
df = df[(df["duration_min"].between(1, 120)) & (df["trip_distance"] > 0)]
df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
df = df.iloc[::10].reset_index(drop=True)
print(f"{len(df)} trips from {df.tpep_pickup_datetime.min()} "
      f"to {df.tpep_pickup_datetime.max()}")

# --- Feature engineering (computed on the full timeline, past-only) ----------
t = df["tpep_pickup_datetime"]
hour = t.dt.hour + t.dt.minute / 60
dow = t.dt.dayofweek

# 1) Cyclical encodings
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

# 2) Rolling stats over time-based windows, strictly past-only via shift(1):
# rolling("60min") at row i covers (t_i - 60min, t_i] INCLUDING row i's own
# duration -> shift(1) so the feature for row i ends at row i-1.
ts = df.set_index("tpep_pickup_datetime")["duration_min"]
for window, label in [("60min", "1h"), ("180min", "3h")]:
    roll = ts.rolling(window)
    df[f"roll_mean_{label}"] = roll.mean().shift(1).to_numpy()
    df[f"roll_std_{label}"] = roll.std().shift(1).to_numpy()

BASE = ["trip_distance", "passenger_count"]
ENGINEERED = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
              "roll_mean_1h", "roll_std_1h", "roll_mean_3h", "roll_std_3h"]

# --- Strict chronological 80/20 split ----------------------------------------
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
print(f"Chronological split: train ends {train.tpep_pickup_datetime.max()}, "
      f"test starts {test.tpep_pickup_datetime.min()}")

# NaNs (first rows have no rolling history) are filled with TRAIN-period
# medians only — filling from full-data stats would leak test information.
fill = train[BASE + ENGINEERED].median()


def run_ridge(features):
    """Scale (fit on train only), fit Ridge, report test MAE / RMSE."""
    X_tr = train[features].fillna(fill).to_numpy(float)
    X_te = test[features].fillna(fill).to_numpy(float)
    scaler = StandardScaler().fit(X_tr)
    model = Ridge(alpha=1.0).fit(scaler.transform(X_tr), train["duration_min"])
    pred = model.predict(scaler.transform(X_te))
    return (mean_absolute_error(test["duration_min"], pred),
            root_mean_squared_error(test["duration_min"], pred))


mae_b, rmse_b = run_ridge(BASE)
mae_e, rmse_e = run_ridge(BASE + ENGINEERED)

print(f"\n{'features':<38}{'test MAE':>10}{'test RMSE':>11}")
print(f"{'baseline (distance, passengers)':<38}{mae_b:>10.3f}{rmse_b:>11.3f}")
print(f"{'+ cyclical time + rolling stats':<38}{mae_e:>10.3f}{rmse_e:>11.3f}")
print(f"{'improvement':<38}{mae_b - mae_e:>10.3f}{rmse_b - rmse_e:>11.3f} "
      f"({(mae_b - mae_e) / mae_b:.1%} MAE)")
