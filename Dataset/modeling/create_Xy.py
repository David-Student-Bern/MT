import sys
from pathlib import Path
# Automatically add the project root (MT folder) to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# flake8: noqa: E402
import pandas as pd
from utils.data_loader import find_repo_root
from utils.modeling import make_lags, make_multistep_target

# =======================================================================================================
# Load Data
# =======================================================================================================
# '20s' does not work due to RAM limitations
sampling_rate = '5min'  # available: '20s', '1min', '5min'
columns = ['orbital_decay', 'trend', '|avg B|', 'F10.7 (LASP)', 'Bz GSE', 'Flow Speed (km/s', 'Temperature (K)', 'median_decay_last_7D', 'median_decay_last_14D', 'median_decay_last_30D']
DATA_file = find_repo_root() / Path(f"Dataset/modeling/GFOC_modeling_{sampling_rate}.parquet")
df = pd.read_parquet(DATA_file, columns=columns)

print(f"✅ Successfully loaded data with shape: {df.shape}")
# column names
print("Column names in the dataset:")
print(df.columns.tolist())
print("Sample rate:", pd.infer_freq(df.index))

# =======================================================================================================
# Setup modeling parameters
# =======================================================================================================
dt_seconds = (df.index[1] - df.index[0]).total_seconds()
hour = int(60 / (dt_seconds / 60))  # number of samples in one hour
lag_config_trend = {
    # removed having target data as feature to avoid data leakage given the target signal is spread out
    # "orbital_decay": {"lags": list(range(1*hour, 24*hour, 3*hour))}, 
    "|avg B|": {'lags': list(range(1, 3*hour, max(hour//10, 1))) + list(range(3*hour, 50*hour, max(hour//2, 1)))},
    "Bz GSE": {'lags': list(range(1, 3*hour, max(hour//10, 1))) + list(range(3*hour, 50*hour, max(hour//2, 1)))},
    "Temperature (K)": {'lags': list(range(1, 3*hour, max(hour//10, 1))) + list(range(3*hour, 50*hour, max(hour//2, 1)))},
    "Flow Speed (km/s": {'lags': list(range(1, 3*hour, max(hour//10, 1))) + list(range(3*hour, 50*hour, max(hour//2, 1)))},
    "F10.7 (LASP)": {'lags': list(range(1, 24*hour, 6*hour)) + list(range(24*hour, 50*hour, 12*hour))},
    "median_decay_last_7D": {'lags': list(range(1, 50*hour, 12*hour))},
    "median_decay_last_14D": {'lags': list(range(1, 50*hour, 12*hour))},
    "median_decay_last_30D": {'lags': list(range(1, 50*hour, 12*hour))}
}

X_trend_list = []

y_trend_list = []
target = 'orbital_decay'
n = 12  # hours ahead
m = 15  # minutes steps
y_stepsize = int(m * (60 / dt_seconds)) # every 15 minutes

X = make_lags(df, lag_config_trend)

y = make_multistep_target(df[[target]].copy(), steps=int(n * 60 * (60 / dt_seconds)), stepsize=y_stepsize)

# Shifting has created indexes that don't match. Only keep times for
na_mask = X.isna().any(axis=1) | y.isna().any(axis=1)
X = X[~na_mask]
y = y[~na_mask]

# Rename columns to avoid MultiIndex issues
X.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in X.columns]

print(f"Final feature matrix shape: {X.shape}")
print(f"Final target matrix shape: {y.shape}")

# Save to parquet
OUTPUT_file_X = find_repo_root() / Path(f"Dataset/modeling/X_{sampling_rate}.parquet")
OUTPUT_file_y = find_repo_root() / Path(f"Dataset/modeling/y_{target}_{m}_{sampling_rate}.parquet")

X.to_parquet(OUTPUT_file_X)
y.to_parquet(OUTPUT_file_y)