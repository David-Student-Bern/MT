import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path

# Automatically add the project root (MT folder) to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# flake8: noqa: E402
import pandas as pd
import numpy as np
from utils.data_loader import find_repo_root
from utils.modeling import make_lags, make_multistep_target
from datetime import datetime
import logging
from sklearn.linear_model import MultiTaskLassoCV ,LinearRegression, Lasso
import joblib
from sklearn.preprocessing import StandardScaler

# =======================================================================================================
# Logging configuration
# =======================================================================================================
LOG_DIR = find_repo_root() / Path("Analysis/modeling/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "multi_forecast_CV.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # writes to file
        logging.StreamHandler(sys.stdout)  # prints to console
    ]
)

# === Start timing ===
log_start_time = datetime.now()
logging.info("=======================================")
logging.info(f"Script started: {log_start_time}")

# =======================================================================================================
# Settings
# =======================================================================================================
sampling_rate = '5min'
columns_needed = ['time', 'orbital_decay', 'trend','|avg B|', 'F10.7 (LASP)', 'Bz GSE', 'Flow Speed (km/s', 'Temperature (K)', 'Kp (LASP)', 'median_decay_last_7D', 'median_decay_last_14D', 'median_decay_last_30D']
start_time = '2023-01-01 00:00:00'
end_time = '2024-07-01 00:00:00'
target = 'trend'
# ---- forecast parameters ----
n = 12  # hours ahead
m = 15  # minutes steps
# ---- Kp sorting ----
Kp_sorting = False  # if True, train only on times with Kp >= 4
# ---- model parameters ----
model_type = 'MultiTaskLassoCV'
model_number = 3  # just for naming purposes
model_name = f"{model_type}_{target}_model_{model_number}"
if model_type == 'lasso':
    model = Lasso(alpha=0.3)
elif model_type == 'LinearRegression':
    model = LinearRegression()
elif model_type == 'MultiTaskLassoCV':
    alphas = alphas = np.logspace(-2, 1, 40)
    model = MultiTaskLassoCV(
        alphas=alphas, 
        cv=5,
        max_iter=5000,
        n_jobs=5
    )
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# ---- name scaler ----
scaler_name = f"{model_type}_{target}_scaler_{model_number}"

# logging settings
logging.info("-- Basic Settings --")
logging.info(f"Sampling rate: {sampling_rate}")
logging.info(f"Time range: {start_time} to {end_time}")
logging.info(f"Kp sorting: {Kp_sorting}")
if Kp_sorting:
    logging.info("-->  Training only on times with Kp >= 4")

# =======================================================================================================
# Load Data
# =======================================================================================================

DATA_DIR = find_repo_root() / Path("Dataset/modeling/")
DATA_file = DATA_DIR / Path(f"GFOC_modeling_{sampling_rate}.parquet")
GFOC_data = pd.read_parquet(DATA_file, columns=columns_needed)
# filter by time range
GFOC_data = GFOC_data.loc[start_time:end_time]
print(f"✅ Successfully loaded data with shape: {GFOC_data.shape}")
# column names
print("Column names in the dataset:")
print(GFOC_data.columns.tolist())
print("Sample rate:", pd.infer_freq(GFOC_data.index))

# =======================================================================================================
# Setup modeling parameters
# =======================================================================================================
dt_seconds = (GFOC_data.index[1] - GFOC_data.index[0]).total_seconds()
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

X_train = make_lags(GFOC_data, lag_config_trend)

y_stepsize = int(m * (60 / dt_seconds)) # every 15 minutes
y_train = make_multistep_target(GFOC_data[[target]].copy(), steps=int(n * 60 * (60 / dt_seconds)), stepsize=y_stepsize)

# Shifting has created indexes that don't match. Only keep times for
na_mask = X_train.isna().any(axis=1) | y_train.isna().any(axis=1)
X_train = X_train[~na_mask]
y_train = y_train[~na_mask]

# Rename columns to avoid MultiIndex issues
X_train.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in X_train.columns]

# log modeling parameters
logging.info("--Modeling Parameters--")
logging.info(f"lag_config_trend: {len(lag_config_trend)} different features")
for feature, cfg in lag_config_trend.items():
    n_lags = len(cfg.get("lags", []))
    logging.info(f'    "{feature}": {n_lags} lags,')
logging.info(f"Total features: {X_train.shape[1]}")
logging.info(f"Target: {target}")
logging.info(f"Forecast horizon: {n} hours ahead, stepsize: {m} minutes (={y_train.shape[1]} steps)")
logging.info(f"Training samples: {X_train.shape[0]}")
# =======================================================================================================
# Training
# =======================================================================================================
scaler = StandardScaler()
# scale and convert back to DataFrames
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    index=X_train.index,
    columns=X_train.columns
)
# save scaler
scaler_path = find_repo_root() / Path(f"Analysis/modeling/saved_models/{scaler_name}.pkl")
joblib.dump(scaler, scaler_path)

if Kp_sorting:
    # Train only on active times
    active_times = GFOC_data[GFOC_data['Kp (LASP)'] >= 4].index.intersection(X_train_scaled.index)
    model.fit(
        X_train_scaled.loc[active_times],
        y_train.loc[active_times]
    )
    logging.info(f"Trained on {len(active_times)} samples (Kp >= 4)")
else:
    # Train on all times
    model.fit(X_train_scaled, y_train)

# save model
model_path = find_repo_root() / Path(f"Analysis/modeling/saved_models/{model_name}.pkl")
joblib.dump(model, model_path)

# logging model info
logging.info("-- Model Description --")
logging.info(f"Model class: {model.__class__.__name__}")

try:
    params = model.get_params()
    for k, v in params.items():
        logging.info(f"  {k}: {v}")
except Exception:
    logging.info("  No parameters available")
logging.info(f"Model ID: {model_name}")
logging.info(f"Scaler ID: {scaler_name}")

# === End timing ===
log_end_time = datetime.now()
elapsed = log_end_time - log_start_time
logging.info(f"Script finished: {log_end_time}")
logging.info(f"Elapsed time: {elapsed}")