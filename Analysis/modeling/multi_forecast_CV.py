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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from statsmodels.tsa.seasonal import STL
from utils.data_loader import load_parquet, find_repo_root
from utils.plot_loader import create_subset
from utils.modeling import make_lags, make_multistep_target
from datetime import datetime
import logging
from sklearn.linear_model import MultiTaskLassoCV #,LinearRegression, Lasso
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
start_time = datetime.now()
logging.info("=======================================")
logging.info(f"Script started: {start_time}")

# =======================================================================================================
# Functions
# =======================================================================================================
def load_intervals(csv_paths):
    """Load start/end intervals from one or more CSV files."""
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path, parse_dates=["start", "end"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def merge_intervals(df):
    """Merge overlapping time intervals."""
    # Sort intervals by start time
    df = df.sort_values("start").reset_index(drop=True)

    merged = []
    current_start = df.loc[0, "start"]
    current_end = df.loc[0, "end"]

    for i in range(1, len(df)):
        row_start = df.loc[i, "start"]
        row_end = df.loc[i, "end"]

        if row_start <= current_end:  
            # Overlapping or touching intervals → extend the end if needed
            current_end = max(current_end, row_end)
        else:  
            # No overlap → push current and reset
            merged.append((current_start, current_end))
            current_start = row_start
            current_end = row_end

    # Append final interval
    merged.append((current_start, current_end))

    return pd.DataFrame(merged, columns=["start", "end"])

def split_by_time_gap(df, dt_seconds):
    """
    Split a DataFrame into subsets whenever time difference > dt_seconds.
    df.index must be a DatetimeIndex unless time_col is given.
    """
    t = pd.to_datetime(df.index)

    gaps = t.diff().total_seconds() > dt_seconds
    groups = pd.Series(gaps.cumsum(), index=df.index)

    return [df.loc[groups == g] for g in groups.unique()]

def plot_subsets(true_subsets, pred_subsets, title_prefix):
    rows = len(true_subsets)
    i = 0
    fig, axes = plt.subplots(rows, 1, figsize=(11, 3*rows))

    if rows == 1:
        axes = [axes]

    for ax, df_true, df_pred in zip(axes, true_subsets, pred_subsets):
        ax.plot(df_true.index, df_true['mean'], color='black')
        ax.plot(df_pred.index, df_pred['mean'], color='C0')
        ax.fill_between(df_pred.index,
                        df_pred['mean'] - 1.96 * df_pred['sem'],
                        df_pred['mean'] + 1.96 * df_pred['sem'],
                        color='C0', alpha=0.3)
        ax.plot(df_pred.index, df_pred['horizon0'], color='green', alpha=0.3)

        ax.legend(['True Trend', 'Mean Prediction', '95% Prediction Band', 'Raw Prediction (horizon 0)'])
        ax.grid(True)

        subset_start = df_true.index.min().strftime('%Y-%m-%d %H:%M:%S')
        ax.set_title(f"{title_prefix} subset {i}, starting {subset_start}")
        i += 1

    plt.tight_layout()
    plt.show()

def log_model_description(model, prefix="Model"):
    logging.info(f"-- {prefix} Description --")
    logging.info(f"Model class: {model.__class__.__name__}")

    try:
        params = model.get_params()
        for k, v in params.items():
            logging.info(f"  {k}: {v}")
    except Exception:
        logging.info("  No parameters available")

# =======================================================================================================
# Load Data
# =======================================================================================================
# Load only specific columns needed for analysis
columns_needed = ['time', 'orbital_decay', '|avg B|', 'F10.7 (LASP)', 'Bz GSE', 'Flow Speed (km/s', 'Temperature (K)', 'Kp (LASP)']

GFOC_data = load_parquet(columns=columns_needed)
print(GFOC_data.head())


csv_files = [
    find_repo_root() / Path("Analysis/modeling/subsets_eflag.csv"),
    find_repo_root() / Path("Analysis/modeling/subsets_meanstd.csv")
]

intervals_df = load_intervals(csv_files)
# add timedelta
t01 = pd.Timedelta(hours=72)  # extension before 0->1
t10 = pd.Timedelta(hours=12)  # extension after 1->0
intervals_df['start'] = intervals_df['start'] - t01
intervals_df['end'] = intervals_df['end'] + t10
# merge
merged_df = merge_intervals(intervals_df)

# =======================================================================================================
# Compute secondary parameters
# =======================================================================================================
df = create_subset(GFOC_data, GFOC_data.iloc[0]['time'], GFOC_data.iloc[-1]['time'], resample_rate='5min')
# time delta in seconds
dt_seconds = (df.index[1] - df.index[0]).total_seconds()

# daily means
df['median_decay_last_7D'] = df['orbital_decay'].rolling(window=int(7*24*60*(60/dt_seconds)), min_periods=1).median()
df['median_decay_last_14D'] = df['orbital_decay'].rolling(window=int(14*24*60*(60/dt_seconds)), min_periods=1).median()
df['median_decay_last_30D'] = df['orbital_decay'].rolling(window=int(30*24*60*(60/dt_seconds)), min_periods=1).median()

# polynomial coefficients for orbital period (minutes) as function of time (years since 2000)
p_list = [1.06605722e+02, -7.21846851e-09]
p = Polynomial(p_list)
# compute orbital period and phase
orbital_period = p(df.index.astype(np.int64) / 1e9) * 60 # seconds
# cumulative orbital cycles over time
cycles = np.cumsum(1 / orbital_period) * dt_seconds   # because your sampling is every 20s
# phase angle
df["orbital_period"] = orbital_period
df["phase"] = (2 * np.pi * cycles) % (2 * np.pi)
# add more harmonics to potentialy capture other periodicities; if not useful the lasso should zero them out
num_harmonics = 4
for n in range(1, num_harmonics + 1):
    df[f'sin_harmonic_{n}'] = np.sin(n * df["phase"])
    df[f'cos_harmonic_{n}'] = np.cos(n * df["phase"])

# STL
subsets = []
# mo flags
interval_starts = pd.to_datetime(merged_df['start']).dropna()
interval_ends = pd.to_datetime(merged_df['end']).dropna()
intervals = list(zip(interval_starts, interval_ends))

for start, end in intervals:
    subset = df.loc[start:end]
    
    # STL parameters
    mean_period_s = np.nanmedian(orbital_period)  # seconds per cycle
    period_samples = int(round(mean_period_s / dt_seconds))
    seasonal = 20 * period_samples + (20 * period_samples % 2 == 0) # Ensure odd
    low_pass_jump = seasonal_jump = int(0.15 * (period_samples + 1))
    trend_jump = int(0.15 * 1.5 * (period_samples + 1))
    # stl = STL(subset["orbital_decay"], period=period_samples, robust=True)
    stl = STL(
        subset["orbital_decay"],
        period=period_samples,
        seasonal=seasonal,
        seasonal_jump=seasonal_jump,
        trend_jump=trend_jump,
        low_pass_jump=low_pass_jump,
    )
    res = stl.fit()
    subset["osc_stl"] = res.seasonal
    subset["trend_stl"] = res.trend
    subset["resid_stl"] = res.resid
    # set variables
    subset["oscillation"] = subset["osc_stl"] + subset["resid_stl"]
    subset["trend"] = subset["trend_stl"]

    subsets.append(subset)

# create single dataframe of all subsets
subset_dfs = []
for subset in subsets:
    subset_dfs.append(subset)
df_all_subsets = pd.concat(subset_dfs)

# =======================================================================================================
# Setup modeling parameters
# =======================================================================================================
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
target = 'trend'
n = 12  # hours ahead
m = 15  # minutes steps
y_stepsize = int(m * (60 / dt_seconds)) # every 15 minutes

for subset in subsets:
    X_trend = make_lags(subset, lag_config_trend)

    y_trend = make_multistep_target(subset[[target]].copy(), steps=int(n * 60 * (60 / dt_seconds)), stepsize=y_stepsize)

    # Shifting has created indexes that don't match. Only keep times for
    na_mask = X_trend.isna().any(axis=1) | y_trend.isna().any(axis=1)
    X_trend = X_trend[~na_mask]
    y_trend = y_trend[~na_mask]

    # Rename columns to avoid MultiIndex issues
    X_trend.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in X_trend.columns]

    X_trend_list.append(X_trend)
    y_trend_list.append(y_trend)

# merge all subsets
no_test_subsets = 8

X_train = pd.concat(X_trend_list[:-no_test_subsets])
y_train = pd.concat(y_trend_list[:-no_test_subsets])
X_test = pd.concat(X_trend_list[-no_test_subsets:])
y_test = pd.concat(y_trend_list[-no_test_subsets:])

# log modeling parameters
logging.info("--Modeling Parameters--")
logging.info(f"lag_config_trend: {len(lag_config_trend)} different features")
for feature, cfg in lag_config_trend.items():
    n_lags = len(cfg.get("lags", []))
    logging.info(f'    "{feature}": {n_lags} lags,')
logging.info(f"Total features: {X_train.shape[1]}")
logging.info(f"Target: {target}")
logging.info(f"Forecast horizon: {n} hours ahead, stepsize: {m} minutes (={y_train.shape[1]} steps)")
logging.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
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
scaler_name = "multitask_lasso_trend_scaler_2"
scaler_path = find_repo_root() / Path(f"Analysis/modeling/saved_models/{scaler_name}.pkl")
joblib.dump(scaler, scaler_path)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    index=X_test.index,
    columns=X_test.columns
)

alphas = np.logspace(-2, 1, 40)
# alphas = np.logspace(-1.3, 0.7, 20)
# model_trend = LinearRegression()
# model_trend = Lasso(alpha=0.3)
model_trend = MultiTaskLassoCV(
    alphas=alphas, 
    cv=5,
    max_iter=5000,
    n_jobs=20
)

model_trend.fit(X_train_scaled, y_train)

# save model
model_name = "multitask_lasso_trend_model_2"
model_path = find_repo_root() / Path(f"Analysis/modeling/saved_models/{model_name}.pkl")
joblib.dump(model_trend, model_path)

y_train_pred = pd.DataFrame(model_trend.predict(X_train_scaled), index=X_train.index, columns=y_train.columns)
y_test_pred = pd.DataFrame(model_trend.predict(X_test_scaled), index=X_test.index, columns=y_test.columns)

# logging model info
log_model_description(model_trend)
logging.info(f"Model ID: {model_name}")
logging.info(f"Scaler ID: {scaler_name}")

# === End timing ===
end_time = datetime.now()
elapsed = end_time - start_time
logging.info(f"Script finished: {end_time}")
logging.info(f"Elapsed time: {elapsed}")