import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
# os.nice(19)

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import MultiTaskLasso
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging
import joblib

# =======================================================================================================
# Logging configuration
# =======================================================================================================
LOG_DIR = find_repo_root() / Path("Analysis/modeling/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "tscv_training.log"

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
train_start_time = '2023-01-01 00:00:00'
train_end_time = '2024-07-01 00:00:00'
test_start_time = '2024-07-24 00:00:00'
test_end_time = '2025-01-01 00:00:00'
target = 'trend'
# ---- forecast parameters ----
n = 12  # hours ahead
m = 15  # minutes steps
# ---- Train only on subset ----
Subset = True  # if True, train only on interesting subsets
if Subset:
    invert = False  # if True, train on the uninteresting times
    Subset_DIR = find_repo_root() / Path("Analysis/Subsets")
    Subset_name = 'subsets_Kp.csv'  # 'subsets_Kp.csv'  # 'subsets_eflag.csv'  # 'subsets_meanstd.csv' # 'subsets_merged.csv' # 'subsets_eflag_meanstd.csv'
    Subset_file = Subset_DIR / Path(Subset_name)
# ---- model parameters ----
model_type = 'MultiTaskLasso'
model_number = '4'  # just for naming purposes

# logging settings
logging.info("-- Basic Settings --")
logging.info(f"Sampling rate: {sampling_rate}")
logging.info(f"Training Time range: {train_start_time} to {train_end_time}")
logging.info(f"Testing Time range: {test_start_time} to {test_end_time}")
logging.info(f"Subset training: {Subset}")
if Subset:
    logging.info(f"Invert Subset: {invert}")
    logging.info(f"Subset name: {Subset_name}")

# =======================================================================================================
# Load Data
# =======================================================================================================

DATA_DIR = find_repo_root() / Path("Dataset/modeling/")
DATA_file = DATA_DIR / Path(f"GFOC_modeling_{sampling_rate}.parquet")
GFOC_data = pd.read_parquet(DATA_file, columns=columns_needed)
# print(f"✅ Successfully loaded data with shape: {GFOC_data.shape}")
# # column names
# print("Column names in the dataset:")
# print(GFOC_data.columns.tolist())


# filter by time range
train_data = GFOC_data.loc[train_start_time:train_end_time]
test_data = GFOC_data.loc[test_start_time:test_end_time]
# print(f"✅ Successfully filtered train data with shape: {train_data.shape} and test data with shape: {test_data.shape}")

X_file = DATA_DIR / Path(f"X_{sampling_rate}.parquet")
X = pd.read_parquet(X_file)
X_train = X.loc[train_start_time:train_end_time]
X_test = X.loc[test_start_time:test_end_time]
# print(f"✅ Successfully loaded features X_train with shape: {X_train.shape} and X_test with shape: {X_test.shape}")
y_file = DATA_DIR / Path(f"y_{target}_{sampling_rate}.parquet")
y = pd.read_parquet(y_file)
y_train = y.loc[train_start_time:train_end_time]
y_test = y.loc[test_start_time:test_end_time]
# print(f"✅ Successfully loaded target y_train with shape: {y_train.shape} and y_test with shape: {y_test.shape}")

# =======================================================================================================
# Select subset if needed
# =======================================================================================================
if Subset:
    intervals_df = pd.read_csv(Subset_file)
    df = GFOC_data.copy()
    mask = pd.Series(False, index=df.index)

    for _, row in intervals_df.iterrows():
        mask |= (df.index >= row["start"]) & (df.index <= row["end"])
    
    if invert:
        mask = ~mask

    # Train only on these times
    active_times = GFOC_data[mask].index.intersection(X_train.index)
    logging.info(f"Training on {len(active_times)} samples of total {X_train.shape[0]} samples")
    X_train = X_train.loc[active_times]
    y_train = y_train.loc[active_times]
    # Test only on these times
    active_test = GFOC_data[mask].index.intersection(X_test.index)
    logging.info(f"Testing on {len(active_test)} samples of total {X_test.shape[0]} samples")
    # X_test = X_test.loc[active_test]
    # y_test = y_test.loc[active_test]
else:
    # Train on all times
    logging.info(f"Using all training samples: {X_train.shape[0]}")
    logging.info(f"Using all testing samples: {X_test.shape[0]}")
    active_test = None

# =======================================================================================================
# Logging modeling parameters
# =======================================================================================================
features = X.columns.str.extract(r'^(.+?)_lag_')[0]
summary = (
    features
    .value_counts()
    .rename_axis('feature')
    .reset_index(name='n_columns')
)
# log modeling parameters
logging.info("--Modeling Parameters--")
logging.info(f"lag_config_trend: {len(summary)} different features")
for feature, k in summary.itertuples(index=False):
    logging.info(f'    "{feature}": {k} lags,')
logging.info(f"Total features: {X_train.shape[1]}")
logging.info(f"Target: {target}")
logging.info(f"Forecast horizon: {n} hours ahead, stepsize: {m} minutes (={y_train.shape[1]} steps)")

# =======================================================================================================
# Training Function
# =======================================================================================================
def tscv_evaluate(X, Y, X_test, Y_test, alpha, tscv, pipeline_name, active_test=None):
    training_start_time = datetime.now()
    train_scores = []
    val_scores = []

    for i, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        logging.info(f"Start with {i}th training")
        start_time = datetime.now()

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]
        logging.info(f"X_train time: {X_train.index[0]} to {X_train.index[-1]}")
        logging.info(f"X_val time: {X_val.index[0]} to {X_val.index[-1]}")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", MultiTaskLasso(
                alpha=alpha,
                max_iter=20000,
                warm_start=True,
                selection="random"
            ))
        ])

        model.fit(X_train, Y_train)

        train_scores.append(
            r2_score(Y_train, model.predict(X_train))
        )
        val_scores.append(
            r2_score(Y_val, model.predict(X_val))
        )

        logging.info(f"Finished after {datetime.now() - start_time}")

    # Refit once on all training data
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", MultiTaskLasso(
            alpha=alpha,
            max_iter=20000,
            warm_start=True,
            selection="random"
        ))
    ])
    logging.info("Start with final training")
    start_time = datetime.now()
    final_model.fit(X, Y)
    # save pipeline
    pipeline_path = find_repo_root() / Path(f"Analysis/modeling/tscv_models/{pipeline_name}.pkl")
    joblib.dump(final_model, pipeline_path)

    logging.info(f"Finished after {datetime.now() - start_time}")
    logging.info(f"Total training time: {datetime.now() - training_start_time}")

    # logging pipeline info
    logging.info("-- Final Model Description --")
    logging.info(f"Model class: {final_model.__class__.__name__}")

    try:
        params = final_model.get_params()
        for k, v in params.items():
            logging.info(f"  {k}: {v}")
    except Exception:
        logging.info("  No parameters available")
    logging.info(f"Model ID: {pipeline_name}")

    if active_test is not None:
        active_mask = X_test.index.isin(active_test)
        test_score_all = r2_score(Y_test, final_model.predict(X_test))
        test_score_sub = r2_score(Y_test.loc[active_mask], final_model.predict(X_test.loc[active_mask]))
        test_score_inv = r2_score(Y_test.loc[~active_mask], final_model.predict(X_test.loc[~active_mask]))
        return {
            "train_mean": np.mean(train_scores),
            "val_mean": np.mean(val_scores),
            "test_all": test_score_all,
            "test_sub": test_score_sub,
            "test_inv": test_score_inv
        }
    else:
        test_score = r2_score(Y_test, final_model.predict(X_test))

        return {
            "train_mean": np.mean(train_scores),
            "val_mean": np.mean(val_scores),
            "test": test_score
        }

# =======================================================================================================
# Training
# =======================================================================================================
alpha_weak = 0.1    # near Model 3 optimum
alpha_strong = 0.5 # strong regularization

tscv_weak = TimeSeriesSplit(
    n_splits=5,
    test_size=None  # expanding window
)

tscv_strong = TimeSeriesSplit(
    n_splits=5,
    test_size=None  # expanding window
)

logging.info("== Training with Time Series Cross-Validation ==")
logging.info(f"-- Weak Regularization α={alpha_weak}, TSCV splits={tscv_weak.get_n_splits()} --")
alpha_str = str(alpha_weak).replace('.', 'p')
results_weak = tscv_evaluate(
    X_train, y_train,
    X_test, y_test,
    alpha=alpha_weak,
    tscv=tscv_weak,
    pipeline_name = f"tscv_{model_type}_{target}_pipeline_{alpha_str}_{model_number}",
    active_test=active_test
)

logging.info(f"-- Strong Regularization α={alpha_strong}, TSCV splits={tscv_strong.get_n_splits()} --")
alpha_str = str(alpha_strong).replace('.', 'p')
results_strong = tscv_evaluate(
    X_train, y_train,
    X_test, y_test,
    alpha=alpha_strong,
    tscv=tscv_strong,
    pipeline_name = f"tscv_{model_type}_{target}_pipeline_{alpha_str}_{model_number}",
    active_test=active_test
)

# log results
logging.info("-- Training Results --")
if Subset:
    logging.info(f"Weak Reg (α={alpha_weak}):\n   Train R²: {results_weak['train_mean']:.4f},\n   Val R²: {results_weak['val_mean']:.4f},\n   Test R² (all): {results_weak['test_all']:.4f},\n   Test R² (subset): {results_weak['test_sub']:.4f},\n   Test R² (inverse): {results_weak['test_inv']:.4f}")
    logging.info(f"Strong Reg (α={alpha_strong}):\n   Train R²: {results_strong['train_mean']:.4f},\n   Val R²: {results_strong['val_mean']:.4f},\n   Test R² (all): {results_strong['test_all']:.4f},\n   Test R² (subset): {results_strong['test_sub']:.4f},\n   Test R² (inverse): {results_strong['test_inv']:.4f}")
else:
    logging.info(f"Weak Reg (α={alpha_weak}):\n   Train R²: {results_weak['train_mean']:.4f},\n   Val R²: {results_weak['val_mean']:.4f},\n   Test R²: {results_weak['test']:.4f}")
    logging.info(f"Strong Reg (α={alpha_strong}):\n   Train R²: {results_strong['train_mean']:.4f},\n   Val R²: {results_strong['val_mean']:.4f},\n   Test R²: {results_strong['test']:.4f}")


# === End timing ===
log_end_time = datetime.now()
elapsed = log_end_time - log_start_time
logging.info(f"Script finished: {log_end_time}")
logging.info(f"Elapsed time: {elapsed}")