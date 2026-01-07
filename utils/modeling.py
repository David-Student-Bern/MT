# utils/modeling.py

from warnings import simplefilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("default")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds), freq='60S')
        preds.plot(ax=ax, alpha=0.5)
    return ax

def plot_results(y_train, y_fit, y_test, y_pred, m=10, n=12):
    # skip each m steps for more efficient plotting
    y_train = y_train[::m]
    y_fit = y_fit[::m]
    y_test = y_test[::m]
    y_pred = y_pred[::m]


    palette = dict(palette='husl', n_colors=64)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    ax1 = y_train.iloc[:,[0]].plot(**plot_params, ax=ax1)
    ax1 = plot_multistep(y_fit, every=60, ax=ax1, palette_kwargs=palette)
    _ = ax1.legend(['Orbital Decay (train)', 'Forecast'])
    ax1.grid(True)

    # plot complete forecast horizon
    temp = y_test.iloc[:,[-1]].copy()
    temp.index = temp.index + pd.Timedelta(hours=n)  #! may need to manualy need to set timedelta here !!!!!!
    temp2 = y_test.iloc[:,[0]].copy()
    temp.columns = temp2.columns
    y_temp = pd.concat([temp2, temp])
    ax2 = y_temp.plot(**plot_params, ax=ax2)
    ax2 = plot_multistep(y_pred, every=60, ax=ax2, palette_kwargs=palette)
    _ = ax2.legend(['Orbital Decay (test)', 'Forecast'])
    # ax2.vlines('2024-05-10 16:37:00', ymin = 0, ymax = 140, color='red', linestyle='--')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def statistics(y_train, y_fit, y_test, y_pred):
    train_rmse = root_mean_squared_error(y_train, y_fit)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    print((f"Train RMSE: {train_rmse:.3}\n" f"Test RMSE: {test_rmse:.3}"))
    train_r2 = r2_score(y_train, y_fit)
    test_r2 = r2_score(y_test, y_pred)
    print((f"Train R2: {train_r2:.3}\n" f"Test R2: {test_r2:.3}"))

    train_rmse_list = []
    test_rmse_list = []
    train_r2_list = []
    test_r2_list = []

    for i in range(len(y_train.columns)):
        train_rmse_list.append(root_mean_squared_error(y_train.iloc[:, i], y_fit.iloc[:, i]))
        test_rmse_list.append(root_mean_squared_error(y_test.iloc[:, i], y_pred.iloc[:, i]))
        train_r2_list.append(r2_score(y_train.iloc[:, i], y_fit.iloc[:, i]))
        test_r2_list.append(r2_score(y_test.iloc[:, i], y_pred.iloc[:, i]))
    fix, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    xtrain = [int(col.split('_')[-1]) for col in y_train.iloc[0].index]
    xtest = [int(col.split('_')[-1]) for col in y_test.iloc[0].index]
    ax[0].plot(xtrain,train_rmse_list, label='Train RMSE')
    ax[0].plot(xtest,test_rmse_list, label='Test RMSE')
    ax[0].set_ylabel('RMSE')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(xtrain, train_r2_list, label='Train R2')
    ax[1].plot(xtest, test_r2_list, label='Test R2')
    ax[1].set_ylabel('R2 Score')
    ax[1].set_xlabel('Forecast Step (20s intervals)')
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()

def make_lags(df, lag_config, lead_time=1):
    out = {}

    for col, params in lag_config.items():
        n_lags = params.get('lags', 1)

        for i in n_lags:
            out[f"{col}_lag_{i}"] = df[col].shift(i)

    return pd.DataFrame(out)


def make_multistep_target(ts, steps, stepsize=1):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-(i+1) * stepsize).iloc[:, 0]
         for i in range(steps//stepsize)},
        axis=1)

def aggregate_multistep_predictions(y_pred, dt_seconds, error_kind='std'):
    """
    Convert y_pred (shape: origins x horizons, columns like 'y_step_1', 'y_step_46')
    to an aggregated DataFrame indexed by absolute forecast time with columns:
      - mean, std, sem, count, min, max

    Parameters
    ----------
    y_pred : pd.DataFrame
        index = origin timestamps, columns = 'y_step_{k}' for k=1..m
    dt_seconds : int or float
        seconds per step
    error_kind : 'std' or 'sem'
        which error measure to present (std = dispersion, sem = std / sqrt(n))

    Returns
    -------
    agg : pd.DataFrame
        index = absolute forecast timestamps; columns = ['mean','std','sem','count']
    """
    pieces = []
    for col in y_pred.columns:
        # extract integer step from column name; assumes 'y_step_{k}'
        try:
            k = int(col.split('_')[-1])
        except Exception:
            raise ValueError(f"Could not parse step number from column '{col}'")
        # shift the index forward by k steps
        shifted = y_pred[col].copy()
        shifted.index = shifted.index + pd.to_timedelta(k * dt_seconds, unit='s')
        # keep name to help debugging, but all columns will be concatenated
        shifted.name = f'{col}'
        pieces.append(shifted)

    # wide table: index = absolute forecast timestamps, columns = all predictions that land there
    wide = pd.concat(pieces, axis=1)

    # optionally drop completely-empty times
    wide = wide.dropna(how='all')

    # compute aggregates over available predictions at each absolute time
    mean = wide.mean(axis=1, skipna=True)
    std  = wide.std(axis=1, ddof=0, skipna=True)   # population std (ddof=0) — change if needed
    count = wide.count(axis=1)
    sem = std / np.sqrt(count.replace(0, np.nan))
    min_ = wide.min(axis=1, skipna=True)
    max_ = wide.max(axis=1, skipna=True)
    horizon0 = wide.bfill(axis=1).iloc[:, 0]

    agg = pd.DataFrame({'mean': mean, 'std': std, 'sem': sem, 'count': count, 'min': min_, 'max': max_, 'horizon0': horizon0})

    # choose error column to return in the same df for easy plotting
    agg['error'] = agg['std'] if error_kind == 'std' else agg['sem']

    return agg