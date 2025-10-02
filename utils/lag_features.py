# utils/lag_features.py

import pandas as pd
import numpy as np

class LagFeatures:
    def __init__(self, df, lag_minutes=5, max_lag_hours=48, freq="30s", return_lags_only=False):
        """
        df              : pandas DataFrame with one or more feature columns
        lag_minutes     : spacing between lags (default = 5 minutes)
        max_lag_hours   : maximum history in hours (default = 48 hours)
        freq            : frequency of the data (default = '30s')
        return_lags_only: if True, only lag columns are returned in self.df
        """
        self.df = df.copy()
        self.lag_minutes = lag_minutes
        self.max_lag_hours = max_lag_hours
        self.freq = pd.Timedelta(freq)
        self.return_lags_only = return_lags_only

        self.create_lags()

    def create_lags(self):
        step = int(pd.Timedelta(minutes=self.lag_minutes) / self.freq)
        max_lag = int((self.max_lag_hours * 60) / self.lag_minutes)

        lag_dict = {}
        for col in self.df.columns:
            for lag in range(1, max_lag + 1):
                lag_col_name = f'{col}_Lag{lag}'
                lag_dict[lag_col_name] = self.df[col].shift(lag * step)

        lag_df = pd.DataFrame(lag_dict, index=self.df.index)

        if self.return_lags_only:
            self.df = lag_df
        else:
            self.df = pd.concat([self.df, lag_df], axis=1)

        # store lag info for easy plotting
        self.lag_minutes_list = [lag * self.lag_minutes for lag in range(1, max_lag + 1)]


def fast_lag_correlation(df, lag_cols, target):
    X = df[lag_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    # remove rows with NaN (first few lags)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    # center
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean

    # numerator = covariance
    cov = np.dot(yc, Xc) / (len(y) - 1)

    # denominator = std_x * std_y
    corr = cov / (Xc.std(axis=0, ddof=1) * yc.std(ddof=1))

    return corr