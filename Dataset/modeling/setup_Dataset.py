import sys
from pathlib import Path
# Automatically add the project root (MT folder) to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# flake8: noqa: E402
import numpy as np
from numpy.polynomial import Polynomial
from statsmodels.tsa.seasonal import STL
from utils.data_loader import load_parquet, find_repo_root
from utils.plot_loader import create_subset

# =======================================================================================================
# Load Data
# =======================================================================================================
# Load only specific columns needed for analysis
columns_needed = ['time', 'orbital_decay', '|avg B|', 'F10.7 (LASP)', 'Bz GSE', 'Flow Speed (km/s', 'Temperature (K)', 'Kp (LASP)']

GFOC_data = load_parquet(columns=columns_needed)
print(GFOC_data.head())

# =======================================================================================================
# Compute secondary parameters
# =======================================================================================================
df = create_subset(GFOC_data, GFOC_data.iloc[0]['time'], GFOC_data.iloc[-1]['time'], resample_rate='20S')
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

# =======================================================================================================¨
# STL decomposition to extract trend and seasonal components
# =======================================================================================================
# assume regular sampling and compute period in samples:
mean_period_s = np.nanmedian(orbital_period)  # seconds per cycle
period_samples = int(round(mean_period_s / dt_seconds))
seasonal = 20 * period_samples + (20 * period_samples % 2 == 0) # Ensure odd
low_pass_jump = seasonal_jump = int(0.15 * (period_samples + 1))
trend_jump = int(0.15 * 1.5 * (period_samples + 1))

# stl = STL(subset["orbital_decay"], period=period_samples, robust=True)
stl = STL(
    df["orbital_decay"],
    period=period_samples,
    seasonal=seasonal,
    seasonal_jump=seasonal_jump,
    trend_jump=trend_jump,
    low_pass_jump=low_pass_jump,
)
res = stl.fit()

# set variables
df["oscillation"] = res.seasonal + res.resid
df["trend"] = res.trend

# =======================================================================================================
# Save updated dataset
# =======================================================================================================
resamp = '20s'
output_path = Path(find_repo_root()) / 'Dataset' / 'modeling' / f'GFOC_modeling_{resamp}.parquet'
df.to_parquet(output_path)