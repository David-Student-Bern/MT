# utils/plot_loader.py

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_flags

def format_ticks(ax, tick_interval, tick_step):
    if tick_interval == 'monthly':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))                         
    elif tick_interval == 'daily':
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick_step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif tick_interval == 'hourly':
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=tick_step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=0)

def create_subset(Data, start_date, end_date):
    subset = Data.copy()
    subset['time'] = pd.to_datetime(subset['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    mask = (subset['time'] >= pd.to_datetime(start_date)) & (subset['time'] <= pd.to_datetime(end_date))
    subset = subset.loc[mask]
    return subset

def plot_orbital_decay(
    Data,
    Data2=None,
    start_date=None,
    end_date=None,
    tick_interval='daily',
    tick_step=15,
    FLAGS=False,
    Lombscargle=False,
    style='default',
    figsize=(12, 6),
    show=True
):
    """
    Plot orbital decay with optional flags.
    Returns (fig, ax).

    Parameters
    ----------
    Data : pd.DataFrame
        New data (must contain 'time' and 'orbital_decay').
    Data2 : pd.DataFrame or None
        Old data (optional).
    start_date, end_date : str or pd.Timestamp
        Date range to plot. If None, inferred from Data.
    tick_interval : {'monthly','daily','hourly'}
    tick_step : int
    FLAGS : bool
        If True, plot flags from Shock/Helio/RC.
    Lombscargle : bool
        If True, perform Lomb-Scargle analysis.
    style : str
        Matplotlib style to use.
    figsize : tuple
    show : bool
        Whether to call plt.show().
    """
    plt.style.use(style)

    # infer date range if not provided
    if start_date is None:
        start_date = pd.to_datetime(Data['time'].min())
    if end_date is None:
        end_date = pd.to_datetime(Data['time'].max())

    # ensure times are datetimes and filter
    subset = create_subset(Data, start_date, end_date)
    times = subset['time']

    # old data (optional)
    if Data2 is not None:
        subset_old = create_subset(Data2, start_date, end_date)
        old_time = subset_old['time']
    else:
        subset_old = pd.DataFrame(columns=subset.columns)
        old_time = pd.Series(dtype='datetime64[ns]')

    # Flags handling (only if requested)
    if FLAGS:
        Shock, Helio, RC = load_flags()

        if Shock is None or Helio is None or RC is None:
            raise ValueError("FLAGS=True but Shock, Helio or RC is None. Pass the flag DataFrames or set FLAGS=False.")

        Shock_subset = Shock[pd.to_datetime(Shock['Time'], errors='coerce') >= pd.to_datetime(start_date)]
        Shock_subset = Shock_subset[pd.to_datetime(Shock_subset['Time'], errors='coerce') <= pd.to_datetime(end_date)]
        Helio_subset = Helio[pd.to_datetime(Helio['mo_end_time'], errors='coerce') >= pd.to_datetime(start_date)]
        Helio_subset = Helio_subset[pd.to_datetime(Helio_subset['icme_start_time'], errors='coerce') <= pd.to_datetime(end_date)]
        RC_subset = RC[pd.to_datetime(RC['ICME_End'], errors='coerce') >= pd.to_datetime(start_date)]
        RC_subset = RC_subset[pd.to_datetime(RC_subset['Disturbance_time'], errors='coerce') <= pd.to_datetime(end_date)]

        Shock_flag = pd.to_datetime(Shock_subset['Time'], errors='coerce')
        Helios_shock_flag = pd.to_datetime(Helio_subset['icme_start_time'], errors='coerce')
        Helios_mostart_flag = pd.to_datetime(Helio_subset['mo_start_time'], errors='coerce')
        Helios_moend_flag = pd.to_datetime(Helio_subset['mo_end_time'], errors='coerce')
        RC_shock_flag = pd.to_datetime(RC_subset['Disturbance_time'], errors='coerce')
        RC_start_flag = pd.to_datetime(RC_subset['ICME_Start'], errors='coerce')
        RC_end_flag = pd.to_datetime(RC_subset['ICME_End'], errors='coerce')

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if FLAGS:
        for t in Shock_flag.dropna():
            ax.axvline(t, color='tab:gray', alpha=0.5, linestyle='-', linewidth=2, label='IP Shock')
        for t in Helios_shock_flag.dropna():
            ax.axvline(t, color='tab:purple', alpha=0.5, linestyle='--', linewidth=1.5, label='Helio Shock')
        for tstart, tend in zip(Helios_mostart_flag.dropna(), Helios_moend_flag.dropna()):
            ax.axvline(tstart, color='tab:cyan', alpha=0.5, linestyle='-.', linewidth=1.0, label='Helio MO Start')
            ax.axvline(tend, color='tab:cyan', alpha=0.5, linestyle=':', linewidth=1.0, label='Helio MO End')
            ax.axvspan(tstart, tend, color='tab:cyan', alpha=0.2, hatch='//', label='Helio MO')
        for t in RC_shock_flag.dropna():
            ax.axvline(t, color='tab:red', alpha=0.5, linestyle=':', linewidth=2.0, label='R&C Shock')
        for tstart, tend in zip(RC_start_flag.dropna(), RC_end_flag.dropna()):
            ax.axvline(tstart, color='tab:orange', alpha=0.5, linestyle='-.', linewidth=1.0, label='R&C ICME Start')
            ax.axvline(tend, color='tab:orange', alpha=0.5, linestyle=':', linewidth=1.0, label='R&C ICME End')
            ax.axvspan(tstart, tend, color='tab:orange', alpha=0.2, hatch='-', label='R&C ICME')

    if not subset_old.empty:
        ax.plot(times, subset['orbital_decay'], color='tab:blue', alpha=0.7, label='New Data')
        ax.plot(old_time, subset_old['orbital_decay'], color='tab:red', alpha=0.3, label='Old Data')
    else:
        ax.plot(times, subset['orbital_decay'], color='tab:blue', label='Orbital Decay')

    format_ticks(ax, tick_interval, tick_step)
    ax.set_xlabel('Time')
    ax.set_ylabel('Orbital Decay')
    start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_str   = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    fig.suptitle(f'Orbital Decay Over Time\n{start_str} to {end_str}', fontsize=16)
    ax.grid(True)

    if FLAGS:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                  ncols=4, mode="expand", borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 1, 1.02])
    else:
        ax.legend()
        plt.tight_layout()

    if show:
        plt.show()
    
    if Lombscargle:
        from astropy.timeseries import LombScargle

        # Convert time to numerical values (in days)
        times_num = (times - times.iloc[0]).dt.total_seconds() / (24 * 3600)  # days since start

        y = subset['orbital_decay'].values

        # Define frequency grid (cycles per day)
        min_period = 0.1   # days
        max_period = 60  # days
        frequency = np.linspace(1/max_period, 1/min_period, 100000)

        # Compute Lomb-Scargle periodogram
        ls = LombScargle(times_num, y)
        power = ls.power(frequency)

        # Find the period with the highest power
        best_frequency = frequency[np.argmax(power)]
        best_period = 1 / best_frequency

        # Plot the periodogram
        plt.figure(figsize=(10, 5))
        plt.plot(1/frequency, power)
        plt.xlabel('Period (days)')
        plt.ylabel('Lomb-Scargle Power')
        plt.title(f'Lomb-Scargle Periodogram of Orbital Decay\n{start_str} to {end_str}')
        plt.xscale('log')
        plt.grid(True)

        # Add vertical lines for solar rotation periods
        plt.axvline(26.24, color='orange', linestyle='--', linewidth=2, label='Solar synodic rotation (Equator, 26.24d)')
        plt.axvline(35, color='purple', linestyle='--', linewidth=2, label='Solar rotation (Near Pole, 35d)')

        plt.legend()
        plt.show()

        print(f"Strongest period: {best_period:.2f} days")

def plot_lombscargle_periodogram(
    Data,
    start_date=None,
    end_date=None,
    min_period=0.1,   # days
    max_period=60,  # days
    vlines=True,
    style='default',
    figsize=(10, 5),
    show=True
):
    """
    Plot Lomb-Scargle periodogram of orbital decay.
    Returns (fig, ax).

    Parameters
    ----------
    Data : pd.DataFrame
        Data (must contain 'time' and 'orbital_decay').
    start_date, end_date : str or pd.Timestamp
        Date range to plot. If None, inferred from Data.
    tick_interval : {'monthly','daily','hourly'}
    tick_step : int
    style : str
        Matplotlib style to use.
    figsize : tuple
    show : bool
        Whether to call plt.show().
    """
    plt.style.use(style)

    # infer date range if not provided
    if start_date is None:
        start_date = pd.to_datetime(Data['time'].min())
    if end_date is None:
        end_date = pd.to_datetime(Data['time'].max())

    # ensure times are datetimes and filter
    subset = create_subset(Data, start_date, end_date)

    from astropy.timeseries import LombScargle

    # Convert time to numerical values (in days)
    times = pd.to_datetime(subset['time'], format='%Y-%m-%d %H:%M:%S')
    times_num = (times - times.iloc[0]).dt.total_seconds() / (24 * 3600)  # days since start

    y = subset['orbital_decay'].values

    # Define frequency grid (cycles per day)
    frequency = np.linspace(1/max_period, 1/min_period, 100000)

    # Compute Lomb-Scargle periodogram
    ls = LombScargle(times_num, y)
    power = ls.power(frequency)

    # Find the period with the highest power
    best_frequency = frequency[np.argmax(power)]
    best_period = 1 / best_frequency

    # Plot the periodogram
    plt.figure(figsize=(10, 5))
    plt.plot(1/frequency, power)
    plt.xlabel('Period (days)')
    plt.ylabel('Lomb-Scargle Power')
    start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_str   = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    plt.title(f'Lomb-Scargle Periodogram of Orbital Decay\n{start_str} to {end_str}')
    plt.xscale('log')
    plt.grid(True)

    # Add vertical lines for solar rotation periods
    if vlines:
        plt.axvline(26.24, color='orange', linestyle='--', linewidth=2, label='Solar synodic rotation (Equator, 26.24d)')
        plt.axvline(35, color='purple', linestyle='--', linewidth=2, label='Solar rotation (Near Pole, 35d)')

    plt.legend()
    plt.show()

    print(f"Strongest period: {best_period:.2f} days")