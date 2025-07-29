def plot_summary(
    data, Shock, Helio, RC,
    start_date, end_date,
    plot_config,
    tick_interval='daily', tick_step=5,
    FLAG=True,
    save_path=None,
    show_plot=True
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.dates as mdates

    # Filter data to the time range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    data = data[(data['time'] >= start) & (data['time'] <= end)]
    times = pd.to_datetime(data['time'])

    fig, axs = plt.subplots(len(plot_config), 1, figsize=(10, 4 * len(plot_config)), sharex=True)
    if len(plot_config) == 1:
        axs = [axs]

    # Helper for tick formatting
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
        ax.tick_params(axis='x', rotation=60)

    # Optional flags
    if FLAG:
        def parse_time(df, col):
            return pd.to_datetime(df[col], errors='coerce')

        def within_range(df, col):
            times = parse_time(df[col])
            return df[(times >= start) & (times <= end)]

        Shock_filtered = within_range(Shock, 'Time')
        Helio_filtered = Helio[
            (parse_time(Helio, 'mo_end_time') >= start) &
            (parse_time(Helio, 'icme_start_time') <= end)
        ]
        RC_filtered = RC[
            (parse_time(RC, 'ICME_End') >= start) &
            (parse_time(RC, 'Disturbance_time') <= end)
        ]

        Shock_flags = parse_time(Shock_filtered, 'Time')
        Helio_shock_flags = parse_time(Helio_filtered, 'icme_start_time')
        Helio_mo_start = parse_time(Helio_filtered, 'mo_start_time')
        Helio_mo_end = parse_time(Helio_filtered, 'mo_end_time')
        RC_shock_flags = parse_time(RC_filtered, 'Disturbance_time')
        RC_start = parse_time(RC_filtered, 'ICME_Start')
        RC_end = parse_time(RC_filtered, 'ICME_End')

        for ax in axs:
            for t in Shock_flags:
                ax.axvline(t, color='tab:gray', alpha=0.5, linestyle='-', linewidth=2, label='IP Shock')
            for t in Helio_shock_flags:
                ax.axvline(t, color='tab:purple', alpha=0.5, linestyle='--', linewidth=1.5, label='Helio Shock')
            for tstart, tend in zip(Helio_mo_start, Helio_mo_end):
                ax.axvline(tstart, color='tab:cyan', alpha=0.5, linestyle='-.', linewidth=1.0)
                ax.axvline(tend, color='tab:cyan', alpha=0.5, linestyle=':', linewidth=1.0)
                ax.axvspan(tstart, tend, color='tab:cyan', alpha=0.2, hatch='//', label='Helio MO')
            for t in RC_shock_flags:
                ax.axvline(t, color='tab:red', alpha=0.5, linestyle=':', linewidth=2.0, label='R&C Shock')
            for tstart, tend in zip(RC_start, RC_end):
                ax.axvline(tstart, color='tab:orange', alpha=0.5, linestyle='-.', linewidth=1.0)
                ax.axvline(tend, color='tab:orange', alpha=0.5, linestyle=':', linewidth=1.0)
                ax.axvspan(tstart, tend, color='tab:orange', alpha=0.2, hatch='-', label='R&C ICME')

        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0., 1.02, 1., .102),
                      loc='lower left', ncol=4, mode="expand", borderaxespad=0.)

    # Plot each axis from config
    for ax, cfg in zip(axs, plot_config):
        left = cfg['left']
        right = cfg['right']

        ax.plot(times, data[left['column']], color=left['color'])
        ax.set_ylabel(left['label'], color=left['color'])
        ax.tick_params(axis='y', labelcolor=left['color'])
        ax.grid()

        axr = ax.twinx()
        axr.plot(times, data[right['column']], color=right['color'])
        axr.set_ylabel(right['label'], color=right['color'])
        axr.tick_params(axis='y', labelcolor=right['color'])

    format_ticks(axs[-1], tick_interval, tick_step)

    fig.suptitle('GRACE-FO-1 decay rate and solar wind parameters (Levin)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

"""
# Example usage:
from plot_helpers import plot_summary

# Your data and flag tables must be loaded already (e.g., via pandas)
# Example config:
plot_config = [
    {
        'left': {'column': 'orbital_decay', 'color': 'k', 'label': 'Orbital decay rate [m/day]'},
        'right': {'column': 'ap (LASP)', 'color': 'y', 'label': 'ap geomagnetic index (nT)'}
    },
    {
        'left': {'column': 'Bz GSE', 'color': 'b', 'label': 'Bz (nT), GSE'},
        'right': {'column': 'Plasma beta', 'color': 'r', 'label': 'Plasma beta'}
    },
    {
        'left': {'column': 'F10.7 (LASP)', 'color': 'm', 'label': '10.7cm Solar Radio Flux (SFU)'},
        'right': {'column': 'Flow pressure (nPa)', 'color': 'c', 'label': 'Flow pressure (nPa)'}
    }
]

# Call the function
plot_summary(
    data=GFOC_data,
    Shock=Shock,
    Helio=Helio,
    RC=RC,
    start_date='2023-02-15 00:00:00',
    end_date='2023-04-07 00:00:00',
    plot_config=plot_config,
    tick_interval='daily',
    tick_step=5,
    FLAG=True,
    save_path='figures/plot_example.png',
    show_plot=True
)
"""