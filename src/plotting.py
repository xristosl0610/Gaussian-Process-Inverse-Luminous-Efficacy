from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

from src import (CM, FSIZE, TSIZE, TDIR, MAJOR, MINOR, STYLE,
                 DATADIR, CONFIGDIR, OUTPUTDIR)
from src.config_dataclass import create_config
from src.preprocess import read_data, read_json, preprocess_df


plt.style.use(STYLE)
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams['font.size'] = FSIZE
plt.rcParams['legend.fontsize'] = TSIZE
plt.rcParams['xtick.direction'] = TDIR
plt.rcParams['ytick.direction'] = TDIR
plt.rcParams['xtick.major.size'] = MAJOR
plt.rcParams['xtick.minor.size'] = MINOR
plt.rcParams['ytick.major.size'] = MAJOR
plt.rcParams['ytick.minor.size'] = MINOR

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def plot_preds(dates: np.ndarray, y_true: np.ndarray,
               y_pred: np.ndarray, y_std: np.ndarray,
               train_size: int, save_filepath: Path | None = None,
               plot_settings: dict = None, month_scale: bool = False,
               cumulative: bool = False) -> None:
    """
    Plots the predictions along with the actual measurements over time, showing the mean prediction, confidence interval, and training data region.

    Args:
        dates: An array of dates corresponding to the predictions and measurements.
        y_true: The true measurements.
        y_pred: The rescaled predicted values.
        y_std: The rescaled standard deviation of the predictions.
        train_size: The index indicating the end of the training data.
        save_filepath: The file path where the plot should be saved, if provided (default is None).
        plot_settings: A dictionary containing the plot settings.
        month_scale: A flag to change minor plotting details in case of testing over a month, instead of a day. Defaults to False.
        cumulative: A flag to plot the cumulative predictions if set to True. Defaults to False.

    Returns:
        None
    """
    if plot_settings is None:
        plot_settings = {}

    figsize = plot_settings.get("figsize", (16 * CM, 8 * CM))
    legend_loc = plot_settings.get("legend_loc", 'upper center')
    bbox_to_anchor = plot_settings.get("bbox_to_anchor", (0.5, 1.2))
    ncols = plot_settings.get("ncols", 3)
    date_format = plot_settings.get("date_format", '%H:%M')
    tick_interval = cumulative * 3750 + 250

    if len(plot_settings.get('ylabels')) == 1:
        y_pred, y_std = y_pred[:, np.newaxis], y_std[:, np.newaxis]

    if cumulative:
        y_true, y_pred, y_std = np.cumsum(y_true, axis=0), np.cumsum(y_pred, axis=0), np.sqrt(np.cumsum(y_std ** 2, axis=0))

    for col, (var, ylabel) in enumerate(plot_settings.get('ylabels').items()):
        plt.figure(figsize=figsize)

        plt.plot(dates, y_pred[:, col], label=plot_settings.get(f"mean_label_{col}", "mean"))
        plt.fill_between(dates, y_pred[:, col] + 2 * y_std[:, col], y_pred[:, col] - 2 * y_std[:, col],
                         alpha=0.2, label=plot_settings.get(f"std_label_{col}", r'mean $\pm$ 2 std'))
        plt.plot(dates, y_true[:, col], plot_settings.get(f"actual_marker_{col}", 'rx'), markersize=2.,
                 label=plot_settings.get(f"actual_label_{col}", 'measurements'))

        plt.ylabel(ylabel)
        plt.xlim(dates[[0, -1]])
        plt.legend(fontsize=FSIZE, loc=legend_loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols)

        ax = plt.gca()
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

        y_min = min(np.min(y_true[:, col]), np.min(y_pred[:, col]), np.min(y_pred[:, col] - 2 * y_std[:, col]))
        y_max = max(np.max(y_true[:, col]), np.max(y_pred[:, col]), np.max(y_pred[:, col] + 2 * y_std[:, col]))
        y_max_adjusted = int(y_max // tick_interval) * tick_interval + tick_interval
        yticks = np.arange(int(y_min // tick_interval) * tick_interval, y_max_adjusted + tick_interval, tick_interval)
        ylim = (int(y_min // tick_interval) * tick_interval, y_max_adjusted)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        plt.ylim(ylim)

        if not month_scale:
            plt.axvspan(dates[0], dates[train_size], alpha=0.08, color='k')
            plt.axvline(dates[train_size], linestyle='--', c='k')

            training_text = plot_settings.get("training_text", "Training data points")
            forecasting_text = plot_settings.get("forecasting_text", "Forecasting")

            plt.text(0.3, 0.95, training_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.text(0.86, 0.95, forecasting_text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        if save_filepath:
            plt.savefig(save_filepath.with_name(f"{save_filepath.stem}_{var}{save_filepath.suffix}"), bbox_inches='tight')

        plt.show()


def plot_hexbin_w_marginals(df: pd.DataFrame, x_col: dict[str, str], y_col: dict[str, str],
                            save_filepath: Path | None = None) -> None:
    """
    Plot a hexbin plot with marginal histograms.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be plotted.
        x_col (dict[str, str]): A dictionary with 'col' and 'label' keys representing the x-axis column name and label.
        y_col (dict[str, str]): A dictionary with 'col' and 'label' keys representing the y-axis column name and label.
        save_filepath (Path, optional): The file path where the plot will be saved. Defaults to None.

    Returns:
        None
    """
    plt.figure(figsize=(16 * CM, 8 * CM))
    g = sns.jointplot(
        data=df,
        x=x_col['col'],
        y=y_col['col'],
        kind='hex',
        height=8,
        marginal_kws=dict(bins=50, fill=True)
    )

    g.set_axis_labels(x_col['label'], y_col['label'], fontsize=FSIZE)
    g.fig.tight_layout()

    g.ax_joint.grid(True, linestyle='--', alpha=0.5)

    if save_filepath:
        plt.savefig(save_filepath, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    config, config_dict = create_config(CONFIGDIR.joinpath('config.toml'),
                                        CONFIGDIR.joinpath('config_overwrite.toml'))

    test_df = read_data(DATADIR.joinpath(config.datafiles.source))
    var_description = read_json(DATADIR.joinpath(config.datafiles.col_desc))
    data_cols = ('GHE', 'GHI')
    x_label = {'col': data_cols[0], 'label': var_description[data_cols[0]]}
    y_label = {'col': data_cols[1], 'label': var_description[data_cols[1]]}
    filt_df = preprocess_df(test_df, config)

    plot_hexbin_w_marginals(filt_df, x_label, y_label,
                            OUTPUTDIR.joinpath('jointplot.png'))

