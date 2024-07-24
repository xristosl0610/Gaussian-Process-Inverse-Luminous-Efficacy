from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src import CM, FSIZE, TSIZE, TDIR, MAJOR, MINOR, STYLE


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


def plot_preds(dates: np.ndarray, y_actual: np.ndarray,
               y_pred: np.ndarray, y_std: np.ndarray,
               y_scaler: StandardScaler | MinMaxScaler,
               train_size: int, save_filepath: Path | None = None,
               plot_settings: dict = None) -> None:
    """
    Plots the predictions along with the actual measurements over time, showing the mean prediction, confidence interval, and training data region.

    Args:
        dates: An array of dates corresponding to the predictions and measurements.
        y_actual: The actual measurements.
        y_pred: The predicted values.
        y_std: The standard deviation of the predictions.
        y_scaler: The scaler used to transform the target variable.
        train_size: The index indicating the end of the training data.
        save_filepath: The file path where the plot should be saved, if provided (default is None).
        plot_settings: A dictionary containing the plot settings.

    Returns:
        None
    """
    if plot_settings is None:
        plot_settings = {}

    plt.figure(figsize=plot_settings.get("figsize", (16 * CM, 8 * CM)))
    y_pred_inv = y_scaler.inverse_transform(y_pred[:, np.newaxis]).squeeze()
    y_std_inv = y_scaler.inverse_transform(y_std[:, np.newaxis]).squeeze()

    plt.plot(dates, y_pred_inv, label=plot_settings.get("mean_label", "mean"))
    plt.fill_between(dates, y_pred_inv + 2 * y_std_inv, y_pred_inv - 2 * y_std_inv, alpha=0.2,
                     label=plot_settings.get("std_label", r'mean $\pm$ 2 std'))
    plt.axvspan(dates[0], dates[train_size], alpha=0.08, color='k')
    plt.axvline(dates[train_size], linestyle='--', c='k')
    plt.plot(dates, y_actual, plot_settings.get("actual_marker", 'rx'), markersize=2.,
             label=plot_settings.get("actual_label", 'measurements'))

    plt.ylabel(plot_settings.get("ylabel", 'Global Horizontal Irradiance [W/m$^2$]'))
    plt.xlim(dates[[0, -1]])
    plt.legend(fontsize=FSIZE, loc=plot_settings.get("legend_loc", 'upper center'),
               bbox_to_anchor=plot_settings.get("bbox_to_anchor", (0.5, 1.2)), ncols=plot_settings.get("ncols", 3))
    ax = plt.gca()
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(plot_settings.get("date_format", '%d-%m-%y')))
    ax.set_yticks(plot_settings.get("yticks", np.arange(-500, 1251, 250)))
    ax.set_yticklabels(plot_settings.get("yticklabels", np.arange(-500, 1251, 250)))
    plt.ylim(plot_settings.get("ylim", (-500, 1250)))

    plt.text(0.3, 0.95, plot_settings.get("training_text", "Training data points"), horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.text(0.86, 0.95, plot_settings.get("forecasting_text", "Forecasting"), horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes)

    if save_filepath:
        plt.savefig(save_filepath, bbox_inches='tight')

    plt.show()
