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

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


def plot_preds(dates: np.ndarray, y: np.ndarray,
               y_pred: np.ndarray, y_std: np.ndarray,
               y_scaler: StandardScaler | MinMaxScaler,
               train_shape: int, save_filepath: Path | None = None) -> None:
    """
    Plots the predictions along with the actual measurements over time, showing the mean prediction, confidence interval, and training data region.

    Args:
        dates: An array of dates corresponding to the predictions and measurements.
        y: The actual measurements.
        y_pred: The predicted values.
        y_std: The standard deviation of the predictions.
        y_scaler: The scaler used to transform the target variable.
        train_shape: The index indicating the end of the training data.
        save_filepath: The file path where the plot should be saved, if provided (default is None).

    Returns:
        None
    """
    plt.figure(figsize=(16 * CM, 8 * CM))
    plt.plot(dates, y_scaler.inverse_transform(y_pred[:, np.newaxis]).squeeze(), label="mean")
    plt.fill_between(dates,
                     y_scaler.inverse_transform(y_pred[:, np.newaxis]).squeeze() + 2 * y_scaler.inverse_transform(
                         y_std[:, np.newaxis]).squeeze(),
                     y_scaler.inverse_transform(y_pred[:, np.newaxis]).squeeze() - 2 * y_scaler.inverse_transform(
                         y_std[:, np.newaxis]).squeeze(),
                     alpha=0.2,
                     label=r'mean $\pm$ 2 std')
    plt.axvspan(dates[0], dates[train_shape], alpha=0.08, color='k')
    plt.axvline(dates[train_shape], linestyle='--', c='k')
    plt.plot(dates, y, 'rx', markersize=2., label='measurements')

    plt.ylabel('Global Horizontal Irradiance [W/m$^2$]')
    plt.xlim(dates[[0, -1]])
    plt.legend(fontsize=FSIZE, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncols=3)
    ax = plt.gca()
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    ax.set_yticks(np.arange(-500, 1251, 250), np.arange(-500, 1251, 250))
    plt.ylim((-500, 1250))

    plt.text(0.3, 0.95, "Training data points", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.text(0.86, 0.95, "Forecasting", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    if save_filepath is not None:
        plt.savefig(save_filepath, bbox_inches='tight')

    plt.show()
