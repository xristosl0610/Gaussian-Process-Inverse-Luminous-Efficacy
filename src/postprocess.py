import joblib
from pathlib import Path
import numpy as np
from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error


def rescale_data(arr_list: list[np.ndarray], scaler: StandardScaler | MinMaxScaler) -> tuple[Any, ...]:
    """
    Rescales a list of numpy arrays using the provided scaler.

    Args:
        arr_list (list[np.ndarray]): A list of numpy arrays to be rescaled.
        scaler (StandardScaler | MinMaxScaler): The scaler to use for rescaling.

    Returns:
        tuple[Any, ...]: A list of rescaled numpy arrays.

    Raises:
        ValueError: If the input arrays have shapes other than 1D or 2D.
    """
    for arr in arr_list:
        if len(arr.shape) > 2:
            raise ValueError('Only 1D and 2D arrays are supported.')

    return tuple(
        scaler.inverse_transform(arr[:, np.newaxis] if len(arr.shape) == 1 else arr).squeeze()
        for arr in arr_list
    )


def calculate_metrics(gpr: GaussianProcessRegressor,
                      y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray,
                      x_scaler: StandardScaler | MinMaxScaler,
                      y_scaler: StandardScaler | MinMaxScaler) -> tuple[float, float, float]:
    """
    Calculate evaluation metrics for a Gaussian Process Regressor model.

    Args:
        gpr (GaussianProcessRegressor): The trained Gaussian Process Regressor model.
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        X (np.ndarray): Input features for prediction.
        x_scaler (StandardScaler | MinMaxScaler): Scaler for input features.
        y_scaler (StandardScaler | MinMaxScaler): Scaler for target values.

    Returns:
        tuple[float, float, float]: A tuple containing the R^2 score, RMSE, and NRMSE metrics.
    """
    y_true_scaled = y_true[:, np.newaxis] if len(y_true.shape) == 1 else y_true
    y_pred_rescaled = rescale_data([y_pred], y_scaler)

    r2score = gpr.score(X=x_scaler.transform(X), y=y_scaler.transform(y_true_scaled))
    rmse = mean_squared_error(y_true, y_pred_rescaled, squared=False)
    nrmse = rmse / (np.abs(y_true.max() - y_true.min()))

    return r2score, rmse, nrmse


def calculate_dependent_variables(y_pred: np.ndarray, y_std: np.ndarray,
                                  sol_alt: np.ndarray, targets: list[str]) -> (np.ndarray, np.ndarray):
    """
    Calculate dependent variables based on the input data and solar altitude.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_std (np.ndarray): Standard deviations of the predictions.
        sol_alt (np.ndarray): Solar altitude values.
        targets (list[str]): List of target variables.

    Returns:
        tuple: A tuple containing the post-processed predicted values and standard deviations.
    """
    y_pred_post, y_std_post = np.zeros_like(y_pred), np.zeros_like(y_std)

    ghi_idx = targets.index('GHI')
    dhi_idx = targets.index('DHI')
    dni_idx = targets.index('DNI')

    ghi_mean, ghi_std = calc_ghi(y_pred[:, dhi_idx], y_std[:, dhi_idx],
                                 y_pred[:, dni_idx], y_std[:, dni_idx], sol_alt)

    dhi_mean, dhi_std = calc_dhi(y_pred[:, ghi_idx], y_std[:, ghi_idx],
                                 y_pred[:, dni_idx], y_std[:, dni_idx], sol_alt)

    dni_mean, dni_std = calc_dni(y_pred[:, ghi_idx], y_std[:, ghi_idx],
                                 y_pred[:, dhi_idx], y_std[:, dhi_idx], sol_alt)

    y_pred_post[:, ghi_idx] = ghi_mean
    y_std_post[:, ghi_idx] = ghi_std

    y_pred_post[:, dhi_idx] = dhi_mean
    y_std_post[:, dhi_idx] = dhi_std

    y_pred_post[:, dni_idx] = dni_mean
    y_std_post[:, dni_idx] = dni_std

    return y_pred_post, y_std_post


def calc_ghi(dhi_mean: np.ndarray, dhi_std: np.ndarray,
             dni_mean: np.ndarray, dni_std: np.ndarray,
             sol_alt: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the global horizontal irradiance (GHI) mean and standard deviation based on the provided values.

    Args:
        dhi_mean (np.ndarray): Mean of diffuse horizontal irradiance (DHI).
        dhi_std (np.ndarray): Standard deviation of DHI.
        dni_mean (np.ndarray): Mean of direct normal irradiance (DNI).
        dni_std (np.ndarray): Standard deviation of DNI.
        sol_alt (np.ndarray): Solar altitude values.

    Returns:
        tuple: A tuple containing the calculated GHI mean and standard deviation.
    """
    ghi_mean = np.sin(sol_alt) * dni_mean + dhi_mean
    ghi_variance = (np.sin(sol_alt) ** 2) * (dni_std ** 2) + dhi_std ** 2
    ghi_std = np.sqrt(ghi_variance)

    return ghi_mean, ghi_std


def calc_dhi(ghi_mean: np.ndarray, ghi_std: np.ndarray,
             dni_mean: np.ndarray, dni_std: np.ndarray,
             sol_alt: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the diffuse horizontal irradiance (DHI) mean and standard deviation based on the provided values.

    Args:
        ghi_mean (np.ndarray): Mean of global horizontal irradiance (GHI).
        ghi_std (np.ndarray): Standard deviation of GHI.
        dni_mean (np.ndarray): Mean of direct normal irradiance (DNI).
        dni_std (np.ndarray): Standard deviation of DNI.
        sol_alt (np.ndarray): Solar altitude values.

    Returns:
        tuple: A tuple containing the calculated DHI mean and standard deviation.
    """
    dhi_mean = ghi_mean - np.sin(sol_alt) * dni_mean
    dhi_variance = ghi_std ** 2 + (np.sin(sol_alt) ** 2) * (dni_std ** 2)
    dhi_std = np.sqrt(dhi_variance)

    return dhi_mean, dhi_std


def calc_dni(ghi_mean: np.ndarray, ghi_std: np.ndarray,
             dhi_mean: np.ndarray, dhi_std: np.ndarray,
             sol_alt: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the direct normal irradiance (DNI) mean and standard deviation based on the provided values.

    Args:
        ghi_mean (np.ndarray): Mean of global horizontal irradiance (GHI).
        ghi_std (np.ndarray): Standard deviation of GHI.
        dhi_mean (np.ndarray): Mean of diffuse horizontal irradiance (DHI).
        dhi_std (np.ndarray): Standard deviation of DHI.
        sol_alt (np.ndarray): Solar altitude values.

    Returns:
        tuple: A tuple containing the calculated DNI mean and standard deviation.
    """
    dni_mean = (ghi_mean - dhi_mean) / np.sin(sol_alt)
    dni_variance = (ghi_std ** 2 + dhi_std ** 2) / (np.sin(sol_alt) ** 2)
    dni_std = np.sqrt(dni_variance)

    return dni_mean, dni_std


def load_gpr(filepath: Path) -> GaussianProcessRegressor:
    """
    Load a Gaussian Process Regressor model from the specified file.

    Args:
        filepath (Path): Path to the file containing the saved model.

    Returns:
        GaussianProcessRegressor: The loaded Gaussian Process Regressor model.
    """
    return joblib.load(filepath)


if __name__ == '__main__':
    from src import OUTPUTDIR

    loaded_model = load_gpr(OUTPUTDIR.joinpath('Test_20240726_154353', 'gpr_model.joblib'))

    metrics = calculate_metrics(loaded_model['gpr'], )

    print(loaded_model)


