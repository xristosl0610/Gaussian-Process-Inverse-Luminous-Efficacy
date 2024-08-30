import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from src.utils import expand_if_vector


def rescale_data(scaled_arr: list[np.ndarray] | np.ndarray, scaler: StandardScaler | MinMaxScaler) -> tuple[Any, ...] | np.ndarray:
    """
    Rescales a list of numpy arrays using the provided scaler.

    Args:
        scaled_arr (list[np.ndarray] | np.ndarray): A list of numpy arrays or a single numpy array to be rescaled.
        scaler (StandardScaler | MinMaxScaler): The scaler to use for rescaling.

    Returns:
        tuple[Any, ...] | np.ndarray: A list of rescaled numpy arrays.

    Raises:
        ValueError: If the input arrays have shapes other than 1D or 2D.
    """
    if isinstance(scaled_arr, np.ndarray):
        if scaled_arr.ndim > 2:
            raise ValueError('Only 1D and 2D arrays are supported.')

        return scaler.inverse_transform(expand_if_vector(scaled_arr)).squeeze()

    else:
        for arr in scaled_arr:
            if arr.ndim > 2:
                raise ValueError('Only 1D and 2D arrays are supported.')

        return tuple(scaler.inverse_transform(expand_if_vector(arr)).squeeze()
                     for arr in scaled_arr)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_scaler: StandardScaler | MinMaxScaler | None = None) -> np.ndarray:
    """
    Calculate evaluation metrics for a Gaussian Process Regressor model.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        y_scaler (StandardScaler | MinMaxScaler): Optional scaler for target values. Defaults to None.

    Returns:
        np.ndarray: An array containing the R^2 score, RMSE, and NRMSE metrics.
                    If the metrics are calculated for multiple outputs, each column contains the same metric,
                    each row correspond to a different output.
    """
    y_true = np.atleast_2d(y_true).T if y_true.ndim == 1 else y_true
    y_pred = np.atleast_2d(y_pred).T if y_pred.ndim == 1 else y_pred

    y_pred_rescaled = y_pred if y_scaler is None else rescale_data(y_pred, y_scaler)

    r2 = r2_score(y_true, y_pred_rescaled, multioutput='raw_values' if y_true.ndim > 1 else 'uniform_average')

    rmse = root_mean_squared_error(y_true, y_pred_rescaled, multioutput='raw_values' if y_true.ndim > 1 else 'uniform_average')

    nrmse = rmse / np.ptp(y_true, axis=0)
    return np.stack([r2, rmse, nrmse], axis=1) if y_true.ndim > 1 else np.array([r2, rmse, nrmse])


def save_metrics(metrics: np.ndarray, model_names: list[str], output_path: Path):
    """
    Save the metrics to a CSV file.

    Args:
        metrics (np.ndarray): An array containing the R^2 score, RMSE, and NRMSE metrics.
                              If the metrics are calculated for multiple outputs, or multiple models,
                              each column contains the same metric, each row correspond to a different output and/or model.
        model_names (list[str]): A list of model names.
        output_path (Path): The directory where the metrics should be saved.
    """
    metrics_df = pd.DataFrame(metrics, index=model_names, columns=['R2 Score', 'RMSE', 'NRMSE'])
    metrics_df.to_csv(output_path)


def load_gpr(filepath: Path) -> GaussianProcessRegressor:
    """
    Load a Gaussian Process Regressor model from the specified file.

    Args:
        filepath (Path): Path to the file containing the saved model.

    Returns:
        GaussianProcessRegressor: The loaded Gaussian Process Regressor model.
    """
    return joblib.load(filepath)


def calculate_dependent_variables(y_pred: np.ndarray, y_std: np.ndarray,
                                  sol_alt: np.ndarray, targets: list[str],
                                  dependent_vars: list[str]) -> (np.ndarray, np.ndarray):
    """
    Calculate dependent variables based on the input data and solar altitude.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_std (np.ndarray): Standard deviations of the predictions.
        sol_alt (np.ndarray): Solar altitude values.
        targets (list[str]): List of target variables.
        dependent_vars (list[str]): List of dependent variables to be calculated.

    Returns:
        tuple: A tuple containing the post-processed predicted values and standard deviations.
    """
    y_pred_post = np.zeros((y_pred.shape[0], len(dependent_vars)))
    y_std_post = np.zeros_like(y_pred_post)

    target_mapping = {target: idx for idx, target in enumerate(targets)}

    ghi_idx = target_mapping.get('GHI')
    dhi_idx = target_mapping.get('DHI')
    dni_idx = target_mapping.get('DNI')

    for col, var in enumerate(dependent_vars):
        if var == 'GHI':
            var_mean, var_std = calc_ghi(y_pred[:, dhi_idx], y_std[:, dhi_idx],
                                         y_pred[:, dni_idx], y_std[:, dni_idx], sol_alt)
        elif var == 'DHI':
            var_mean, var_std = calc_dhi(y_pred[:, ghi_idx], y_std[:, ghi_idx],
                                         y_pred[:, dni_idx], y_std[:, dni_idx], sol_alt)
        elif var == 'DNI':
            var_mean, var_std = calc_dni(y_pred[:, ghi_idx], y_std[:, ghi_idx],
                                         y_pred[:, dhi_idx], y_std[:, dhi_idx], sol_alt)
        else:
            raise ValueError(f'Variable {var} is not supported for post calculation.')
        y_pred_post[:, col] = var_mean
        y_std_post[:, col] = var_std

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


if __name__ == '__main__':
    from src import OUTPUTDIR

    loaded_model = load_gpr(OUTPUTDIR.joinpath('Case_1_20240830_181708', 'gpr_model.joblib'))
    y_pred, _ = loaded_model['gpr'].predict(loaded_model['x_scaler'].transform(loaded_model['X_test']), return_std=True)
    test_metrics = calculate_metrics(y_true=loaded_model['y_test'], y_pred=y_pred, y_scaler=loaded_model['y_scaler'])

    print(loaded_model)
    print(test_metrics)
