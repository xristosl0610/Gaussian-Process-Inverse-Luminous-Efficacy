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


if __name__ == '__main__':
    from src import OUTPUTDIR

    loaded_model = load_gpr(OUTPUTDIR.joinpath('Case_1_20240830_181708', 'gpr_model.joblib'))
    y_pred, _ = loaded_model['gpr'].predict(loaded_model['x_scaler'].transform(loaded_model['X_test']), return_std=True)
    test_metrics = calculate_metrics(y_true=loaded_model['y_test'], y_pred=y_pred, y_scaler=loaded_model['y_scaler'])

    print(loaded_model)
    print(test_metrics)
