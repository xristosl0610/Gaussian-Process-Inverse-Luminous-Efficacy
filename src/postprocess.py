import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import root_mean_squared_error


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
        if len(scaled_arr.shape) > 2:
            raise ValueError('Only 1D and 2D arrays are supported.')

        return scaler.inverse_transform(scaled_arr[:, np.newaxis] if len(scaled_arr.shape) == 1 else scaled_arr).squeeze()

    else:
        for arr in scaled_arr:
            if len(arr.shape) > 2:
                raise ValueError('Only 1D and 2D arrays are supported.')

        return tuple(scaler.inverse_transform(arr[:, np.newaxis] if len(arr.shape) == 1 else arr).squeeze()
                     for arr in scaled_arr)


def calculate_metrics(model: Any,
                      y_true: np.ndarray, X: np.ndarray,
                      x_scaler: StandardScaler | MinMaxScaler,
                      y_scaler: StandardScaler | MinMaxScaler,
                      y_pred: np.ndarray | None = None) -> tuple[float, float, float]:
    """
    Calculate evaluation metrics for a Gaussian Process Regressor model.

    Args:
        model (Any): The trained Gaussian Process Regressor model.
        y_true (np.ndarray): True target values.
        X (np.ndarray): Input features for prediction.
        x_scaler (StandardScaler | MinMaxScaler): Scaler for input features.
        y_scaler (StandardScaler | MinMaxScaler): Scaler for target values.
        y_pred (np.ndarray | None): Predicted target values. Optional. Defaults to None.

    Returns:
        tuple[float, float, float]: A tuple containing the R^2 score, RMSE, and NRMSE metrics.
    """
    y_true_scaled = y_true[:, np.newaxis] if len(y_true.shape) == 1 else y_true
    y_pred_rescaled = rescale_data(model.predict(x_scaler.transform(X)) if y_pred is None else y_pred, y_scaler)

    # TODO Calculate the r2 score based on the function by sklearn, and NOT the gpr method

    r2score = model.score(X=x_scaler.transform(X), y=y_scaler.transform(y_true_scaled))
    rmse = root_mean_squared_error(y_true, y_pred_rescaled)
    nrmse = rmse / (np.abs(y_true.max() - y_true.min()))

    return r2score, rmse.item(), nrmse.item()


def save_metrics(metrics: list[tuple[float, float, float]], model_names: list[str], output_path: Path):
    """
    Save the metrics to a CSV file.

    Args:
        metrics (list[tuple[float, float, float]]): A tuple containing the R^2 score, RMSE, and NRMSE metrics.
        model_names (list[str]): A list of model names.
        output_path (Path): The directory where the metrics should be saved.
    """
    # metrics_df = pd.DataFrame([{
    #     "R2 Score": metrics[0],
    #     "RMSE": metrics[1],
    #     "NRMSE": metrics[2]
    # }])
    # metrics_df.to_csv(output_path, index=False)

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

    loaded_model = load_gpr(OUTPUTDIR.joinpath('Test_20240726_164259', 'gpr_model.joblib'))

    test_metrics = calculate_metrics(loaded_model['gpr'], y_true=loaded_model['y_test'], X=loaded_model['X_test'],
                                     x_scaler=loaded_model['x_scaler'], y_scaler=loaded_model['y_scaler'])

    print(loaded_model)
    print(test_metrics)
