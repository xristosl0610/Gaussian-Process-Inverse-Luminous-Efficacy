import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor

from src.config_dataclass import Config
from src import OUTPUTDIR, DATADIR
from src.datatypes import PreprocessedData
from src.gaussian_process import GPR
from src.preprocess import (read_data, read_json, preprocess_df,
                            split_train_test, scale_data, make_kernel, make_gp)
from src.postprocess import rescale_data


logger = logging.getLogger(__name__)


def setup_output_directories(_config: Config) -> None:
    """
    Setup the output directories for the training. Use of the pathlib.Path objects defined in src
    Args:
        _config (Config): the config objects with the training parameters
    Returns:
        None
    """
    timestamp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{_config.run.name}_" if _config.run.name else ""
    run_dir = OUTPUTDIR.joinpath(f"{run_name}{timestamp_name}")
    plot_dir = run_dir.joinpath("plots")

    plot_dir.mkdir(parents=True, exist_ok=True)

    _config.output.run_dir = run_dir
    _config.output.plot_dir = plot_dir


def setup_logging(log_file_path: Path) -> None:
    """
    Setup the logger
    Args:
        log_file_path (Path): the path to the log file

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )


def preprocess_dataset(config: Config) -> PreprocessedData:
    """
    Preprocesses the dataset according to the provided configuration.

    Args:
        config: Configuration object containing dataset and preprocessing settings.

    Returns:
        A PreprocessedData object containing preprocessed data arrays, scalers, and variable descriptions.
    """
    df = read_data(DATADIR.joinpath(config.datafiles.source))
    var_description = read_json(DATADIR.joinpath(config.datafiles.col_desc))
    filt_df = preprocess_df(df, config)
    X_train, X_test, y_train, y_test, date_objs, sol_alt = split_train_test(filt_df, config)
    X_train_scaled, y_train_scaled, x_scaler, y_scaler = scale_data(X_train, y_train, mode=config.train_test.scaling_mode)

    return PreprocessedData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        date_objs=date_objs, X_train_scaled=X_train_scaled,
        y_train_scaled=y_train_scaled, x_scaler=x_scaler,
        y_scaler=y_scaler, var_description=var_description, sol_alt=sol_alt
    )


def train_models(config: Config, X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, X_test: np.ndarray,
                 x_scaler: StandardScaler | MinMaxScaler, y_scaler: StandardScaler | MinMaxScaler) -> (np.ndarray, np.ndarray, GPR | GaussianProcessRegressor):
    """
    Trains a Gaussian Process Regressor model using the provided training data and configuration.

    Args:
        config: Configuration object containing model training settings.
        X_train_scaled: Scaled features of the training data.
        y_train_scaled: Scaled target values of the training data.
        X_test: Features of the test data.
        x_scaler: Scaler used for scaling the features.
        y_scaler: Scaler used for scaling the target values.

    Returns:
        Tuple of predicted and standard deviation arrays for the test data.
    """
    kernel = make_kernel()
    gpr = make_gp(kernel, max_iters=config.gpr.max_iters, alpha=config.gpr.alpha, n_restarts_optimizer=config.gpr.n_restarts_optimizer)
    logger.info("Training Gaussian Process Regressor...")
    gpr.fit(X_train_scaled, y_train_scaled)
    logger.info("Training completed.")

    y_pred, y_std = gpr.predict(x_scaler.transform(X_test), return_std=True)
    y_pred_rescaled, y_std_rescaled = rescale_data([y_pred, y_std], y_scaler)

    return y_pred_rescaled, y_std_rescaled, gpr
