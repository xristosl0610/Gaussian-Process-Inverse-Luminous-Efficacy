import logging
import random
from pathlib import Path
from joblib import dump
from datetime import datetime

import numpy as np
from src.plotting import plot_preds
from src.config_dataclass import Config, create_config, save_config_to_toml
from src import CONFIGDIR, DATADIR, OUTPUTDIR, BENCHMARK_MODELS
from src.preprocess import (read_data, read_json, preprocess_df,
                            split_train_test, scale_data, make_kernel, make_gp)
from src.plotting import plot_preds
from src.postprocess import rescale_data, calculate_metrics, save_metrics
from main import setup_logging, setup_output_directories

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    config, config_dict = create_config(CONFIGDIR.joinpath('config.toml'), CONFIGDIR.joinpath('config_overwrite.toml'))

    setup_output_directories(config)
    save_config_to_toml(config_dict, config.output.run_dir.joinpath('config.toml'))
    setup_logging(config.output.run_dir.joinpath("model_training.log"))

    logger.info(f"Configuration settings:\n" +
                "\n".join([str(value) for value in config.__dict__.values()]))

    random.seed(config.run.random_seed)

    df = read_data(DATADIR.joinpath(config.datafiles.source))
    var_description = read_json(DATADIR.joinpath(config.datafiles.col_desc))

    filt_df = preprocess_df(df, config)
    X_train, X_test, y_train, y_test, date_objs, sol_alt = split_train_test(filt_df, config)
    X_train_scaled, y_train_scaled, x_scaler, y_scaler = scale_data(X_train, y_train, mode=config.train_test.scaling_mode)

    kernel = make_kernel()
    gpr = make_gp(kernel, max_iters=config.gpr.max_iters, alpha=config.gpr.alpha, n_restarts_optimizer=config.gpr.n_restarts_optimizer)

    logger.info("Training Gaussian Process Regressor...")
    gpr.fit(X_train_scaled, y_train_scaled)
    logger.info("Training completed.")

    dump({'gpr': gpr, 'x_scaler': x_scaler, 'y_scaler': y_scaler, 'X_test': X_test, 'y_test': y_test},
         (model_path := config.output.run_dir.joinpath('gpr_model.joblib')))
    logger.info(f"Model saved to {model_path}")

    y_pred, y_std = gpr.predict(x_scaler.transform(X_test), return_std=True)
    y_pred_rescaled, y_std_rescaled = rescale_data([y_pred, y_std], y_scaler)

    y_pred_post, y_std_post = calculate_dependent_variables(y_pred_rescaled, y_std_rescaled,
                                                            sol_alt, config.train_test.target,
                                                            config.output.dependent_vars)
    target_set = set(config.output.dependent_vars)
    plot_settings = {'ylabels': {key: val for key, val in var_description.items() if key in target_set},
                     'date_format': '%H:%M' if config.datafiles.source == 'fivemin_data.csv' else '%d-%m-%y'}

    plot_preds(date_objs, y_test, y_pred_post, y_std_post,
               X_train.shape[0], config.output.plot_dir.joinpath('forecasting_post.png'),
               plot_settings=plot_settings)

    gpr_metrics = calculate_metrics(y_true=y_test,  y_pred=y_pred_post)
    save_metrics(gpr_metrics, config.output.dependent_vars,
                 config.output.run_dir.joinpath('metrics.csv'))

