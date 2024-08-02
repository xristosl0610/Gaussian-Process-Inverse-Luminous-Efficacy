import logging
import random
from pathlib import Path
from joblib import dump
from datetime import datetime

from src.config_dataclass import Config, create_config, save_config_to_toml
from src import CONFIGDIR, DATADIR, OUTPUTDIR, BENCHMARK_MODELS
from src.preprocess import (read_data, read_json, preprocess_df,
                            split_train_test, scale_data, make_kernel, make_gp)
from src.plotting import plot_preds
from src.postprocess import rescale_data, calculate_metrics, save_metrics

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


def main():
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
    X_train, X_test, y_train, y_test, date_objs = split_train_test(filt_df, config)
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

    target_set = set(config.train_test.target)
    plot_settings = {'ylabels': {key: val for key, val in var_description.items() if key in target_set},
                     'date_format': '%H:%M' if config.datafiles.source == 'fivemin_data.csv' else '%d-%m-%y'}
    plot_preds(date_objs, y_test, y_pred_rescaled, y_std_rescaled, X_train.shape[0], config.output.plot_dir.joinpath('forecasting.png'),
               plot_settings=plot_settings, month_scale=config.datafiles.source == 'hourly_data.csv')

    if config.output.cumulative:
        plot_preds(date_objs, y_test, y_pred_rescaled, y_std_rescaled,
                   X_train.shape[0], config.output.plot_dir.joinpath('forecasting_cumulative.png'),
                   plot_settings=plot_settings, month_scale=config.datafiles.source == 'hourly_data.csv', cumulative=True)

    gpr_metrics = calculate_metrics(gpr, y_true=y_test, X=X_test, x_scaler=x_scaler, y_scaler=y_scaler, y_pred=y_pred)

    total_metrics = [gpr_metrics]
    for model in config.output.benchmarks:
        if model == 'random_forest':
            bench_mod = BENCHMARK_MODELS[model].fit(X_train_scaled, y_train_scaled.ravel())
        else:
            bench_mod = BENCHMARK_MODELS[model].fit(X_train_scaled, y_train_scaled)
        y_pred = bench_mod.predict(x_scaler.transform(X_test))
        metrics = calculate_metrics(bench_mod, y_true=y_test, X=X_test, x_scaler=x_scaler, y_scaler=y_scaler, y_pred=y_pred)
        total_metrics.append(metrics)
    save_metrics(total_metrics, ['GP', *config.output.benchmarks], config.output.run_dir.joinpath('metrics.csv'))


if __name__ == '__main__':
    main()
