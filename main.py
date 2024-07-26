import logging
import random
from pathlib import Path
from joblib import dump
from datetime import datetime

from src.config_dataclass import Config, load_params_from_toml, update_params_from_toml
from src import CONFIGDIR, DATADIR, OUTPUTDIR
from src.preprocess import (read_data, clean_data, read_json,
                            create_time_cols, filter_data,
                            split_train_test, scale_data,
                            make_kernel, make_gp, rescale_data)
from src.plotting import plot_preds
from src.postprocess import calculate_dependent_variables


def setup_output_directories(_config: Config) -> None:
    """
    Setup the output directories for the training. Use of the pathlib.Path objects defined in src
    Args:
        _config (Config): the config objects with the training parameters
    Returns:
        None
    """
    timestamp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = OUTPUTDIR.joinpath(f"{f'{config.run.name}_' if config.run.name else ''}{timestamp_name}")
    PLOT_DIR = RUN_DIR.joinpath("plots")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(RUN_DIR.joinpath("model_training.log"))

    _config.output.run_dir = RUN_DIR
    _config.output.plot_dir = PLOT_DIR


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
    

if __name__ == '__main__':
    config = load_params_from_toml(CONFIGDIR.joinpath('config.toml'))
    config = update_params_from_toml(config, CONFIGDIR.joinpath('config_overwrite.toml'))
    
    setup_output_directories(config)

    logging.info(f"Configuration settings:\n" + "\n".join(map(str, config.__dict__.values())))

    random.seed(config.run.random_seed)

    df = read_data(DATADIR.joinpath(config.datafiles.source))
    df = clean_data(df, config.datafiles.drop_cols, config.datafiles.rename_dict)
    df = create_time_cols(df)

    var_description = read_json(DATADIR.joinpath(config.datafiles.col_desc))

    filt_df = filter_data(df,
                          config.filter.days,
                          config.filter.month,
                          config.filter.year)
    filt_df = filt_df.loc[~(filt_df[config.train_test.target] == 0).all(axis=1)]

    X_train, X_test, y_train, y_test = split_train_test(filt_df,
                                                        config.train_test.predictors,
                                                        config.train_test.target,
                                                        config.train_test.training_ratio)

    X_train_scaled, y_train_scaled, x_scaler, y_scaler = scale_data(X_train, y_train, mode=config.train_test.scaling_mode)

    kernel = make_kernel()
    gpr = make_gp(kernel, alpha=config.gpr.alpha, n_restarts_optimizer=config.gpr.n_restarts_optimizer,)

    logging.info("Training Gaussian Process Regressor...")
    gpr.fit(X_train_scaled, y_train_scaled)
    logging.info("Training completed.")

    dump(gpr, (model_path := config.output.run_dir.joinpath('gpr_model.joblib')))
    logging.info(f"Model saved to {model_path}")

    y_pred, y_std = gpr.predict(x_scaler.transform(X_test), return_std=True)

    y_pred_rescaled, y_std_rescaled = rescale_data([y_pred, y_std], y_scaler)

    target_set = set(config.train_test.target)
    plot_preds(filt_df['datetime'].values,
               filt_df[config.train_test.target].values,
               y_pred_rescaled, y_std_rescaled, X_train.shape[0],
               config.output.plot_dir.joinpath('forecasting.png'),
               plot_settings={'ylabels': {key: val for key, val in var_description.items() if key in target_set}})

    if all(value in target_set for value in ['GHI', 'DHI', 'DNI']):
        y_pred_post, y_std_post = calculate_dependent_variables(y_pred_rescaled, y_std_rescaled, filt_df['sol_alt'], config.train_test.target)

        plot_preds(filt_df['datetime'].values,
                   filt_df[config.train_test.target].values,
                   y_pred_post, y_std_post, X_train.shape[0],
                   config.output.plot_dir.joinpath('forecasting_post.png'),
                   plot_settings={'ylabels': {key: val for key, val in var_description.items() if key in config.train_test.target}})

