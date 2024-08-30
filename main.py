import logging
import random
from joblib import dump
import numpy as np

from src.config_dataclass import create_config, save_config_to_toml
from src.workflow import setup_output_directories, setup_logging, preprocess_dataset, train_models
from src import CONFIGDIR, BENCHMARK_MODELS
from src.plotting import plot_preds
from src.postprocess import calculate_metrics, save_metrics

logger = logging.getLogger(__name__)


def main() -> None:
    config, config_dict = create_config(CONFIGDIR.joinpath('config.toml'), CONFIGDIR.joinpath('config_overwrite.toml'))

    setup_output_directories(config)
    save_config_to_toml(config_dict, config.output.run_dir.joinpath('config.toml'))
    setup_logging(config.output.run_dir.joinpath("model_training.log"))

    logger.info(f"Configuration settings:\n" +
                "\n".join([str(value) for value in config.__dict__.values()]))

    random.seed(config.run.random_seed)

    proc_data = preprocess_dataset(config)

    y_pred_rescaled, y_std_rescaled, gpr = train_models(config,
                                                        X_train_scaled=proc_data.X_train_scaled,
                                                        y_train_scaled=proc_data.y_train_scaled,
                                                        X_test=proc_data.X_test,
                                                        x_scaler=proc_data.x_scaler,
                                                        y_scaler=proc_data.y_scaler)

    dump({'gpr': gpr,
          'x_scaler': proc_data.x_scaler,
          'y_scaler': proc_data.y_scaler,
          'X_test': proc_data.X_test,
          'y_test': proc_data.y_test},
         (model_path := config.output.run_dir.joinpath('gpr_model.joblib')))
    logger.info(f"Model saved to {model_path}")

    target_set = set(config.output.dependent_vars)
    plot_settings = {'ylabels': {key: val for key, val in proc_data.var_description.items() if key in target_set},
                     'date_format': '%H:%M' if config.datafiles.source == 'fivemin_data.csv' else '%d-%m-%y'}
    plot_preds(proc_data.date_objs, proc_data.y_test,
               y_pred_rescaled, y_std_rescaled, proc_data.X_train.shape[0],
               config.output.plot_dir.joinpath('forecasting.png'),
               plot_settings=plot_settings, month_scale=config.datafiles.source == 'hourly_data.csv')

    if config.output.cumulative:
        plot_preds(proc_data.date_objs, proc_data.y_test, y_pred_rescaled, y_std_rescaled,
                   proc_data.X_train.shape[0], config.output.plot_dir.joinpath('forecasting_cumulative.png'),
                   plot_settings=plot_settings, month_scale=config.datafiles.source == 'hourly_data.csv', cumulative=True)

    gpr_metrics = calculate_metrics(y_true=proc_data.y_test, y_pred=y_pred_rescaled)

    total_metrics = [gpr_metrics]
    for model in config.output.benchmarks:
        if model == 'random_forest':
            bench_mod = BENCHMARK_MODELS[model].fit(proc_data.X_train_scaled, proc_data.y_train_scaled.ravel())
        else:
            bench_mod = BENCHMARK_MODELS[model].fit(proc_data.X_train_scaled, proc_data.y_train_scaled)
        y_pred = bench_mod.predict(proc_data.x_scaler.transform(proc_data.X_test))
        metrics = calculate_metrics(y_true=proc_data.y_test, y_pred=y_pred, y_scaler=proc_data.y_scaler)
        total_metrics.append(metrics)
    save_metrics(np.stack(total_metrics, axis=1).squeeze(), ['GP', *config.output.benchmarks], config.output.run_dir.joinpath('metrics.csv'))


if __name__ == '__main__':
    main()
