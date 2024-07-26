from dataclasses import dataclass, is_dataclass
from pathlib import Path
import toml
from dacite import from_dict
from src import CONFIGDIR


@dataclass
class ConfigBase:
    """
    Generate a string representation of the class instance based on its attributes.

    Returns:
        str: A string representation of the class instance.
    """

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = ', '.join(f'{field}={getattr(self, field)}' for field in self.__annotations__)
        return f'{class_name}({attributes})'


@dataclass
class RunConfig(ConfigBase):
    """
    A dataclass to store configuration settings for a run.

    Args:
        name: A string representing the name of the run.
        random_seed: An integer representing the random seed for reproducibility.

    Returns:
        None
    """
    name: str = ''
    random_seed: int = 64


@dataclass
class DatafilesConfig(ConfigBase):
    """
    A dataclass to store configuration settings for data files.

    Args:
        source: The path to the source data file.
        drop_cols: A list of columns to drop from the data.
        rename_dict: A dictionary for renaming columns, or None if no renaming is needed.
        model_path: The path to save the model.

    Returns:
        None

    Examples:
        config = DatafilesConfig(source='data.csv', drop_cols=['col1', 'col2'], rename_dict={'old_name': 'new_name'}, model_path='model.pkl')
    """
    source: str | Path
    drop_cols: list[str]
    rename_dict: dict | None
    model_path: str | Path
    col_desc: str | Path


@dataclass
class FilterBoundsConfig(ConfigBase):
    """
    A dataclass to store configuration settings for filtering bounds.

    Args:
        days: A list of integers representing days.
        month: An integer representing the month.
        year: An integer representing the year.

    Returns:
        None
    """
    days: list[int]
    month: int
    year: int


@dataclass
class TrainTestConfig(ConfigBase):
    """
    A dataclass to store configuration settings for training and testing data.

    Args:
        predictors: A list of strings representing predictor variables.
        target: A list of strings representing the target variable(s).
        training_ratio: A float representing the ratio of training data.
        scaling_mode: A string representing the scaling mode to be used.

    Returns:
        None
    """
    predictors: list[str]
    target: list[str]
    training_ratio: float
    scaling_mode: str


@dataclass
class GaussianProcessConfig(ConfigBase):
    """
    A dataclass to store configuration settings for Gaussian Process regression.

    Args:
        alpha: A float representing the regularization parameter.
        n_restarts_optimizer: An integer or None, representing the number of restarts of the optimizer.

    Returns:
        None
    """
    alpha: float
    n_restarts_optimizer: int | None


@dataclass
class OutputConfig(ConfigBase):
    """
    A dataclass to store configuration settings for output directories.

    Args:
        run_dir: A string or Path representing the directory for run outputs.
        plot_dir: A string or Path representing the directory for plot outputs.

    Returns:
        None
    """
    run_dir: str | Path
    plot_dir: str | Path


@dataclass
class Config:
    """
    A dataclass to store various configuration settings.

    Args:
        run: An instance of RunConfig for run configuration.
        datafiles: An instance of DatafilesConfig for data file configuration.
        filter: An instance of FilterBoundsConfig for filtering bounds configuration.
        train_test: An instance of TrainTestConfig for training and testing configuration.
        gpr: An instance of GaussianProcessConfig for Gaussian Process regression configuration.
        output: An instance of OutputConfig for output directories.

    Returns:
        None
    """
    run: RunConfig = RunConfig
    datafiles: DatafilesConfig = DatafilesConfig
    filter: FilterBoundsConfig = FilterBoundsConfig
    train_test: TrainTestConfig = TrainTestConfig
    gpr: GaussianProcessConfig = GaussianProcessConfig
    output: OutputConfig = OutputConfig


def load_params_from_toml(config_path: str | Path) -> Config:
    """
    Load the parameters of a Config object from a TOML file
    Args:
        config_path: The path to the config TOML file containing the default parameter values

    Returns:
        Config: The Config object with the default parameters.
    """
    with open(config_path, 'r') as file:
        config_dict = toml.load(file)

    return from_dict(data_class=Config, data=config_dict)


def update_params(obj: Config, updates: dict) -> None:
    """
    Recursively update attributes of an object based on a dictionary.

    Args:
        obj: The Config object whose attributes will be updated.
        updates: A dictionary containing the updates to be applied to the object's attributes.

    Returns:
        None
    """
    for key, value in updates.items():
        if hasattr(obj, key):
            attr = getattr(obj, key)
            if is_dataclass(attr) and isinstance(value, dict):
                update_params(attr, value)
            else:
                setattr(obj, key, value)


def update_params_from_toml(params: Config, toml_file: str | Path) -> Config:
    """
    Update the parameters of a Config object from a TOML file.

    Args:
        params: The Config object whose parameters will be updated.
        toml_file: The path to the TOML file containing parameter updates.

    Returns:
        Config: The Config object with updated parameters.
    """
    with open(toml_file, 'r') as file:
        config_dict = toml.load(file)
    update_params(params, config_dict)
    return params


def delete_config_file_contents(toml_file: str | Path) -> None:
    """
    Delete the contents of a TOML file.

    Args:
        toml_file (str | Path): The path to the TOML file containing config parameters.

    Returns:

    """
    with open(toml_file, 'w') as file:
        toml.dump({}, file)


def update_config_file(config_file: Path, new_config: dict):
    """
    Update the TOML configuration file with new values.

    Args:
        config_file (Path): The path to the TOML configuration file.
        new_config (Dict[str, Any]): The new configuration values to update.
    """
    with open(config_file, 'r') as file:
        config_dict = toml.load(file)

    config_dict.update(new_config)

    with open(config_file, 'w') as file:
        toml.dump(config_dict, file)


if __name__ == '__main__':
    config_test = load_params_from_toml(CONFIGDIR.joinpath('config.toml'))
    config_test = update_params_from_toml(config_test, CONFIGDIR.joinpath('config_overwrite.toml'))
    print(config_test)
    # output_config = OutputConfig(run_dir='output/run', plot_dir='output/plot')
    # print(output_config)
