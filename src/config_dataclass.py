from dataclasses import dataclass
from pathlib import Path
import toml
from dacite import from_dict
from src.utils import merge_dicts
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


def create_config(config_path: str | Path, config_overwrite_path: str | Path) -> (Config, dict):
    """
    Load the parameters of a Config object from a TOML file
    Args:
        config_path: The path to the config TOML file containing the default parameter values.
        config_overwrite_path: The path to the config TOML file containing the parameters to be overwritten.

    Returns:
        Config: The Config object with the run's parameters.
        dict: A dictionary containing the Config parameter values.
    """
    config_dict = merge_toml_files(config_path, config_overwrite_path)

    return from_dict(data_class=Config, data=config_dict), config_dict


def merge_toml_files(base_file: Path, overwrite_file: Path) -> dict:
    """
    Merge two TOML files, giving priority to the overwrite file.

    Args:
        base_file (Path): Path to the base TOML file.
        overwrite_file (Path): Path to the overwrite TOML file.

    Returns:
        dict: Merged configuration.
    """
    base_config = toml.load(base_file)
    overwrite_config = toml.load(overwrite_file)

    return merge_dicts(base_config, overwrite_config)


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


def save_config_to_toml(config_dict: dict, filepath: Path) -> None:
    """
    Save the merged configuration dictionary to a TOML file.

    Args:
        config_dict (dict): Merged configuration dictionary.
        filepath (Path): Path to the output TOML file.
    """
    with filepath.open('w') as f:
        toml.dump(config_dict, f)


if __name__ == '__main__':
    config_test, config_dict_test = create_config(CONFIGDIR.joinpath('config.toml'), CONFIGDIR.joinpath('config_overwrite.toml'))
    print(config_test)
    output_config = OutputConfig(run_dir='output/run', plot_dir='output/plot')
    print(output_config)
