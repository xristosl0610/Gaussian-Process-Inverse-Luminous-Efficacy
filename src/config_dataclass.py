from dataclasses import dataclass, is_dataclass, fields
from pathlib import Path
import toml
from dacite import from_dict, Config as DaciteConfig
from src import CONFIGDIR


@dataclass
class DatafilesConfig:
    source: str | Path
    drop_cols: list[str]
    rename_dict: dict | None
    model_path: str | Path


@dataclass
class FilterBoundsConfig:
    days: list[int]
    month: int
    year: int


@dataclass
class TrainTestConfig:
    predictors: list[str]
    target: list[str]
    training_ratio: float
    scaling_mode: str


@dataclass
class GaussianProcessConfig:
    alpha: float
    n_restarts_optimizer: int | None


@dataclass
class Config:
    datafiles: DatafilesConfig = DatafilesConfig
    filter: FilterBoundsConfig = FilterBoundsConfig
    train_test: TrainTestConfig = TrainTestConfig
    gpr: GaussianProcessConfig = GaussianProcessConfig


def load_params_from_toml(config_path: str | Path) -> Config:
    """
    Load the parameters of a Config object from a TOML file
    Args:
        config_path (str | Path): The path to the config TOML file containing the default parameter values

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
        obj (Config): The Config object whose attributes will be updated.
        updates (dict): A dictionary containing the updates to be applied to the object's attributes.

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
        params (Config): The Config object whose parameters will be updated.
        toml_file (str | Path): The path to the TOML file containing parameter updates.

    Returns:
        Config: The Config object with updated parameters.
    """
    with open(toml_file, 'r') as file:
        config_dict = toml.load(file)
    update_params(params, config_dict)
    return params


if __name__ == '__main__':
    config_test = load_params_from_toml(CONFIGDIR.joinpath('config.toml'))
    config_test = update_params_from_toml(config_test, CONFIGDIR.joinpath('config_overwrite.toml'))
    print(config_test)
