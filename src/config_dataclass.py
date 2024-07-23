from dataclasses import dataclass
from pathlib import Path
import toml
from typing import Any


@dataclass
class Config:
    datafile: str
    drop_cols: list[str]
    rename_dict: dict[str, str]
    filter_bounds: dict[str, list[int, int] | int]
    predictors: list[str]
    target: list[str]
    training_ratio: float
    scaling_mode: str
    gpr_params: dict[str, int | float]
    model_path: str


def load_config(filepath: Path) -> Config:
    with open(filepath, 'r') as f:
        config_dict = toml.load(f)
    return Config(**config_dict)