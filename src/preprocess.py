from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, ExpSineSquared, Kernel

from src.config_dataclass import Config

pd.options.mode.chained_assignment = None


def read_data(filepath: str | Path) -> pd.DataFrame:
    """
    Reads data from a CSV file and returns it as a pandas DataFrame.

    Args:
        filepath: The path to the CSV file to be read.

    Returns:
        pd.DataFrame: The data read from the CSV file.
    """
    return pd.read_csv(filepath)


def read_json(filepath: str | Path) -> dict[str, str]:
    """
    Reads a JSON file containing column descriptions and returns them as a dictionary.

    Args:
        filepath (str | Path): The path to the JSON file containing column descriptions.

    Returns:
        dict[str, str]: A dictionary mapping variable names to descriptions.
    """
    with open(filepath, 'r') as file:
        col_desc = json.load(file)
    return col_desc


def preprocess_df(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Preprocess a DataFrame based on the provided configuration settings.

    Args:
        df (pd.DataFrame): The input DataFrame to be preprocessed.
        config (Config): The configuration settings for data cleaning, time column creation, and data filtering.

    Returns:
        pd.DataFrame: The preprocessed DataFrame after cleaning, time column creation, and data filtering.
    """
    df = clean_data(df, config.datafiles.drop_cols, config.datafiles.rename_dict)
    df = create_time_cols(df)
    df = filter_data(df, config.filter.days, config.filter.month, config.filter.year)

    return df.loc[~(df[config.train_test.target] == 0).all(axis=1)]


def clean_data(df: pd.DataFrame, drop_cols: list[str], rename_dict: dict[str, str]) -> pd.DataFrame:
    """
    Cleans the input DataFrame by renaming columns and dropping specified columns, then removes rows with any missing values.

    Args:
        df: The input pandas DataFrame to be cleaned.
        drop_cols: A list of column names to be dropped from the DataFrame.
        rename_dict: A dictionary mapping old column names to new column names.

    Returns:
        pd.DataFrame: The cleaned DataFrame with specified columns dropped and missing values removed.
    """
    df.rename(columns=rename_dict, inplace=True)
    df.drop(drop_cols, axis=1, inplace=True)
    return df.dropna()


def create_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new time-related columns in the DataFrame by converting a 'datetime' column to datetime format and calculating minutes elapsed from the first timestamp.

    Args:
        df: The input pandas DataFrame containing a 'datetime' column.

    Returns:
        pd.DataFrame: The DataFrame with additional 'datetime' and 'minutes' columns.
    """
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df['minutes'] = (df['datetime'] - (base_time := df['datetime'].iloc[0])).dt.total_seconds() / 60
    df['hours'] = (df['datetime'] - base_time).dt.total_seconds() / 3600
    return df


def filter_data(df: pd.DataFrame, days: (list[int] | tuple[int, int]),
                month: int, year: int) -> pd.DataFrame:
    """
    Filters the DataFrame based on the specified year, month, and range of days.

    Args:
        df: The input pandas DataFrame to be filtered.
        days: A tuple or list representing the range of days to filter data for.
        month: The month to filter data for.
        year: The year to filter data for.

    Returns:
        pd.DataFrame: The filtered DataFrame based on the provided year, month, and day range.
    """
    return df[(df['datetime'].dt.year == year) &
              (df['datetime'].dt.month == month) &
              (df['datetime'].dt.day >= days[0]) &
              (df['datetime'].dt.day <= days[1])]


def split_train_test(df: pd.DataFrame, predictors: list[str],
                     target: list[str], training_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the input DataFrame into training and testing sets based on the provided training ratio.

    Args:
        df: The input pandas DataFrame to be split.
        predictors: A list of column names to be used as predictors.
        target: A list of column names to be used as the target variable.
        training_ratio: The ratio of data to be allocated for training.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, and y_test arrays.
    """
    n_train = round(training_ratio * df.shape[0])
    X, y = df[predictors].values, df[target].values
    return X[:n_train], X, y[:n_train], y


def scale_data(X_train: np.ndarray, y_train: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, StandardScaler | MinMaxScaler, StandardScaler | MinMaxScaler]:
    """
    Scales the input training data using either MinMaxScaler or StandardScaler based on the specified mode.

    Args:
        X_train: The input features for training.
        y_train: The target variable for training.
        mode: The scaling mode to be used ('minmax' for MinMaxScaler, 'standard' for StandardScaler).

    Returns:
        tuple: A tuple containing the scaled X_train, scaled y_train, X scaler, and y scaler.
    """
    scaler_class = MinMaxScaler if mode == 'minmax' else StandardScaler
    x_scaler, y_scaler = scaler_class(), scaler_class()
    X_train_scaled, y_train_scaled = x_scaler.fit_transform(X_train), y_scaler.fit_transform(y_train)

    return X_train_scaled, y_train_scaled, x_scaler, y_scaler


def make_kernel() -> Kernel:
    """
    Creates a composite kernel for Gaussian Process regression consisting of a nonlinear, seasonal, and noise component.

    Returns:
        Kernel: The composite kernel combining the nonlinear, seasonal, and noise components.
    """
    nonlinear_kernel = (ConstantKernel(constant_value=50.0, constant_value_bounds=(1e-20, 1e20)) *
                        Matern(length_scale=1.0, nu=2.5))
    seasonal_kernel = (ConstantKernel(constant_value=20.0, constant_value_bounds=(1e-20, 1e20)) *
                       ExpSineSquared(length_scale=1.0,
                                      periodicity=12 * 24,
                                      length_scale_bounds=(1e-20, 1e20)))
    noise_kernel = WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-50, 1e50))

    return nonlinear_kernel + seasonal_kernel + noise_kernel


def make_gp(kernel: Kernel, **kwargs) -> GaussianProcessRegressor:
    """
    Creates a Gaussian Process regressor with the specified kernel and additional keyword arguments.

    Args:
        kernel: The kernel function to be used in the Gaussian Process regressor.
        **kwargs: Additional keyword arguments to be passed to the GaussianProcessRegressor constructor.

    Returns:
        GaussianProcessRegressor: A Gaussian Process regressor initialized with the provided kernel and keyword arguments.
    """
    return GaussianProcessRegressor(kernel=kernel, **kwargs)
