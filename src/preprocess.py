from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, ExpSineSquared, Kernel

from src.config_dataclass import Config
from src.gaussian_process import GPR

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
    df['base_time'] = pd.to_datetime(df['datetime'].dt.year.astype(str) + '-01-01')
    df['minutes'] = (df['datetime'] - df['base_time']).dt.total_seconds() / 60
    df['hours'] = (df['datetime'] - df['base_time']).dt.total_seconds() / 3600

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


def split_train_test(df: pd.DataFrame, config: Config) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Splits the input DataFrame into training and testing sets based on the provided training ratio.

    Args:
        df: The input pandas DataFrame to be split.
        config: Configuration settings for data cleaning, time column creation, and data filtering.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test and test datetime arrays.
    """
    if config.train_test.test_days and config.train_test.test_month and config.train_test.test_year:
        df_train = filter_data(df, config.train_test.train_days, config.train_test.train_month, config.train_test.train_year)
        df_test = filter_data(df, config.train_test.test_days, config.train_test.test_month, config.train_test.test_year)
        return (df_train[config.train_test.predictors].values, df_test[config.train_test.predictors].values,
                df_train[config.train_test.target].values, df_test[config.train_test.target].values, df_test['datetime'].values)
    else:
        df = filter_data(df, config.train_test.train_days, config.train_test.train_month, config.train_test.train_year)
        n_train = round(config.train_test.training_ratio * df.shape[0])
        X, y = df[config.train_test.predictors].values, df[config.train_test.target].values
        return X[:n_train], X, y[:n_train], y, df['datetime'].values


def scale_data(X_train: np.ndarray, y_train: np.ndarray, mode: str = 'Standard')\
        -> tuple[np.ndarray, np.ndarray, StandardScaler | MinMaxScaler, StandardScaler | MinMaxScaler]:
    """
    Scales the predictor (X) and target (y) data using either StandardScaler or MinMaxScaler based on the specified mode.
    Args:
        X_train: The input features for training.
        y_train: The target variable for training.
        mode (str, optional): The scaling mode to use ('Standard' or 'MinMax'). Defaults to 'Standard'.

    Returns:
        tuple: A tuple containing the scaled X_train, scaled y_train, X scaler, and y scaler.
    """
    scaler_classes = {'Standard': StandardScaler, 'MinMax': MinMaxScaler}
    if mode not in scaler_classes:
        raise ValueError(f"Unsupported scaling mode: {mode}. Supported modes: {list(scaler_classes.keys())}")

    x_scaler, y_scaler = scaler_classes[mode](), scaler_classes[mode]()
    return x_scaler.fit_transform(X_train), y_scaler.fit_transform(y_train), x_scaler, y_scaler


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
                                      periodicity=31 * 24, periodicity_bounds=(1e-20, 1e20),
                                      length_scale_bounds=(1e-20, 1e20)))
    noise_kernel = WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-50, 1e50))

    return nonlinear_kernel + seasonal_kernel + noise_kernel


def make_gp(kernel: Kernel, max_iters: int | float, **kwargs) -> GPR | GaussianProcessRegressor:
    """
    Creates a Gaussian Process regressor with the specified kernel and additional keyword arguments.

    Args:
        kernel: The kernel function to be used in the Gaussian Process regressor.
        max_iters: The maximum number of iteration for the optimizer.
        **kwargs: Additional keyword arguments to be passed to the GaussianProcessRegressor constructor.

    Returns:
        GPR | GaussianProcessRegressor: A Gaussian Process regressor initialized with the provided kernel and keyword arguments.
    """
    return GPR(kernel=kernel, max_iter=max_iters, **kwargs)
