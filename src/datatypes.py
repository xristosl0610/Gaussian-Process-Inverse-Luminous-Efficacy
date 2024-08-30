from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@dataclass
class PreprocessedData:
    """
    Represents preprocessed data including training and test sets, scalers, and variable descriptions.

    Attributes:
        X_train: Features of the training set.
        X_test: Features of the test set.
        y_train: Target values of the training set.
        y_test: Target values of the test set.
        date_objs: Date objects associated with the data.
        X_train_scaled: Scaled features of the training set.
        y_train_scaled: Scaled target values of the training set.
        x_scaler: Scaler used for scaling the features.
        y_scaler: Scaler used for scaling the target values.
        var_description: Description of variables in the dataset.
        sol_alt: Solar altitude data.
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    date_objs: np.ndarray
    X_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    x_scaler: StandardScaler | MinMaxScaler
    y_scaler: StandardScaler | MinMaxScaler
    var_description: dict[str, str]
    sol_alt: np.ndarray
