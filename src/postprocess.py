import numpy as np


def calculate_dependent_variables(y_pred: np.ndarray, y_std: np.ndarray,
                                  sol_alt: np.ndarray, targets: list[str]) -> (np.ndarray, np.ndarray):
    """
    Calculate dependent variables based on the input data and solar altitude.

    Args:
        y_pred (np.ndarray): Predicted values.
        y_std (np.ndarray): Standard deviations of the predictions.
        sol_alt (np.ndarray): Solar altitude values.
        targets (list[str]): List of target variables.

    Returns:
        tuple: A tuple containing the post-processed predicted values and standard deviations.
    """
    y_pred_post, y_std_post = np.zeros_like(y_pred), np.zeros_like(y_std)

    ghi_idx = targets.index('GHI')
    dhi_idx = targets.index('DHI')
    dni_idx = targets.index('DNI')

    ghi_mean, ghi_std = calc_ghi(y_pred[:, dhi_idx], y_std[:, dhi_idx],
                                 y_pred[:, dni_idx], y_std[:, dni_idx], sol_alt)

    dhi_mean, dhi_std = calc_dhi(y_pred[:, ghi_idx], y_std[:, ghi_idx],
                                 y_pred[:, dni_idx], y_std[:, dni_idx], sol_alt)

    dni_mean, dni_std = calc_dni(y_pred[:, ghi_idx], y_std[:, ghi_idx],
                                 y_pred[:, dhi_idx], y_std[:, dhi_idx], sol_alt)

    y_pred_post[:, ghi_idx] = ghi_mean
    y_std_post[:, ghi_idx] = ghi_std

    y_pred_post[:, dhi_idx] = dhi_mean
    y_std_post[:, dhi_idx] = dhi_std

    y_pred_post[:, dni_idx] = dni_mean
    y_std_post[:, dni_idx] = dni_std

    return y_pred_post, y_std_post


def calc_ghi(dhi_mean: np.ndarray, dhi_std: np.ndarray,
             dni_mean: np.ndarray, dni_std: np.ndarray,
             sol_alt: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the global horizontal irradiance (GHI) mean and standard deviation based on the provided values.

    Args:
        dhi_mean (np.ndarray): Mean of diffuse horizontal irradiance (DHI).
        dhi_std (np.ndarray): Standard deviation of DHI.
        dni_mean (np.ndarray): Mean of direct normal irradiance (DNI).
        dni_std (np.ndarray): Standard deviation of DNI.
        sol_alt (np.ndarray): Solar altitude values.

    Returns:
        tuple: A tuple containing the calculated GHI mean and standard deviation.
    """
    ghi_mean = np.sin(sol_alt) * dni_mean + dhi_mean
    ghi_variance = (np.sin(sol_alt) ** 2) * (dni_std ** 2) + dhi_std ** 2
    ghi_std = np.sqrt(ghi_variance)

    return ghi_mean, ghi_std


def calc_dhi(ghi_mean: np.ndarray, ghi_std: np.ndarray,
             dni_mean: np.ndarray, dni_std: np.ndarray,
             sol_alt: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the diffuse horizontal irradiance (DHI) mean and standard deviation based on the provided values.

    Args:
        ghi_mean (np.ndarray): Mean of global horizontal irradiance (GHI).
        ghi_std (np.ndarray): Standard deviation of GHI.
        dni_mean (np.ndarray): Mean of direct normal irradiance (DNI).
        dni_std (np.ndarray): Standard deviation of DNI.
        sol_alt (np.ndarray): Solar altitude values.

    Returns:
        tuple: A tuple containing the calculated DHI mean and standard deviation.
    """
    dhi_mean = ghi_mean - np.sin(sol_alt) * dni_mean
    dhi_variance = ghi_std ** 2 + (np.sin(sol_alt) ** 2) * (dni_std ** 2)
    dhi_std = np.sqrt(dhi_variance)

    return dhi_mean, dhi_std


def calc_dni(ghi_mean: np.ndarray, ghi_std: np.ndarray,
             dhi_mean: np.ndarray, dhi_std: np.ndarray,
             sol_alt: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the direct normal irradiance (DNI) mean and standard deviation based on the provided values.

    Args:
        ghi_mean (np.ndarray): Mean of global horizontal irradiance (GHI).
        ghi_std (np.ndarray): Standard deviation of GHI.
        dhi_mean (np.ndarray): Mean of diffuse horizontal irradiance (DHI).
        dhi_std (np.ndarray): Standard deviation of DHI.
        sol_alt (np.ndarray): Solar altitude values.

    Returns:
        tuple: A tuple containing the calculated DNI mean and standard deviation.
    """
    dni_mean = (ghi_mean - dhi_mean) / np.sin(sol_alt)
    dni_variance = (ghi_std ** 2 + dhi_std ** 2) / (np.sin(sol_alt) ** 2)
    dni_std = np.sqrt(dni_variance)

    return dni_mean, dni_std
