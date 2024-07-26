import json
from pathlib import Path
import pandas as pd
from src import DATADIR


def save_column_descriptions(col_desc: dict[str, str], file_path: Path) -> None:
    """
    Save the column descriptions dictionary to a JSON file.

    Args:
        col_desc (dict[str, str]): Dictionary mapping variable names to descriptions.
        file_path (Path): Path to the JSON file to save the descriptions.

    Returns:
        None
    """
    with open(file_path, 'w') as file:
        json.dump(col_desc, file, indent=4)


def change_and_save_date_format(filepath: Path, new_format: str, col_name: str = 'datetime') -> None:
    """
    Changes the date format in the specified column of a CSV file and saves it back to the same file.

    Args:
        filepath (Path): The path to the CSV file.
        new_format (str): The new date format to apply.
        col_name (str, optional): The name of the column containing the date (default is 'datetime').

    Returns:
        None
    """
    df = pd.read_csv(filepath)
    df[col_name] = pd.to_datetime(df[col_name]).dt.strftime(new_format)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":

    column_descriptions = {
        'sol_alt': 'solar altitude [degree]',
        'sol_az': 'solar azimuth [degree]',
        'GHI': 'global horizontal irradiance [W/m2]',
        'DNI': 'direct normal irradiance [W/m2]',
        'DHI': 'diffuse horizontal irradiance [W/m2]',
        'GHE': 'global horizontal illuminance [lux]',
        'cld_ttl_amt_id': 'total cloud cover [oktas 0-8]',
        'air_temperature': 'air temperature [degree C]',
        'dewpoint': 'dew point temperature [degree C]',
        'rltv_hum': 'relative humidity [%]',
        'stn_pres': 'pressure [hPa]',
        'wmo_hr_sun_dur': 'sunshine duration [hrs]'
    }

    save_column_descriptions(column_descriptions, DATADIR.joinpath("variable_description.json"))

    # change_and_save_date_format(DATADIR.joinpath('fivemin_data.csv'), new_format='%Y-%m-%d %H:%M:%S')
