import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from .load_save_data import *
    

def preprocess_telemetry(telemetry):
    """Select and clean the telemetry columns used for analysis.

    Parameters
    ----------
    telemetry : pandas.DataFrame
        Raw telemetry data for a single lap.

    Returns
    -------
    pandas.DataFrame
        Cleaned telemetry with only the relevant columns.
    """

    relevant_columns = [
        "Time",
        "Speed",
        "RPM",
        "nGear",
        "Throttle",
        "Brake",
        "Distance",
    ]

    preprocessed_telemetry = telemetry[relevant_columns].copy()
    
    # Convert Brake column to integer type
    preprocessed_telemetry["Brake"] = preprocessed_telemetry["Brake"].astype(int)
    
    preprocessed_telemetry["RPM"] = preprocessed_telemetry["RPM"].astype(int)
    
    # Convert Time to seconds since the start of the lap
    preprocessed_telemetry["Time"] = (preprocessed_telemetry["Time"] - preprocessed_telemetry["Time"].iloc[0]).dt.total_seconds()

    # Remove rows containing missing values and duplicate time entries
    preprocessed_telemetry = preprocessed_telemetry.dropna()
    preprocessed_telemetry = preprocessed_telemetry.drop_duplicates(subset="Time", keep="first")

    preprocessed_telemetry.reset_index(drop=True, inplace=True)

    return preprocessed_telemetry


def preprocess_driver_data(driver_data):
    """Normalize a driver's payload into analysis-ready values.

    Parameters:
    - driver_data: dict containing driver telemetry data and metadata.

    Returns:
    - dict containing the cleaned driver metadata and telemetry.
    """
    
    preprocessed_data = {}

    # Preserve the driver identifier in the processed payload.
    preprocessed_data['Driver'] = driver_data.get('Driver')
    
    # Ensure LapTime is in seconds
    lap_time_str = driver_data.get('LapTime')
    if lap_time_str is not None:
        try:
            preprocessed_data['LapTime'] = pd.to_timedelta(lap_time_str).total_seconds()
        except Exception:
            preprocessed_data['LapTime'] = None
    else:
        preprocessed_data['LapTime'] = None

    # Ensure LapStartTime is a timedelta object
    lap_start_time_str = driver_data.get('LapStartTime')
    if lap_start_time_str is not None:
        try:
            preprocessed_data['LapStartTime'] = pd.to_timedelta(lap_start_time_str)
        except Exception:
            preprocessed_data['LapStartTime'] = None
    else:
        preprocessed_data['LapStartTime'] = None

    # Copy TyreCompound and TyreAge directly
    preprocessed_data['TyreCompound'] = driver_data.get('TyreCompound')
    preprocessed_data['TyreAge'] = driver_data.get('TyreAge').astype(int)

    # Ensure Telemetry is a DataFrame
    telemetry = driver_data.get('Telemetry')
    if telemetry is not None and isinstance(telemetry, pd.DataFrame):
        preprocessed_data['Telemetry'] = preprocess_telemetry(telemetry)
    else:
        preprocessed_data['Telemetry'] = pd.DataFrame()

    return preprocessed_data

def preprocess_teammates_data(year, grand_prix, segment, driver1_data, driver2_data):
    """Preprocess both teammates' payloads.

    Parameters:
    - driver1_data: first driver's payload.
    - driver2_data: second driver's payload.

    Returns:
    - Tuple with both preprocessed payloads.
    """
    
    preprocessed_driver1_data = load_driver_cache(year, grand_prix, driver1_data, segment)
    preprocessed_driver2_data = load_driver_cache(year, grand_prix, driver2_data, segment)
    
    if preprocessed_driver1_data is not None and preprocessed_driver2_data is not None:
        return preprocessed_driver1_data, preprocessed_driver2_data

    preprocessed_driver1_data = preprocess_driver_data(driver1_data)
    preprocessed_driver2_data = preprocess_driver_data(driver2_data)

    return preprocessed_driver1_data, preprocessed_driver2_data


def interpolate_telemetry(d1_telemetry,d2_telemetry, n_points=1000):
    """
    Interpolate two drivers' telemetry onto a common distance grid.

    Continuous variables are linearly interpolated.
    Discrete variables use nearest-neighbor interpolation.

    Parameters
    ----------
    d1_telemetry : pandas.DataFrame
        First driver's telemetry data.

    d2_telemetry : pandas.DataFrame
        Second driver's telemetry data.

    n_points : int, default=1000
        Number of points in the common distance grid.

    Returns
    -------
    d1_interpolated : pandas.DataFrame
        First driver's interpolated telemetry.

    d2_interpolated : pandas.DataFrame
        Second driver's interpolated telemetry.
    """

    continuous_columns = [
        "Time",
        "Speed",
        "RPM",
        "Throttle"
    ]

    discrete_columns = [
        "nGear",
        "Brake"
    ]

    # Find the distance range shared by both drivers
    min_distance = max(
        d1_telemetry["Distance"].min(),
        d2_telemetry["Distance"].min()
    )

    max_distance = min(
        d1_telemetry["Distance"].max(),
        d2_telemetry["Distance"].max()
    )

    # Create a common distance grid
    distance = np.linspace(
        min_distance,
        max_distance,
        n_points
    )

    # Create the output DataFrames
    d1_interpolated = pd.DataFrame({
        "Distance": distance
    })

    d2_interpolated = pd.DataFrame({
        "Distance": distance
    })

    # Linearly interpolate continuous variables
    for column in continuous_columns:

        d1_interpolated[column] = np.interp(
            distance,
            d1_telemetry["Distance"],
            d1_telemetry[column]
        )

        d2_interpolated[column] = np.interp(
            distance,
            d2_telemetry["Distance"],
            d2_telemetry[column]
        )

    # Use nearest-neighbor interpolation for discrete variables
    for column in discrete_columns:

        d1_interpolator = interp1d(
            d1_telemetry["Distance"],
            d1_telemetry[column],
            kind="nearest"
        )

        d2_interpolator = interp1d(
            d2_telemetry["Distance"],
            d2_telemetry[column],
            kind="nearest"
        )

        d1_interpolated[column] = d1_interpolator(distance).astype(int)
        d2_interpolated[column] = d2_interpolator(distance).astype(int)
        
    d1_interpolated["RPM"] = d1_interpolated["RPM"].astype(int)
    d2_interpolated["RPM"] = d2_interpolated["RPM"].astype(int)

    return d1_interpolated, d2_interpolated