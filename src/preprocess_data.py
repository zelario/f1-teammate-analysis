from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from . import load_data


def check_processed_cache(year, grand_prix, driver, segment=None):
    """Return a cached processed payload for a driver if it exists.

    The cache key uses year, grand prix, segment, and driver abbreviation.
    Returns the cached payload dict, or None when the file is missing.
    """
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "processed"

    # Accept either a driver string or a driver payload dict
    if isinstance(driver, dict):
        driver_key = driver.get("Driver") or driver.get("driver") or driver.get("Abbreviation") or driver.get("abbr")
    else:
        driver_key = driver

    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment if segment is not None else "session"
    key = f"{year}_{gp}_{seg}_{driver_key}"
    out_path = cache_dir / f"{key}.pkl"

    if out_path.exists():
        return pd.read_pickle(out_path)

    return None

def save_lap_cache(year, grand_prix, driver, processed_data, segment=None):
    """Persist a driver's processed payload to the processed cache.

    The cache key uses year, grand prix, segment, and driver abbreviation.
    Returns the output path after the payload is saved successfully.
    """
    
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "processed"
    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment if segment is not None else "session"
    key = f"{year}_{gp}_{seg}_{driver}"
    out_path = cache_dir / f"{key}.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed payload to {out_path}")
    pd.to_pickle(processed_data, out_path)
    print(f"Saved: {out_path}")

    return out_path


def load_processed_data(year, grand_prix, team):
    driver_1, driver_2, segment = load_data.find_drivers_and_segment(year, grand_prix, team)
    lap_1 = check_processed_cache(year, grand_prix, driver_1, segment)
    lap_2 = check_processed_cache(year, grand_prix, driver_2, segment)
    print(f"\nProcessed data for {year} {grand_prix} {team} found in cache.")
    print(f"Driver 1: {lap_1['Driver']}, Driver 2: {lap_2['Driver']}, Segment: {segment}")
    return lap_1, lap_2, segment


def save_processed_data(year, grand_prix, team, lap_1, lap_2, segment):
    driver_1 = lap_1.get("Driver")
    driver_2 = lap_2.get("Driver")
    save_lap_cache(year, grand_prix, driver_1, lap_1, segment)
    save_lap_cache(year, grand_prix, driver_2, lap_2, segment)
    print(f"\nProcessed data for {year} {grand_prix} {team} saved to cache.")
    print(f"Driver 1: {driver_1}, Driver 2: {driver_2}, Segment: {segment}")
    

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
    
    preprocessed_driver1_data = check_processed_cache(year, grand_prix, driver1_data, segment)
    preprocessed_driver2_data = check_processed_cache(year, grand_prix, driver2_data, segment)
    
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

    return d1_interpolated, d2_interpolated