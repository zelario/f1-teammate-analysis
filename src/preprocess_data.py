from pathlib import Path
import pandas as pd

def check_processed_cache(year, grand_prix, driver, segment=None):
    """Return a cached processed payload for a driver if it exists.

    The cache key uses year, grand prix, segment, and driver abbreviation.
    Returns the cached payload dict, or None when the file is missing.
    """
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "processed"
    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment if segment is not None else "session"
    key = f"{year}_{gp}_{seg}_{driver}"
    out_path = cache_dir / f"{key}.pkl"

    if out_path.exists():
        return pd.read_pickle(out_path)

    return None

def save_processed_cache(year, grand_prix, driver, processed_data, segment=None):
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