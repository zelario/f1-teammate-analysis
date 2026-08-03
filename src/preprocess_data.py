import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from .load_save_data import *


def preprocess_telemetry(telemetry):
    """Selects and cleans the telemetry columns used for teammate analysis.

    This function extracts a set of relevant telemetry columns (Time, Speed, RPM,
    nGear, Throttle, Brake, Distance) from the raw lap telemetry, casts discrete
    variables to proper types, converts Time to total seconds elapsed since the
    beginning of the lap, and removes missing or duplicated time records.

    Parameters:
        telemetry (pandas.DataFrame): Raw telemetry data for a single lap from FastF1.

    Returns:
        pandas.DataFrame: Cleaned and structured telemetry containing only the
                          processed columns of interest.
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
    preprocessed_telemetry["Time"] = (
        preprocessed_telemetry["Time"] - preprocessed_telemetry["Time"].iloc[0]
    ).dt.total_seconds()

    # Remove rows containing missing values and duplicate time entries
    preprocessed_telemetry = preprocessed_telemetry.dropna()
    preprocessed_telemetry = preprocessed_telemetry.drop_duplicates(
        subset="Time", keep="first"
    )

    preprocessed_telemetry.reset_index(drop=True, inplace=True)

    return preprocessed_telemetry


def preprocess_driver_data(driver_data):
    """Normalize a driver's payload into analysis-ready values.

    Parameters:
        driver_data (dict): Dictionary containing driver telemetry data and
                            metadata.

    Returns:
        dict: Dictionary containing the cleaned driver metadata and telemetry.
    """

    preprocessed_data = {}

    # Preserve the driver identifier in the processed payload.
    preprocessed_data["Driver"] = driver_data.get("Driver")

    # Ensure LapTime is in seconds
    lap_time_str = driver_data.get("LapTime")
    if lap_time_str is not None:
        try:
            preprocessed_data["LapTime"] = pd.to_timedelta(lap_time_str).total_seconds()
        except Exception:
            preprocessed_data["LapTime"] = None
    else:
        preprocessed_data["LapTime"] = None

    # Ensure LapStartTime is a timedelta object
    lap_start_time_str = driver_data.get("LapStartTime")
    if lap_start_time_str is not None:
        try:
            preprocessed_data["LapStartTime"] = pd.to_timedelta(lap_start_time_str)
        except Exception:
            preprocessed_data["LapStartTime"] = None
    else:
        preprocessed_data["LapStartTime"] = None

    # Copy TyreCompound and TyreAge directly
    preprocessed_data["TyreCompound"] = driver_data.get("TyreCompound")
    preprocessed_data["TyreAge"] = driver_data.get("TyreAge").astype(int)

    # Ensure Telemetry is a DataFrame
    telemetry = driver_data.get("Telemetry")
    if telemetry is not None and isinstance(telemetry, pd.DataFrame):
        preprocessed_data["Telemetry"] = preprocess_telemetry(telemetry)
    else:
        preprocessed_data["Telemetry"] = pd.DataFrame()

    return preprocessed_data


def preprocess_teammates_data(year, grand_prix, segment, driver1_data, driver2_data):
    """Preprocess both teammates' payloads.

    Parameters:
        year (int): The year of the race.
        grand_prix (str): The name or ID of the Grand Prix.
        segment (str): The session segment (e.g., 'Race').
        driver1_data (dict): First driver's payload.
        driver2_data (dict): Second driver's payload.

    Returns:
        tuple: A tuple containing both preprocessed payloads (driver1, driver2).
    """

    preprocessed_driver1_data = load_driver_cache(
        year, grand_prix, driver1_data, segment
    )
    preprocessed_driver2_data = load_driver_cache(
        year, grand_prix, driver2_data, segment
    )

    if preprocessed_driver1_data is not None and preprocessed_driver2_data is not None:
        return preprocessed_driver1_data, preprocessed_driver2_data

    preprocessed_driver1_data = preprocess_driver_data(driver1_data)
    preprocessed_driver2_data = preprocess_driver_data(driver2_data)

    return preprocessed_driver1_data, preprocessed_driver2_data


def interpolate_telemetry(
    telemetry_1, telemetry_2, lap_time_1, lap_time_2, n_points=1000
):
    """Interpolates telemetry data between two drivers to a common distance/fraction grid.

    Parameters:
        telemetry_1 (pandas.DataFrame): Telemetry data for the first driver.
        telemetry_2 (pandas.DataFrame): Telemetry data for the second driver.
        lap_time_1 (float): Total lap time for the first driver in seconds.
        lap_time_2 (float): Total lap time for the second driver in seconds.
        n_points (int, optional): Number of points for interpolation.
                                  Defaults to 1000.

    Returns:
        tuple: A tuple containing the interpolated telemetry DataFrames for both drivers.
    """

    continuous_columns = ["Speed", "RPM", "Throttle"]

    discrete_columns = ["nGear", "Brake"]

    telemetry_1 = telemetry_1.sort_values("Distance").reset_index(drop=True).copy()
    telemetry_2 = telemetry_2.sort_values("Distance").reset_index(drop=True).copy()

    telemetry_1["Distance"] -= telemetry_1["Distance"].iloc[0]
    telemetry_2["Distance"] -= telemetry_2["Distance"].iloc[0]

    telemetry_1["Time"] -= telemetry_1["Time"].iloc[0]
    telemetry_2["Time"] -= telemetry_2["Time"].iloc[0]

    telemetry_1["Time"] = (
        telemetry_1["Time"] / telemetry_1["Time"].iloc[-1] * lap_time_1
    )
    telemetry_2["Time"] = (
        telemetry_2["Time"] / telemetry_2["Time"].iloc[-1] * lap_time_2
    )

    max_distance = min(telemetry_1["Distance"].max(), telemetry_2["Distance"].max())

    distance = np.linspace(0, max_distance, n_points)

    d1_interpolated = pd.DataFrame({"Distance": distance})
    d2_interpolated = pd.DataFrame({"Distance": distance})

    for column in continuous_columns:
        d1_interpolated[column] = np.interp(
            distance, telemetry_1["Distance"], telemetry_1[column]
        )
        d2_interpolated[column] = np.interp(
            distance, telemetry_2["Distance"], telemetry_2[column]
        )

    for column in discrete_columns:
        d1_interpolator = interp1d(
            telemetry_1["Distance"], telemetry_1[column], kind="nearest"
        )
        d2_interpolator = interp1d(
            telemetry_2["Distance"], telemetry_2[column], kind="nearest"
        )

        d1_interpolated[column] = d1_interpolator(distance).astype(int)
        d2_interpolated[column] = d2_interpolator(distance).astype(int)

    d1_interpolated["RPM"] = d1_interpolated["RPM"].astype(int)
    d2_interpolated["RPM"] = d2_interpolated["RPM"].astype(int)

    # Time: own axis (fraction of the lap), not the common physical distance
    telemetry_1["Fraction"] = telemetry_1["Distance"] / telemetry_1["Distance"].iloc[-1]
    telemetry_2["Fraction"] = telemetry_2["Distance"] / telemetry_2["Distance"].iloc[-1]

    fraction_grid = np.linspace(0, 1, n_points)

    d1_interpolated["Time"] = np.interp(
        fraction_grid, telemetry_1["Fraction"], telemetry_1["Time"]
    )
    d2_interpolated["Time"] = np.interp(
        fraction_grid, telemetry_2["Fraction"], telemetry_2["Time"]
    )

    # Final check: ensure that the final distances and times match the expected lap times and distances
    distance_1_final = telemetry_1["Distance"].iloc[-1]
    distance_2_final = telemetry_2["Distance"].iloc[-1]
    diff = abs(distance_1_final - distance_2_final)

    time_1_final = d1_interpolated["Time"].iloc[-1]
    time_2_final = d2_interpolated["Time"].iloc[-1]

    delta_final = time_1_final - time_2_final
    delta_expected = lap_time_1 - lap_time_2

    print("\nInterpolation check:")
    print(f"Total distance driver 1: {distance_1_final:.3f} m")
    print(f"Total distance driver 2: {distance_2_final:.3f} m")
    print(f"Difference: {diff:.3f} m")
    print(
        f"Percentage difference: {diff / max(distance_1_final, distance_2_final) * 100:.3f}%"
    )

    if diff < 1.0:
        print("OK: distances near, alignment is a very close.")
    else:
        print("WARNING: distances differ noticeably, Time/Distance alignment is off.")

    print(
        f"\nTime 1 (interpolated): {time_1_final:.4f}  |  LapTime 1: {lap_time_1:.4f}"
    )
    print(f"Time 2 (interpolated): {time_2_final:.4f}  |  LapTime 2: {lap_time_2:.4f}")
    print(f"Delta (calculated): {delta_final:.4f}")
    print(f"Delta (expected):   {delta_expected:.4f}")

    if abs(delta_final - delta_expected) < 1e-6:
        print("OK: final delta matches the official lap time gap.")
    else:
        print("WARNING: final delta does not match the official lap time gap.")

    return d1_interpolated, d2_interpolated
