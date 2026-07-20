import numpy as np
import pandas as pd

def analyze_braking_zones(telemetry):

    telemetry = telemetry.reset_index(drop=True)

    is_braking = telemetry["Brake"] > 0

    groups = is_braking.ne(is_braking.shift()).cumsum()

    braking_zones = []

    for _, zone in telemetry[is_braking].groupby(groups):

        braking_zones.append({
            "StartDistance": zone["Distance"].iloc[0],
            "EndDistance": zone["Distance"].iloc[-1],
            "BrakingDuration": (zone["Time"].iloc[-1] - zone["Time"].iloc[0]),
            "BrakingDistance": (zone["Distance"].iloc[-1] - zone["Distance"].iloc[0]),
            "EntrySpeed": zone["Speed"].iloc[0],
            "MinimumSpeed": zone["Speed"].min(),
        })

    return pd.DataFrame(braking_zones)

def analyze_acceleration_zones(telemetry):

    telemetry = telemetry.reset_index(drop=True)

    is_accelerating = telemetry["Throttle"] > 0

    groups = is_accelerating.ne(is_accelerating.shift()).cumsum()

    acceleration_zones = []

    for _, zone in telemetry[is_accelerating].groupby(groups):

        full_throttle_points = zone[zone["Throttle"] == 100]

        time_to_full_throttle = None
        speed_gain_to_full_throttle = None

        if not full_throttle_points.empty:

            full_throttle = full_throttle_points.iloc[0]

            time_to_full_throttle = (full_throttle["Time"] - zone["Time"].iloc[0])

            speed_gain_to_full_throttle = (full_throttle["Speed"] - zone["Speed"].iloc[0])

        acceleration_zones.append({
            "StartDistance": zone["Distance"].iloc[0],
            "EndDistance": zone["Distance"].iloc[-1],
            "AccelerationDuration": (zone["Time"].iloc[-1] - zone["Time"].iloc[0]),
            "AccelerationDistance": (zone["Distance"].iloc[-1] - zone["Distance"].iloc[0]),
            "InitialSpeed": zone["Speed"].iloc[0],
            "TimeToFullThrottle": time_to_full_throttle,
            "SpeedGainToFullThrottle": speed_gain_to_full_throttle
        })

    return pd.DataFrame(acceleration_zones)

# Gearshift Features

def analyze_gear_shifts(telemetry):
    """
    Analyzes all gear shifts in a lap, providing context for each.

    Parameters
    ----------
    telemetry : pandas.DataFrame
        The telemetry data for a lap.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row is a gear shift event.
    """
    
    gear_changes = telemetry["nGear"].diff()

    shift_indices = gear_changes[gear_changes.abs() >= 1].index

    if shift_indices.empty:
        return pd.DataFrame()

    shifts = []
    for idx in shift_indices:
        if idx == 0:
            continue

        after_shift = telemetry.loc[idx]
        before_shift = telemetry.loc[idx - 1]
        
        change = after_shift['nGear'] - before_shift['nGear']
        direction = 'up' if change > 0 else 'down'

        shifts.append({
            'Distance': after_shift['Distance'],
            'Direction': direction,
            'GearFrom': before_shift['nGear'],
            'GearTo': after_shift['nGear'],
            'RPMbefore': before_shift['RPM'],
            'RPMafter': after_shift['RPM'],
            'Speed': after_shift['Speed'],
        })

    return pd.DataFrame(shifts)


def assign_braking_zones_to_corners(braking_zones, corners, braking_distance=300):

    assigned_zones = []
    used_indices = set()

    for _, corner in corners.iterrows():

        corner_number = corner["Number"]
        corner_distance = corner["Distance"]

        # Calculate distance from braking start to the corner
        distances = (corner_distance - braking_zones["StartDistance"])

        # Only consider unused braking zones before the corner
        valid_zones = braking_zones[
            (distances >= 0)
            & (distances <= braking_distance)
            & (~braking_zones.index.isin(used_indices))
        ]

        if valid_zones.empty:
            continue

        # Select the braking zone closest to the corner
        closest_idx = (corner_distance - valid_zones["StartDistance"]).idxmin()

        braking_zone = braking_zones.loc[closest_idx].copy()

        braking_zone["Corner"] = corner_number
        braking_zone["CornerDistance"] = corner_distance

        assigned_zones.append(braking_zone)
        used_indices.add(closest_idx)

    return pd.DataFrame(assigned_zones).reset_index(drop=True)