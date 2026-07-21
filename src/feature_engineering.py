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

def assign_braking_zones_to_turns(braking_zones, turns, braking_distance=300):

    assigned_zones = []
    used_indices = set()

    for _, turn in turns.iterrows():

        turn_number = turn["Number"]
        turn_distance = turn["Distance"]

        # Calculate distance from braking start to the turn
        distances = (turn_distance - braking_zones["StartDistance"])

        # Only consider unused braking zones before the turn
        valid_zones = braking_zones[
            (distances >= 0)
            & (distances <= braking_distance)
            & (~braking_zones.index.isin(used_indices))
        ]

        if valid_zones.empty:
            continue

        # Select the braking zone closest to the turn
        closest_idx = (turn_distance - valid_zones["StartDistance"]).idxmin()

        braking_zone = braking_zones.loc[closest_idx].copy()

        braking_zone["Turn"] = turn_number

        assigned_zones.append(braking_zone)
        used_indices.add(closest_idx)

    assigned_zones_df = pd.DataFrame(assigned_zones).reset_index(drop=True)
    assigned_zones_df = assigned_zones_df[["Turn"] + [column for column in assigned_zones_df.columns if column != "Turn"]]
    assigned_zones_df["Turn"] = assigned_zones_df["Turn"].astype(int)
    
    return assigned_zones_df


def analyze_acceleration_zones(telemetry, braking_zones, full_throttle_threshold=95):

    telemetry = telemetry.reset_index(drop=True)

    acceleration_zones = []

    for _, braking_zone in braking_zones.iterrows():

        braking_end_distance = braking_zone["EndDistance"]

        # To identify an acceleration zone, we look for the apex (minimum speed)
        # in the corner between two braking zones.

        next_braking_zones = braking_zones[braking_zones["StartDistance"] > braking_end_distance]

        if not next_braking_zones.empty:

            next_braking_start = next_braking_zones["StartDistance"].min()

            apex_search_window = telemetry[(telemetry["Distance"] > braking_end_distance) & (telemetry["Distance"] < next_braking_start)]

        else:

            apex_search_window = telemetry[telemetry["Distance"] > braking_end_distance]

        if apex_search_window.empty:
            continue

        # The start of acceleration is the apex (minimum speed)
        acceleration_start = apex_search_window.loc[apex_search_window["Speed"].idxmin()]

        # Get telemetry until the next braking zone
        if not next_braking_zones.empty:

            acceleration_telemetry = telemetry[(telemetry["Distance"] >= acceleration_start["Distance"]) & (telemetry["Distance"] < next_braking_start)]

        else:

            acceleration_telemetry = telemetry[telemetry["Distance"] >= acceleration_start["Distance"]]

        if acceleration_telemetry.empty:
            continue

        # Find first point of full throttle
        full_throttle_points = acceleration_telemetry[acceleration_telemetry["Throttle"] >= full_throttle_threshold]

        time_to_full_throttle = None
        distance_to_full_throttle = None
        speed_gain_to_full_throttle = None

        if not full_throttle_points.empty:

            full_throttle = full_throttle_points.iloc[0]

            time_to_full_throttle = (full_throttle["Time"] - acceleration_start["Time"])

            distance_to_full_throttle = (full_throttle["Distance"] - acceleration_start["Distance"])

            speed_gain_to_full_throttle = (full_throttle["Speed"] - acceleration_start["Speed"])

        acceleration_zones.append({
            "Turn": braking_zone["Turn"],
            "StartDistance": acceleration_start["Distance"],
            "EndDistance": acceleration_telemetry["Distance"].iloc[-1],
            "AccelerationDuration": (acceleration_telemetry["Time"].iloc[-1] - acceleration_start["Time"]),
            "AccelerationDistance": (acceleration_telemetry["Distance"].iloc[-1] - acceleration_start["Distance"]),
            "ExitSpeed": acceleration_start["Speed"],
            "TimeToFullThrottle": time_to_full_throttle,
            "DistanceToFullThrottle": distance_to_full_throttle,
            "SpeedGainToFullThrottle": speed_gain_to_full_throttle
        })

    acceleration_zones_df = pd.DataFrame(acceleration_zones).reset_index(drop=True)
    acceleration_zones_df["Turn"] = acceleration_zones_df["Turn"].astype(int)

    return acceleration_zones_df

def analyze_throttle_lifts(
    telemetry,
    acceleration_zones,
    full_throttle_threshold=95,
    lift_margin=50
):

    telemetry = telemetry.reset_index(drop=True)
    acceleration_zones = acceleration_zones.copy()

    number_of_lifts = []
    maximum_throttle_reductions = []

    for _, acceleration_zone in acceleration_zones.iterrows():

        acceleration_telemetry = telemetry[
            (telemetry["Distance"] >= acceleration_zone["StartDistance"])
            & (telemetry["Distance"] <= acceleration_zone["EndDistance"])
        ]

        if acceleration_telemetry.empty:

            number_of_lifts.append(0)
            maximum_throttle_reductions.append(0)

            continue

        end_distance = acceleration_telemetry["Distance"].iloc[-1]

        lift_telemetry = acceleration_telemetry[
            acceleration_telemetry["Distance"] <= end_distance - lift_margin
        ]

        lifts = 0
        maximum_reduction = 0

        in_full_throttle = False
        lift_active = False
        previous_throttle = None
        current_lift_minimum = None

        for throttle in lift_telemetry["Throttle"]:

            if previous_throttle is None:

                previous_throttle = throttle
                continue

            if throttle >= full_throttle_threshold:

                if lift_active and current_lift_minimum is not None:
                    maximum_reduction = max(maximum_reduction, 100 - current_lift_minimum)

                in_full_throttle = True
                lift_active = False
                current_lift_minimum = None

            elif in_full_throttle:

                lift_active = True

                if current_lift_minimum is None:
                    current_lift_minimum = throttle
                else:
                    current_lift_minimum = min(current_lift_minimum, throttle)

                if previous_throttle >= full_throttle_threshold:
                    lifts += 1

            previous_throttle = throttle

        if lift_active and current_lift_minimum is not None:
            maximum_reduction = max(maximum_reduction, 100 - current_lift_minimum)

        number_of_lifts.append(lifts)
        maximum_throttle_reductions.append(maximum_reduction)

    acceleration_zones["NumberOfThrottleLifts"] = number_of_lifts
    acceleration_zones["MaximumThrottleReduction"] = maximum_throttle_reductions

    return acceleration_zones

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


