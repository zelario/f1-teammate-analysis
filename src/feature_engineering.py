import pandas as pd

def analyze_speed_metrics(telemetry, turns, entry_distance=50, exit_distance=50, minimum_speed_window=50):

    telemetry = telemetry.reset_index(drop=True)

    corner_features = []

    for _, turn in turns.iterrows():

        turn_number = turn["Number"]
        turn_distance = turn["Distance"]

        # Get entry speed
        entry_distance_target = turn_distance - entry_distance

        entry_telemetry = telemetry[
            telemetry["Distance"] <= entry_distance_target
        ]

        if entry_telemetry.empty:
            entry_speed = None

        else:

            entry_point = entry_telemetry.iloc[
                (entry_telemetry["Distance"] - entry_distance_target).abs().argmin()
            ]

            entry_speed = entry_point["Speed"]

        # Get minimum speed around the corner
        minimum_speed_telemetry = telemetry[
            (telemetry["Distance"] >= turn_distance - minimum_speed_window)
            & (telemetry["Distance"] <= turn_distance + minimum_speed_window)
        ]

        if minimum_speed_telemetry.empty:
            minimum_speed = None

        else:
            minimum_speed = minimum_speed_telemetry["Speed"].min()

        # Get exit speed
        exit_distance_target = turn_distance + exit_distance

        exit_telemetry = telemetry[
            telemetry["Distance"] >= exit_distance_target
        ]

        if exit_telemetry.empty:
            exit_speed = None

        else:

            exit_point = exit_telemetry.iloc[
                (exit_telemetry["Distance"] - exit_distance_target).abs().argmin()
            ]

            exit_speed = exit_point["Speed"]

        corner_features.append({
            "Turn": turn_number,
            "EntrySpeed": entry_speed,
            "MinimumSpeed": minimum_speed,
            "ExitSpeed": exit_speed
        })

    corner_features_df = pd.DataFrame(corner_features).reset_index(drop=True)
    corner_features_df["Turn"] = corner_features_df["Turn"].astype(int)

    return corner_features_df


def analyze_braking_zones(telemetry, turns, braking_distance=300):

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
        })
        
    braking_zones = pd.DataFrame(braking_zones).reset_index(drop=True)

    # Assign braking zones to turns
    assigned_zones = []
    used_indices = set()

    for _, turn in turns.iterrows():

        turn_number = turn["Number"]
        turn_distance = turn["Distance"]

        distances = turn_distance - braking_zones["StartDistance"]

        valid_zones = braking_zones[
            (distances >= 0)
            & (distances <= braking_distance)
            & (~braking_zones.index.isin(used_indices))
        ]

        if valid_zones.empty:
            continue

        closest_idx = (turn_distance - valid_zones["StartDistance"]).idxmin()

        braking_zone = braking_zones.loc[closest_idx].copy()

        braking_zone["Turn"] = turn_number

        braking_zone["BrakingPoint"] = (
            turn_distance - braking_zone["StartDistance"]
        )

        assigned_zones.append(braking_zone)
        used_indices.add(closest_idx)

    assigned_zones_df = pd.DataFrame(assigned_zones).reset_index(drop=True)

    assigned_zones_df = assigned_zones_df[
        ["Turn"] + [
            column for column in assigned_zones_df.columns
            if column != "Turn"
        ]
    ]

    assigned_zones_df["Turn"] = assigned_zones_df["Turn"].astype(int)

    return assigned_zones_df


def analyze_throttle_segments(telemetry, turns, full_throttle_threshold=95):

    telemetry = telemetry.reset_index(drop=True)

    turns = turns.copy()

    start_turn = pd.DataFrame([{
        "Number": 0,
        "Distance": telemetry["Distance"].min()
    }])

    turns = pd.concat(
        [start_turn, turns],
        ignore_index=True
    )

    throttle_features = []

    for i in range(len(turns)):

        turn = turns.iloc[i]

        start_distance = turn["Distance"]

        if i + 1 < len(turns):

            end_distance = turns.iloc[i + 1]["Distance"]

        else:

            end_distance = telemetry["Distance"].max()

        segment = telemetry[
            (telemetry["Distance"] >= start_distance)
            & (telemetry["Distance"] < end_distance)
        ]

        if segment.empty:

            continue

        throttle = segment["Throttle"]

        full_throttle = (
            throttle >= full_throttle_threshold
        )

        throttle_features.append({
            "Turn": turn["Number"],
            "StartDistance": start_distance,
            "EndDistance": end_distance,
            "MeanThrottle": throttle.mean(),
            "StdThrottle": throttle.std(),
            "FullThrottlePercentage": (full_throttle.mean() * 100)
        })
        
    throttle_features_df = pd.DataFrame(throttle_features).reset_index(drop=True)
    throttle_features_df["Turn"] = throttle_features_df["Turn"].astype(int)

    return throttle_features_df


def analyze_gear_shifts(telemetry):

    telemetry = telemetry.reset_index(drop=True)

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

        change = after_shift["nGear"] - before_shift["nGear"]

        direction = "up" if change > 0 else "down"

        rpm_change = after_shift["RPM"] - before_shift["RPM"]

        shifts.append({
            "Distance": after_shift["Distance"],
            "Direction": direction,
            "GearFrom": before_shift["nGear"],
            "GearTo": after_shift["nGear"],
            "RPMbefore": before_shift["RPM"],
            "RPMafter": after_shift["RPM"],
            "RPMChange": rpm_change,
        })

    shifts_df = pd.DataFrame(shifts).reset_index(drop=True)

    shifts_df["GearFrom"] = shifts_df["GearFrom"].astype(int)
    shifts_df["GearTo"] = shifts_df["GearTo"].astype(int)
    shifts_df["RPMbefore"] = shifts_df["RPMbefore"].astype(int)
    shifts_df["RPMafter"] = shifts_df["RPMafter"].astype(int)
    shifts_df["RPMChange"] = shifts_df["RPMChange"].astype(int)

    return shifts_df