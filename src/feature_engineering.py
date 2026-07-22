import pandas as pd


def analyze_corners(telemetry, turns, entry_distance=50, exit_distance=50, minimum_speed_window=50):

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
        })

    return pd.DataFrame(braking_zones)


def assign_braking_zones_to_turns(braking_zones, turns, braking_distance=300):

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


def analyze_acceleration_zones(telemetry, braking_zones, full_throttle_threshold=95):

    telemetry = telemetry.reset_index(drop=True)

    acceleration_zones = []

    for _, braking_zone in braking_zones.iterrows():

        braking_end_distance = braking_zone["EndDistance"]

        # To identify the apex (minimum speed) in the corner between two braking zones

        next_braking_zones = braking_zones[
            braking_zones["StartDistance"] > braking_end_distance
        ]

        if not next_braking_zones.empty:

            next_braking_start = next_braking_zones["StartDistance"].min()

            apex_search_window = telemetry[
                (telemetry["Distance"] > braking_end_distance)
                & (telemetry["Distance"] < next_braking_start)
            ]

        else:

            apex_search_window = telemetry[
                telemetry["Distance"] > braking_end_distance
            ]

        if apex_search_window.empty:
            continue

        # The start of acceleration is the apex (point of minimum speed)

        acceleration_start = apex_search_window.loc[
            apex_search_window["Speed"].idxmin()
        ]

        # Get telemetry until the next braking zone

        if not next_braking_zones.empty:

            acceleration_telemetry = telemetry[
                (telemetry["Distance"] >= acceleration_start["Distance"])
                & (telemetry["Distance"] < next_braking_start)
            ]

        else:

            acceleration_telemetry = telemetry[
                telemetry["Distance"] >= acceleration_start["Distance"]
            ]

        if acceleration_telemetry.empty:
            continue

        # Find first point of full throttle

        full_throttle_points = acceleration_telemetry[
            acceleration_telemetry["Throttle"] >= full_throttle_threshold
        ]

        time_to_full_throttle = None
        distance_to_full_throttle = None
        speed_gain_to_full_throttle = None

        if not full_throttle_points.empty:

            full_throttle = full_throttle_points.iloc[0]

            time_to_full_throttle = (
                full_throttle["Time"]
                - acceleration_start["Time"]
            )

            distance_to_full_throttle = (
                full_throttle["Distance"]
                - acceleration_start["Distance"]
            )

            speed_gain_to_full_throttle = (
                full_throttle["Speed"]
                - acceleration_start["Speed"]
            )

        acceleration_zones.append({
            "Turn": braking_zone["Turn"],
            "StartDistance": acceleration_start["Distance"],
            "EndDistance": acceleration_telemetry["Distance"].iloc[-1],
            "AccelerationDuration": acceleration_telemetry["Time"].iloc[-1] - acceleration_start["Time"],
            "AccelerationDistance": acceleration_telemetry["Distance"].iloc[-1] - acceleration_start["Distance"],
            "TimeToFullThrottle": time_to_full_throttle,
            "DistanceToFullThrottle": distance_to_full_throttle,
            "SpeedGainToFullThrottle": speed_gain_to_full_throttle
        })

    acceleration_zones_df = pd.DataFrame(acceleration_zones).reset_index(drop=True)
    acceleration_zones_df["Turn"] = acceleration_zones_df["Turn"].astype(int)

    return acceleration_zones_df


def analyze_throttle_lifts(telemetry, acceleration_zones, full_throttle_threshold=95, lift_distance_threshold=50):

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
            acceleration_telemetry["Distance"] <= end_distance - lift_distance_threshold
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

                    maximum_reduction = max(
                        maximum_reduction,
                        100 - current_lift_minimum
                    )

                in_full_throttle = True
                lift_active = False
                current_lift_minimum = None

            elif in_full_throttle:

                lift_active = True

                if current_lift_minimum is None:
                    current_lift_minimum = throttle

                else:
                    current_lift_minimum = min(
                        current_lift_minimum,
                        throttle
                    )

                if previous_throttle >= full_throttle_threshold:
                    lifts += 1

            previous_throttle = throttle

        if lift_active and current_lift_minimum is not None:

            maximum_reduction = max(
                maximum_reduction,
                100 - current_lift_minimum
            )

        number_of_lifts.append(lifts)
        maximum_throttle_reductions.append(maximum_reduction)

    acceleration_zones["NumberOfThrottleLifts"] = number_of_lifts
    acceleration_zones["MaximumThrottleReduction"] = maximum_throttle_reductions

    return acceleration_zones


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