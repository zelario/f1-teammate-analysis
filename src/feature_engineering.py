import pandas as pd
import numpy as np


def analyze_time_turns(
    lap,
    turns,
    braking_distance=300,
    entry_distance=50,
    exit_distance=50,
):
    """
    Calculates turn time.

    If a braking zone exists:
        TurnTime = Braking Point -> Exit

    Otherwise:
        TurnTime = Entry Point -> Exit
    """

    telemetry = lap["Telemetry"].reset_index(drop=True)

    is_braking = telemetry["Brake"] > 0

    groups = is_braking.ne(is_braking.shift()).cumsum()

    braking_zones = []

    for _, zone in telemetry[is_braking].groupby(groups):

        braking_zones.append(
            {
                "BrakingPointDistance": zone["Distance"].iloc[0],
            }
        )

    braking_zones = pd.DataFrame(braking_zones).reset_index(drop=True)

    corner_times = []

    used_indices = set()

    for _, turn in turns.iterrows():

        turn_number = int(turn["Number"])
        turn_distance = turn["Distance"]

        distances = turn_distance - braking_zones["BrakingPointDistance"]

        valid_zones = braking_zones[
            (distances >= 0)
            & (distances <= braking_distance)
            & (~braking_zones.index.isin(used_indices))
        ]

        if valid_zones.empty:

            start_distance = turn_distance - entry_distance

        else:

            closest_idx = (turn_distance - valid_zones["BrakingPointDistance"]).idxmin()

            start_distance = braking_zones.loc[
                closest_idx,
                "BrakingPointDistance",
            ]

            used_indices.add(closest_idx)

        start_row = telemetry.iloc[
            (telemetry["Distance"] - start_distance).abs().argmin()
        ]

        exit_distance_target = turn_distance + exit_distance

        end_row = telemetry.iloc[
            (telemetry["Distance"] - exit_distance_target).abs().argmin()
        ]

        corner_times.append(
            {
                "Turn": turn_number,
                "TurnTime": end_row["Time"] - start_row["Time"],
            }
        )

    corner_times_df = (
        pd.DataFrame(corner_times).sort_values("Turn").reset_index(drop=True)
    )

    corner_times_df["Turn"] = corner_times_df["Turn"].astype(int)

    return corner_times_df


def analyze_speed_turns(
    lap, turns, entry_distance=50, exit_distance=50, minimum_speed_window=50
):
    """Analyzes speed metrics around each turn, including entry, minimum, and exit speeds.

    This function processes telemetry data to calculate key speed-related features for each
    turn. It determines the speed at a specified distance before the turn (entry speed),
    the minimum speed within a window around the turn, and the speed at a specified
    distance after the turn (exit speed).

    Parameters:
        telemetry (pandas.DataFrame): Telemetry data for a lap, must include 'Distance' and 'Speed' columns.
        turns (pandas.DataFrame): DataFrame with turn information, must include 'Number' and 'Distance' columns.
        entry_distance (int, optional): The distance before the turn to measure entry speed. Defaults to 50.
        exit_distance (int, optional): The distance after the turn to measure exit speed. Defaults to 50.
        minimum_speed_window (int, optional): The window size around the turn to find the minimum speed. Defaults to 50.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'Turn', 'EntrySpeed', 'MinimumSpeed', and 'ExitSpeed'
                          for each turn.
    """

    telemetry = lap["Telemetry"]

    telemetry = telemetry.reset_index(drop=True)

    corner_features = []

    for _, turn in turns.iterrows():

        turn_number = turn["Number"]
        turn_distance = turn["Distance"]

        # Get entry speed
        entry_distance_target = turn_distance - entry_distance

        entry_telemetry = telemetry[telemetry["Distance"] <= entry_distance_target]

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

        exit_telemetry = telemetry[telemetry["Distance"] >= exit_distance_target]

        if exit_telemetry.empty:
            exit_speed = None

        else:

            exit_point = exit_telemetry.iloc[
                (exit_telemetry["Distance"] - exit_distance_target).abs().argmin()
            ]

            exit_speed = exit_point["Speed"]

        corner_features.append(
            {
                "Turn": turn_number,
                "EntrySpeed": entry_speed,
                "ApexSpeed": minimum_speed,
                "ExitSpeed": exit_speed,
            }
        )

    corner_features_df = pd.DataFrame(corner_features).reset_index(drop=True)
    corner_features_df["Turn"] = corner_features_df["Turn"].astype(int)

    return corner_features_df


def analyze_braking_turns(lap, turns, braking_distance=300):
    """
    Identifies braking zones and assigns each one to the nearest following turn.
    """

    telemetry = lap["Telemetry"].reset_index(drop=True)

    is_braking = telemetry["Brake"] > 0

    groups = is_braking.ne(is_braking.shift()).cumsum()

    braking_zones = []

    for _, zone in telemetry[is_braking].groupby(groups):

        braking_zones.append(
            {
                "BrakingPointDistance": zone["Distance"].iloc[0],
                "BrakingDuration": (zone["Time"].iloc[-1] - zone["Time"].iloc[0]),
                "BrakingDistance": (
                    zone["Distance"].iloc[-1] - zone["Distance"].iloc[0]
                ),
            }
        )

    braking_zones = pd.DataFrame(braking_zones).reset_index(drop=True)

    assigned_zones = []

    used_indices = set()

    for _, turn in turns.iterrows():

        turn_number = int(turn["Number"])
        turn_distance = turn["Distance"]

        distances = turn_distance - braking_zones["BrakingPointDistance"]

        valid_zones = braking_zones[
            (distances >= 0)
            & (distances <= braking_distance)
            & (~braking_zones.index.isin(used_indices))
        ]

        if valid_zones.empty:

            continue

        closest_idx = (turn_distance - valid_zones["BrakingPointDistance"]).idxmin()

        braking_zone = braking_zones.loc[closest_idx]

        assigned_zones.append(
            {
                "Turn": turn_number,
                "BrakingPoint": (turn_distance - braking_zone["BrakingPointDistance"]),
                "BrakingDistance": braking_zone["BrakingDistance"],
                "BrakingDuration": braking_zone["BrakingDuration"],
            }
        )

        used_indices.add(closest_idx)

    assigned_turns = {zone["Turn"] for zone in assigned_zones}

    for _, turn in turns.iterrows():

        turn_number = int(turn["Number"])

        if turn_number in assigned_turns:
            continue

        assigned_zones.append(
            {
                "Turn": turn_number,
                "BrakingPoint": np.nan,
                "BrakingDistance": np.nan,
                "BrakingDuration": np.nan,
            }
        )

    assigned_zones_df = (
        pd.DataFrame(assigned_zones).sort_values("Turn").reset_index(drop=True)
    )

    assigned_zones_df["Turn"] = assigned_zones_df["Turn"].astype(int)

    return assigned_zones_df


def analyze_time_segments(lap, turns):
    """
    Calculates segment times using interpolated telemetry timestamps
    at each turn distance.
    """

    telemetry = lap["Telemetry"].reset_index(drop=True)

    telemetry = telemetry.sort_values("Distance")

    segment_distances = [0] + turns["Distance"].tolist()

    # Interpolate the exact timestamp at each turn distance
    boundary_times = np.interp(
        segment_distances, telemetry["Distance"], telemetry["Time"]
    )

    segment_times = []

    for i in range(len(segment_distances)):
        start_distance = segment_distances[i]
        start_time = boundary_times[i]

        if i < len(segment_distances) - 1:
            end_distance = segment_distances[i + 1]
            end_time = boundary_times[i + 1]
        else:
            end_distance = telemetry["Distance"].iloc[-1]
            end_time = telemetry["Time"].iloc[-1]

        segment_times.append(
            {
                "Segment": i,
                "StartDistance": start_distance,
                "EndDistance": end_distance,
                "SegmentTime": end_time - start_time,
            }
        )

    return pd.DataFrame(segment_times)


def analyze_speed_segments(
    lap,
    turns,
):

    telemetry = lap["Telemetry"].reset_index(drop=True)

    segments = []

    segment_distances = [0] + turns["Distance"].tolist()

    for i in range(len(segment_distances)):

        start_distance = segment_distances[i]

        if i < len(segment_distances) - 1:
            end_distance = segment_distances[i + 1]
        else:
            end_distance = float("inf")

        segment_window = telemetry[
            (telemetry["Distance"] >= start_distance)
            & (telemetry["Distance"] < end_distance)
        ]

        if segment_window.empty:

            mean_speed = np.nan
            std_speed = np.nan
            maximum_speed = np.nan
            minimum_speed = np.nan

        else:

            mean_speed = segment_window["Speed"].mean()
            std_speed = segment_window["Speed"].std()
            maximum_speed = segment_window["Speed"].max()
            minimum_speed = segment_window["Speed"].min()

        segments.append(
            {
                "Segment": i,
                "MeanSpeed": mean_speed,
                "StdSpeed": std_speed,
                "MaximumSpeed": maximum_speed,
                "MinimumSpeed": minimum_speed,
            }
        )

    return pd.DataFrame(segments)


def analyze_throttle_segments(lap, turns, full_throttle_threshold=95):
    """Analyzes throttle application in segments between turns.

    This function divides the lap into segments based on turn locations and calculates
    throttle-related metrics for each segment. These metrics include mean throttle,
    standard deviation of throttle, and the percentage of time spent at full throttle.

    Parameters:
        telemetry (pandas.DataFrame): Telemetry data for a lap, must include 'Distance' and 'Throttle' columns.
        turns (pandas.DataFrame): DataFrame with turn information, must include 'Number' and 'Distance' columns.
        full_throttle_threshold (int, optional): The throttle percentage to be considered as full throttle.
                                                 Defaults to 95.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'Turn', 'MeanThrottle', 'StdThrottle',
                          and 'FullThrottlePercentage' for each segment.
    """

    telemetry = lap["Telemetry"].reset_index(drop=True)

    turns = turns.copy()

    start_turn = pd.DataFrame([{"Number": 0, "Distance": telemetry["Distance"].min()}])

    turns = pd.concat([start_turn, turns], ignore_index=True)

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

        full_throttle = throttle >= full_throttle_threshold

        throttle_features.append(
            {
                "Segment": turn["Number"],
                "MeanThrottle": throttle.mean(),
                "StdThrottle": throttle.std(),
                "FullThrottlePercentage": (full_throttle.mean() * 100),
            }
        )

    throttle_features_df = pd.DataFrame(throttle_features).reset_index(drop=True)
    throttle_features_df["Segment"] = throttle_features_df["Segment"].astype(int)

    return throttle_features_df


def analyze_gear_shifts_segments(lap, turns):
    """Analyzes gear shifting patterns and RPM metrics within each track segment.

    This function segments the lap based on turn locations and calculates various metrics
    related to gear shifts and RPM within each segment. This includes the number of shifts,
    lowest and highest gears used, and statistics about RPM (min, max, mean, std).

    Parameters:
        telemetry (pandas.DataFrame): Telemetry data for a lap, must include 'Distance', 'nGear', and 'RPM' columns.
        turns (pandas.DataFrame): DataFrame with turn information, must include 'Distance' column.

    Returns:
        pandas.DataFrame: A DataFrame with columns for each segment's 'Turn', 'NumberOfShifts', 'LowestGear',
                          'HighestGear', 'MinimumRPM', 'MaximumRPM', 'MeanRPM', and 'StdRPM'.
    """

    telemetry = lap["Telemetry"].reset_index(drop=True)

    gear_changes = telemetry["nGear"].diff()

    telemetry["GearChange"] = gear_changes.abs() >= 1

    segment_features = []

    segment_distances = [0] + turns["Distance"].tolist()

    for i in range(len(segment_distances)):

        turn_number = i

        start_distance = segment_distances[i]

        if i < len(segment_distances) - 1:
            end_distance = segment_distances[i + 1]

        else:
            end_distance = float("inf")

        segment_telemetry = telemetry[
            (telemetry["Distance"] >= start_distance)
            & (telemetry["Distance"] < end_distance)
        ]

        if segment_telemetry.empty:
            continue

        segment_shifts = segment_telemetry[segment_telemetry["GearChange"]]

        segment_features.append(
            {
                "Segment": turn_number,
                "NumberOfShifts": (segment_shifts["GearChange"].sum()),
                "LowestGear": (segment_telemetry["nGear"].min()),
                "HighestGear": (segment_telemetry["nGear"].max()),
                "MinimumRPM": (segment_telemetry["RPM"].min()),
                "MaximumRPM": (segment_telemetry["RPM"].max()),
                "MeanRPM": (segment_telemetry["RPM"].mean()),
                "StdRPM": (segment_telemetry["RPM"].std()),
            }
        )

    return pd.DataFrame(segment_features).reset_index(drop=True)


def build_segments_dataframe(
    times,
    speed,
    throttle,
    gear_shifts,
):
    """
    Merges every segment-level feature into a single dataframe.

    Parameters:
        segment_times (pd.DataFrame)
        speed_metrics (pd.DataFrame)
        braking_zones (pd.DataFrame)
        throttle_segments (pd.DataFrame)
        gear_shifts (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    segments = times.copy()

    segments = segments.merge(
        speed,
        on="Segment",
        how="left",
    )

    segments = segments.merge(
        throttle,
        on="Segment",
        how="left",
    )

    segments = segments.merge(
        gear_shifts,
        on="Segment",
        how="left",
    )

    segments = segments.sort_values("Segment").reset_index(drop=True)

    return segments


def build_turns_dataframe(
    times,
    speed,
    braking,
):
    """
    Merges all turn-level features into a single dataframe.

    Parameters
    ----------
    speed : pd.DataFrame
        Speed-related features for each turn.
    braking : pd.DataFrame
        Braking-related features for each turn.

    Returns
    -------
    pd.DataFrame
        Combined turn-level feature dataframe.
    """

    turn_features = times.merge(
        braking,
        on="Turn",
        how="left",
    )

    turn_features = turn_features.merge(
        speed,
        on="Turn",
        how="left",
    )

    turn_features = turn_features.sort_values("Turn").reset_index(drop=True)

    return turn_features
