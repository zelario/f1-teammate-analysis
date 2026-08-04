import pandas as pd


def analyze_speed_segments(
    lap,
    turns,
    entry_distance=50,
    exit_distance=50,
    minimum_speed_window=50,
):
    """Analyzes speed metrics for each segment."""

    telemetry = lap["Telemetry"].reset_index(drop=True)

    speed_metrics = []

    for _, turn in turns.iterrows():

        turn_number = int(turn["Number"])
        turn_distance = turn["Distance"]

        segment = turn_number - 1

        entry_distance_target = turn_distance - entry_distance

        entry_telemetry = telemetry[telemetry["Distance"] <= entry_distance_target]

        if entry_telemetry.empty:

            entry_speed = None

        else:

            entry_point = entry_telemetry.iloc[
                (entry_telemetry["Distance"] - entry_distance_target).abs().argmin()
            ]

            entry_speed = entry_point["Speed"]

        speed_window = telemetry[
            (telemetry["Distance"] >= turn_distance - minimum_speed_window)
            & (telemetry["Distance"] <= turn_distance + minimum_speed_window)
        ]

        if speed_window.empty:

            minimum_speed = None
            mean_speed = None
            speed_std = None

        else:

            minimum_speed = speed_window["Speed"].min()
            mean_speed = speed_window["Speed"].mean()
            speed_std = speed_window["Speed"].std()

        exit_distance_target = turn_distance + exit_distance

        exit_telemetry = telemetry[telemetry["Distance"] >= exit_distance_target]

        if exit_telemetry.empty:

            exit_speed = None

        else:

            exit_point = exit_telemetry.iloc[
                (exit_telemetry["Distance"] - exit_distance_target).abs().argmin()
            ]

            exit_speed = exit_point["Speed"]

        speed_metrics.append(
            {
                "Segment": segment,
                "ExitSpeed": round(exit_speed, 3) if exit_speed is not None else None,
                "MinimumSpeed": (
                    round(minimum_speed, 3) if minimum_speed is not None else None
                ),
                "MeanSpeed": round(mean_speed, 3) if mean_speed is not None else None,
                "StdSpeed": round(speed_std, 3) if speed_std is not None else None,
                "EntrySpeed": (
                    round(entry_speed, 3) if entry_speed is not None else None
                ),                
            }
        )

    speed_metrics_df = pd.DataFrame(speed_metrics)

    speed_metrics_df["Segment"] = speed_metrics_df["Segment"].astype(int)

    return speed_metrics_df


def analyze_braking_segments(
    lap,
    turns,
    braking_distance=300,
):
    """Analyzes braking metrics for each segment."""

    telemetry = lap["Telemetry"].reset_index(drop=True)

    is_braking = telemetry["Brake"] > 0

    groups = is_braking.ne(
        is_braking.shift()
    ).cumsum()

    braking_zones = []

    for _, zone in telemetry[is_braking].groupby(groups):

        braking_zones.append(
            {
                "StartDistance": zone["Distance"].iloc[0],
                "BrakingDuration": (
                    zone["Time"].iloc[-1]
                    - zone["Time"].iloc[0]
                ),
                "BrakingDistance": (
                    zone["Distance"].iloc[-1]
                    - zone["Distance"].iloc[0]
                ),
            }
        )

    braking_zones = pd.DataFrame(
        braking_zones
    )

    braking_metrics = []

    used_indices = set()

    for _, turn in turns.iterrows():

        turn_number = int(
            turn["Number"]
        )

        segment = turn_number - 1

        turn_distance = turn["Distance"]

        distances = (
            turn_distance
            - braking_zones["StartDistance"]
        )

        valid_zones = braking_zones[
            (distances >= 0)
            &
            (distances <= braking_distance)
            &
            (~braking_zones.index.isin(used_indices))
        ]

        if valid_zones.empty:

            braking_metrics.append(
                {
                    "Segment": segment,
                    "BrakingPoint": 0,
                    "BrakingDistance": 0,
                    "BrakingDuration": 0,
                }
            )

            continue

        closest_idx = (
            turn_distance
            - valid_zones["StartDistance"]
        ).idxmin()

        braking_zone = braking_zones.loc[
            closest_idx
        ]

        braking_metrics.append(
            {
                "Segment": segment,
                "BrakingPoint": (
                    turn_distance
                    - braking_zone["StartDistance"]
                ),
                "BrakingDistance": (
                    braking_zone["BrakingDistance"]
                ),
                "BrakingDuration": (
                    braking_zone["BrakingDuration"]
                ),
            }
        )

        used_indices.add(
            closest_idx
        )

    braking_metrics_df = pd.DataFrame(
        braking_metrics
    )

    braking_metrics_df["Segment"] = (
        braking_metrics_df["Segment"]
        .astype(int)
    )

    braking_metrics_df = (
        braking_metrics_df
        .sort_values("Segment")
        .reset_index(drop=True)
    )

    return braking_metrics_df


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


def analyze_time_segments(lap, turns):
    """Analyzes and calculates the time spent in each segment between turns.

    This function takes telemetry data and turn information to calculate the time duration
    for each segment of the track defined by the turns.

    Parameters:
        telemetry (pandas.DataFrame): Telemetry data for a specific lap.
                                      Must contain 'Distance' and 'Time' columns.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Distance' column.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'Turn', 'StartDistance', 'EndDistance', and 'SegmentTime',
                          representing the time taken for each segment.
    """

    telemetry = lap["Telemetry"]

    telemetry = telemetry.reset_index(drop=True)

    segment_distances = [0] + turns["Distance"].tolist()

    segment_times = []

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

        start_time = segment_telemetry["Time"].iloc[0]

        end_time = segment_telemetry["Time"].iloc[-1]

        segment_times.append(
            {
                "Segment": turn_number,
                "StartDistance": start_distance,
                "EndDistance": end_distance,
                "SegmentTime": end_time - start_time,
            }
        )

    return pd.DataFrame(segment_times).reset_index(drop=True)


def build_segments_dataframe(
    segment_times,
    speed_metrics,
    braking_zones,
    throttle_segments,
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

    segments = segment_times.copy()

    segments = segments.merge(
        speed_metrics,
        on="Segment",
        how="left",
    )

    segments = segments.merge(
        braking_zones,
        on="Segment",
        how="left",
    )

    segments = segments.merge(
        throttle_segments,
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
