import numpy as np
import pandas as pd
from .explore_data import calculate_feature_correlation


def find_priority_zones(lap_1, lap_2, feature_type):
    """
    Identify the track zones with the largest time differences.

    Parameters
    lap_1, lap_2 (dict): Lap data for the two drivers.
    feature_type (str): Either "Turns" or "Segments".

    Returns
    pd.DataFrame: Zones ranked by absolute time delta.
    """

    if feature_type == "Turns":
        time_column = "TurnTime"
        label_column = "Turn"
    elif feature_type == "Segments":
        time_column = "SegmentTime"
        label_column = "Segment"
    else:
        raise ValueError("feature_type must be 'Turns' or 'Segments'.")

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    dataframe_1 = lap_1[feature_type][[label_column, time_column]].copy()
    dataframe_2 = lap_2[feature_type][[label_column, time_column]].copy()

    comparison = dataframe_1.merge(
        dataframe_2,
        on=label_column,
        suffixes=(f"_{driver_1}", f"_{driver_2}"),
    )

    comparison["Delta"] = (
        comparison[f"{time_column}_{driver_1}"]
        - comparison[f"{time_column}_{driver_2}"]
    )

    return comparison.loc[
        comparison["Delta"].abs().sort_values(ascending=False).index
    ].reset_index(drop=True)


def find_feature_contributors(
    lap_1,
    lap_2,
    priority_zones,
    feature_type,
):
    """
    Identify possible feature contributors within priority zones.

    Parameters:
    lap_1, lap_2 (dict): Lap data for the two drivers.
    priority_zones (pd.DataFrame): Output from find_priority_zones().
    feature_type (str): Either "Turns" or "Segments".
    """

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    if feature_type == "Turns":
        label_column = "Turn"
        time_column = "TurnTime"
    elif feature_type == "Segments":
        label_column = "Segment"
        time_column = "SegmentTime"
    else:
        raise ValueError("feature_type must be 'Turns' or 'Segments'.")

    dataframe_1 = lap_1[feature_type].copy()
    dataframe_2 = lap_2[feature_type].copy()

    features = [
        feature
        for feature in dataframe_1.columns
        if feature in dataframe_2.columns
        and feature not in {label_column, time_column}
        and pd.api.types.is_numeric_dtype(dataframe_1[feature])
        and pd.api.types.is_numeric_dtype(dataframe_2[feature])
    ]

    comparison = dataframe_1.merge(
        dataframe_2,
        on=label_column,
        suffixes=(f"_{driver_1}", f"_{driver_2}"),
    )

    comparison["TimeDelta"] = (
        comparison[f"{time_column}_{driver_1}"]
        - comparison[f"{time_column}_{driver_2}"]
    )

    # Calculate the average absolute delta for each feature
    magnitude_reference = {}

    for feature in features:
        delta = (
            comparison[f"{feature}_{driver_1}"] - comparison[f"{feature}_{driver_2}"]
        )
        magnitude_reference[feature] = delta.abs().mean()

    # Calculate correlations between feature deltas and time deltas
    correlation_results = {}

    for feature in features:

        correlation_dataframe = calculate_feature_correlation(
            lap_1,
            lap_2,
            feature,
            time_column,
        )

        correlation_row = correlation_dataframe.iloc[0]

        correlation_results[feature] = {
            "Pearson": correlation_row["Pearson"],
            "PearsonP": correlation_row["PearsonP"],
            "Spearman": correlation_row["Spearman"],
            "SpearmanP": correlation_row["SpearmanP"],
        }

    results = []

    for _, zone in priority_zones.iterrows():

        zone_id = zone[label_column]

        zone_data = comparison[comparison[label_column] == zone_id]

        if zone_data.empty:
            continue

        time_delta = zone_data["TimeDelta"].iloc[0]

        for feature in features:

            value_1 = zone_data[f"{feature}_{driver_1}"].iloc[0]

            value_2 = zone_data[f"{feature}_{driver_2}"].iloc[0]

            if pd.isna(value_1) or pd.isna(value_2):
                continue

            delta = value_1 - value_2
            abs_delta = abs(delta)

            mean_abs_delta = magnitude_reference[feature]

            # Normalize the feature difference
            if mean_abs_delta > 0:
                magnitude_score = abs_delta / mean_abs_delta
            else:
                magnitude_score = 0.0

            correlations = correlation_results[feature]

            pearson = correlations["Pearson"]
            pearson_p = correlations["PearsonP"]
            spearman = correlations["Spearman"]
            spearman_p = correlations["SpearmanP"]

            valid_correlations = [
                abs(value) for value in [pearson, spearman] if not pd.isna(value)
            ]

            correlation_strength = (
                np.mean(valid_correlations) if valid_correlations else 0.0
            )

            valid_p_values = [
                value for value in [pearson_p, spearman_p] if not pd.isna(value)
            ]

            statistical_confidence = 1 - max(valid_p_values) if valid_p_values else 0.0

            # Check whether the observed feature difference agrees  with the direction of the correlation and time delta
            if delta != 0 and time_delta != 0 and correlation_strength > 0:
                expected_time_direction = np.sign(
                    delta * (pearson if not pd.isna(pearson) else spearman)
                )

                direction_consistency = (
                    1.0 if np.sign(time_delta) == expected_time_direction else 0.0
                )
                
            else:
                direction_consistency = 0.0

            contributor_score = (
                magnitude_score
                * (0.5 + 0.5 * correlation_strength)
                * statistical_confidence
                * direction_consistency
            )

            if abs_delta == 0:
                direction = "similar"
            elif delta > 0:
                direction = f"higher for {driver_1}"
            else:
                direction = f"lower for {driver_1}"

            results.append(
                {
                    label_column: zone_id.astype(int),
                    "Feature": feature,
                    "Delta": delta,
                    "MeanAbsDelta": mean_abs_delta,
                    "Pearson": pearson,
                    "PearsonP": pearson_p,
                    "Spearman": spearman,
                    "SpearmanP": spearman_p,
                    "CorrelationStrength": correlation_strength,
                    "Confidence": statistical_confidence,
                    "MagnitudeScore": magnitude_score,
                    "DirectionConsistency": direction_consistency,
                    "ContributorScore": contributor_score,
                    "Direction": direction,
                }
            )

    return (
        pd.DataFrame(results)
        .sort_values(
            [label_column, "ContributorScore"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def calculate_contributor_summary(
    priority_zones,
    contributors,
    feature_type,
):
    """
    Select the strongest possible contributor for each priority zone.

    Parameters:
    priority_zones (pd.DataFrame): Output from find_priority_zones().
    contributors (pd.DataFrame): Output from find_feature_contributors().
    feature_type (str): Either "Turns" or "Segments".

    Returns:
    pd.DataFrame: One summary row per priority zone.
    """

    if feature_type == "Turns":
        label_column = "Turn"
    elif feature_type == "Segments":
        label_column = "Segment"
    else:
        raise ValueError("feature_type must be 'Turns' or 'Segments'.")

    summaries = []

    for _, zone in priority_zones.iterrows():

        zone_id = zone[label_column]

        zone_contributors = contributors[contributors[label_column] == zone_id]

        if zone_contributors.empty:
            continue

        for _, contributor in zone_contributors.head(5).iterrows():

            summaries.append(
                {
                    label_column: zone_id.astype(int),
                    "TimeDelta": zone["Delta"],
                    "PossibleMainContributor": contributor["Feature"],
                    "ContributorScore": contributor["ContributorScore"],
                    "Direction": contributor["Direction"],
                }
            )

    return pd.DataFrame(summaries)
