import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from .config import *


def choose_color(driver):
    """Assigns a consistent color to a driver based on their abbreviation.

    This function maps specific driver abbreviations to predefined colors. If a driver's
    abbreviation is not found in the `colors` dictionary, it defaults to black.

    Parameters:
        driver (str): The three-letter abbreviation of the driver (e.g., "VER", "HAM").

    Returns:
        str: The hexadecimal color string associated with the driver. Defaults to "black"
             if the driver abbreviation is not recognized.
    """

    return COLORS.get(driver, "black")  # Default to black if driver not found


def plot_variable_comparison(lap_1, lap_2, variable, turns, y_unit=None):
    """Plot a telemetry variable against distance for two drivers.

    Generates a line plot comparing `variable` between two driver laps over distance
    and marks turn locations on the x-axis for context.

    Parameters:
        lap_1 (dict): First driver's lap payload containing at least keys 'Telemetry' and 'Driver'.
        lap_2 (dict): Second driver's lap payload containing at least keys 'Telemetry' and 'Driver'.
        variable (str): Telemetry column name to plot (e.g., 'Speed', 'RPM').
        turns (pandas.DataFrame): DataFrame with 'Number' and 'Distance' columns for turns.
        y_unit (str, optional): Unit string for y-axis label (e.g., 'km/h'). Defaults to None.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """

    lap_1_telemetry = lap_1["Telemetry"]
    lap_2_telemetry = lap_2["Telemetry"]
    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    title = f"{variable} of {driver_1} and {driver_2} throughout the lap"

    plt.figure(figsize=(12, 6))

    if driver_1 is not None:
        plt.plot(
            lap_1_telemetry["Distance"],
            lap_1_telemetry[variable],
            label=driver_1,
            color=choose_color(driver_1),
            alpha=1,
            linewidth=0.8,
        )

    if driver_2 is not None:
        plt.plot(
            lap_2_telemetry["Distance"],
            lap_2_telemetry[variable],
            label=driver_2,
            color=choose_color(driver_2),
            alpha=1,
            linewidth=0.8,
        )

    plt.xlabel("Distance (m)")
    plt.ylabel(f"{variable} ({y_unit})" if y_unit is not None else variable)
    plt.title(title)

    plt.legend()
    plt.grid()

    # Set x-axis ticks to turn numbers
    plt.xticks(ticks=turns["Distance"], labels=turns["Number"])
    plt.xlabel("Turn Number")

    fig = plt.gcf()
    plt.show()
    return fig


def plot_variable_delta(lap_1, lap_2, variable, turns, y_unit=None):
    """Plot the delta of a telemetry variable between two drivers over distance.

    Computes the pointwise difference of `variable` between `lap_1` and `lap_2` and
    plots it, using turn distances as x-axis markers.

    Parameters:
        lap_1 (dict): First driver's lap payload containing 'Telemetry' and 'Driver'.
        lap_2 (dict): Second driver's lap payload containing 'Telemetry' and 'Driver'.
        variable (str): Telemetry column to compute the delta for.
        turns (pandas.DataFrame): Turn DataFrame with 'Number' and 'Distance'.
        y_unit (str, optional): Unit string for y-axis label. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """

    lap_1_telemetry = lap_1["Telemetry"]
    lap_2_telemetry = lap_2["Telemetry"]
    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    delta_variable = lap_1_telemetry[variable] - lap_2_telemetry[variable]

    plt.figure(figsize=(12, 6))
    plt.plot(
        lap_1_telemetry["Distance"],
        delta_variable,
        label=f"{driver_1} - {driver_2}",
        color="skyblue",
    )

    plt.xlabel("Distance (m)")
    plt.ylabel(
        f" {variable} Delta ({y_unit})" if y_unit is not None else f" {variable} Delta"
    )
    plt.title(f"{variable} Delta between {driver_1} and {driver_2}")

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.legend()
    plt.grid()

    # Set x-axis ticks to turn numbers
    plt.xticks(ticks=turns["Distance"], labels=turns["Number"])
    plt.xlabel("Turn Number")

    fig = plt.gcf()
    plt.show()
    return fig


def barplot_feature_comparison(lap_1, lap_2, feature, y_unit=None):
    """Bar plot comparison of a segment/turn feature between two drivers.

    Parameters:
        lap_1 (dict): First driver's lap payload with 'Segments' or 'Turns' DataFrame.
        lap_2 (dict): Second driver's lap payload with 'Segments' or 'Turns' DataFrame.
        feature (str): Column name of the feature to compare.
        y_unit (str, optional): Unit string for y-axis label. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    lap_1_features = (
        lap_1["Segments"] if feature in lap_1["Segments"].columns else lap_1["Turns"]
    )
    lap_2_features = (
        lap_2["Segments"] if feature in lap_2["Segments"].columns else lap_2["Turns"]
    )

    x_column = lap_1_features.columns[0]

    comparison = lap_1_features[[x_column, feature]].copy()

    comparison = comparison.merge(
        lap_2_features[[x_column, feature]],
        on=x_column,
        how="left",
        suffixes=(f"_{driver_1}", f"_{driver_2}"),
    )

    x = np.arange(len(comparison))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        comparison[f"{feature}_{driver_1}"],
        width,
        label=driver_1,
        color=choose_color(driver_1),
    )

    plt.bar(
        x + width / 2,
        comparison[f"{feature}_{driver_2}"],
        width,
        label=driver_2,
        color=choose_color(driver_2),
    )

    plt.xticks(x, comparison[x_column])

    plt.xlabel(x_column)

    if y_unit is not None:

        plt.ylabel(f"{feature} ({y_unit})")

    else:

        plt.ylabel(feature.replace("_", " "))

    plt.title(feature.replace("_", " "))

    plt.legend()

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    fig = plt.gcf()

    plt.show()

    return fig


def barplot_feature_delta(lap_1, lap_2, feature, y_unit=None):
    """Bar plot of the delta for a segment/turn feature between two drivers.

    Parameters:
        lap_1 (dict): First driver's lap payload with 'Segments' or 'Turns' DataFrame.
        lap_2 (dict): Second driver's lap payload with 'Segments' or 'Turns' DataFrame.
        feature (str): Column name of the feature to compute the delta for.
        y_unit (str, optional): Unit string for y-axis label. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    lap_1_features = (
        lap_1["Segments"] if feature in lap_1["Segments"].columns else lap_1["Turns"]
    )
    lap_2_features = (
        lap_2["Segments"] if feature in lap_2["Segments"].columns else lap_2["Turns"]
    )

    x_column = lap_1_features.columns[0]

    comparison = lap_1_features[[x_column, feature]].copy()

    comparison = comparison.merge(
        lap_2_features[[x_column, feature]],
        on=x_column,
        how="left",
        suffixes=(f"_{driver_1}", f"_{driver_2}"),
    )

    comparison["Delta"] = comparison[f"{feature}_{driver_1}"].fillna(0) - comparison[
        f"{feature}_{driver_2}"
    ].fillna(0)

    plt.figure(figsize=(10, 4))

    plt.axhline(0, color="black", linewidth=1)

    plt.bar(comparison[x_column], comparison["Delta"], color="skyblue")

    plt.xlabel(x_column)

    if y_unit is not None:

        plt.ylabel(f"{feature} ({y_unit})")

    else:

        plt.ylabel(feature.replace("_", " "))

    plt.title(f"{feature} ({driver_1} - {driver_2})")

    plt.xticks(comparison[x_column])

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    fig = plt.gcf()

    plt.show()

    return fig


def calculate_feature_correlation(
    lap_1,
    lap_2,
    x_feature,
    y_feature,
):
    """Calculate Pearson and Spearman correlations between feature deltas.

    Computes the delta for `x_feature` and `y_feature` between two drivers and
    returns Pearson and Spearman correlation statistics.

    Parameters:
        lap_1 (dict): First driver's lap payload with 'Segments' or 'Turns' DataFrame.
        lap_2 (dict): Second driver's lap payload with 'Segments' or 'Turns' DataFrame.
        x_feature (str): Column name used as the X feature.
        y_feature (str): Column name used as the Y feature.

    Returns:
        pandas.DataFrame: DataFrame containing correlation coefficients and p-values.
    """

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    if x_feature in lap_1["Segments"].columns:
        dataframe_1 = lap_1["Segments"]
        dataframe_2 = lap_2["Segments"]
        index_column = "Segment"
    else:
        dataframe_1 = lap_1["Turns"]
        dataframe_2 = lap_2["Turns"]
        index_column = "Turn"

    features_1 = dataframe_1[
        [
            index_column,
            x_feature,
            y_feature,
        ]
    ].copy()

    features_2 = dataframe_2[
        [
            index_column,
            x_feature,
            y_feature,
        ]
    ].copy()

    comparison = features_1.merge(
        features_2,
        on=index_column,
        suffixes=(
            f"_{driver_1}",
            f"_{driver_2}",
        ),
    )

    braking_features = {
        "BrakingPoint",
        "BrakingDistance",
        "BrakingDuration",
    }

    if x_feature in braking_features or y_feature in braking_features:
        comparison = comparison.fillna(0)

    comparison["DeltaX"] = (
        comparison[f"{x_feature}_{driver_1}"] - comparison[f"{x_feature}_{driver_2}"]
    )

    comparison["DeltaY"] = (
        comparison[f"{y_feature}_{driver_1}"] - comparison[f"{y_feature}_{driver_2}"]
    )

    comparison = comparison.dropna()

    if len(comparison) < 2:
        return pd.DataFrame(
            {
                "FeatureX": [x_feature],
                "FeatureY": [y_feature],
                "Pearson": [np.nan],
                "PearsonP": [np.nan],
                "Spearman": [np.nan],
                "SpearmanP": [np.nan],
            }
        )

    pearson_corr, pearson_p = pearsonr(
        comparison["DeltaX"],
        comparison["DeltaY"],
    )

    spearman_corr, spearman_p = spearmanr(
        comparison["DeltaX"],
        comparison["DeltaY"],
    )

    return pd.DataFrame(
        {
            "FeatureX": [x_feature],
            "FeatureY": [y_feature],
            "Pearson": [pearson_corr],
            "PearsonP": [pearson_p],
            "Spearman": [spearman_corr],
            "SpearmanP": [spearman_p],
        }
    )


def scatterplot_features_relationship(
    lap_1,
    lap_2,
    x_feature,
    y_feature,
):
    """Scatter plot of delta relationship between two features across segments/turns.

    Parameters:
        lap_1 (dict): First driver's lap payload with 'Segments' or 'Turns' DataFrame.
        lap_2 (dict): Second driver's lap payload with 'Segments' or 'Turns' DataFrame.
        x_feature (str): Feature to use on the x-axis.
        y_feature (str): Feature to use on the y-axis.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    if x_feature in lap_1["Segments"].columns:
        dataframe_1 = lap_1["Segments"]
        dataframe_2 = lap_2["Segments"]
        index_column = "Segment"
    else:
        dataframe_1 = lap_1["Turns"]
        dataframe_2 = lap_2["Turns"]
        index_column = "Turn"

    features_1 = dataframe_1[
        [
            index_column,
            x_feature,
            y_feature,
        ]
    ].copy()

    features_2 = dataframe_2[
        [
            index_column,
            x_feature,
            y_feature,
        ]
    ].copy()

    comparison = features_1.merge(
        features_2,
        on=index_column,
        suffixes=(
            f"_{driver_1}",
            f"_{driver_2}",
        ),
    )

    comparison["DeltaX"] = (
        comparison[f"{x_feature}_{driver_1}"] - comparison[f"{x_feature}_{driver_2}"]
    )

    comparison["DeltaY"] = (
        comparison[f"{y_feature}_{driver_1}"] - comparison[f"{y_feature}_{driver_2}"]
    )

    comparison = comparison.dropna()

    plt.figure(figsize=(7, 6))

    plt.scatter(
        comparison["DeltaX"],
        comparison["DeltaY"],
        s=150,
        color="skyblue",
    )

    for _, row in comparison.iterrows():

        plt.text(
            row["DeltaX"],
            row["DeltaY"],
            str(int(row[index_column])),
            ha="center",
            va="center",
        )

    plt.axhline(
        0,
        color="black",
        linewidth=1,
    )

    plt.axvline(
        0,
        color="black",
        linewidth=1,
    )

    plt.xlabel(f"Delta {x_feature.replace('_', ' ')} " f"({driver_1} - {driver_2})")

    plt.ylabel(f"Delta {y_feature.replace('_', ' ')} " f"({driver_1} - {driver_2})")

    plt.title(
        f"Delta {y_feature.replace('_', ' ')} vs "
        f"Delta {x_feature.replace('_', ' ')}"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    fig = plt.gcf()

    plt.show()

    return fig


def compare_times(lap_1, lap_2):
    """Compare segment and turn times between two drivers.

    Creates and displays DataFrames with segment times, cumulative delta, and
    turn times deltas for the two drivers.

    Parameters:
        lap_1 (dict): First driver's lap payload with 'Segments' and 'Turns' DataFrames.
        lap_2 (dict): Second driver's lap payload with 'Segments' and 'Turns' DataFrames.

    Returns:
        None
    """

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    segment_df = pd.DataFrame(
        {
            "Segment": lap_1["Segments"]["Segment"].values,
            driver_1: lap_1["Segments"]["SegmentTime"].values,
            driver_2: lap_2["Segments"]["SegmentTime"].values,
        }
    )

    segment_df["Delta"] = segment_df[driver_1] - segment_df[driver_2]
    segment_df["CumulativeDelta"] = segment_df["Delta"].cumsum()

    turn_df = pd.DataFrame(
        {
            "Turn": lap_1["Turns"]["Turn"].values,
            driver_1: lap_1["Turns"]["TurnTime"].values,
            driver_2: lap_2["Turns"]["TurnTime"].values,
        }
    )

    turn_df["Delta"] = turn_df[driver_1] - turn_df[driver_2]

    display(segment_df)
    display(turn_df)


def compare_features(lap_1, lap_2, features):
    """Compare selected features between two drivers and display paired table.

    Parameters:
        lap_1 (dict): First driver's lap payload with 'Segments' or 'Turns'.
        lap_2 (dict): Second driver's lap payload with 'Segments' or 'Turns'.
        features (list[str]): List of feature column names to compare.

    Returns:
        None
    """

    feature_type = "Segments" if features[0] in lap_1["Segments"].columns else "Turns"

    label = feature_type[:-1] if feature_type.endswith("s") else feature_type

    df1 = lap_1[feature_type][[label] + features].copy()
    df2 = lap_2[feature_type][[label] + features].copy()

    df1 = df1.rename(
        columns={feature: f"{feature}_{lap_1['Driver']}" for feature in features}
    )

    df2 = df2.rename(
        columns={feature: f"{feature}_{lap_2['Driver']}" for feature in features}
    )

    result = df1.merge(df2, on=label, how="outer")

    columns = [label]

    for feature in features:
        columns.extend([f"{feature}_{lap_1['Driver']}", f"{feature}_{lap_2['Driver']}"])

    display(result[columns])

