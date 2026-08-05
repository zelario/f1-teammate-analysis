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
    """Plots a specified telemetry variable against distance for two drivers.

    This function generates a line plot comparing a chosen telemetry variable (e.g., Speed, RPM)
    between two drivers over the course of a lap, represented by distance. It highlights turn
    numbers on the x-axis for better contextualization.

    Parameters:
        lap_1_telemetry (pandas.DataFrame): Telemetry data for the first driver for a specific lap.
                                            Must contain 'Distance' and the specified `variable` columns.
        lap_2_telemetry (pandas.DataFrame): Telemetry data for the second driver for a specific lap.
                                            Must contain 'Distance' and the specified `variable` columns.
        variable (str): The name of the telemetry column to plot on the y-axis (e.g., "Speed", "RPM").
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number'
                                  and 'Distance' columns. These distances are used for x-axis ticks.
        driver_1 (str, optional): The abbreviation of the first driver. Used for plot labels and
                                  color assignment. Defaults to None.
        driver_2 (str, optional): The abbreviation of the second driver. Used for plot labels and
                                  color assignment. Defaults to None.
        y_unit (str, optional): The unit of the variable being plotted (e.g., "km/h", "RPM").
                                 Appears in the y-axis label. Defaults to None.
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
    """Plots the difference in a telemetry variable between two drivers against distance.

    This function calculates the delta (difference) of a specified telemetry variable
    between two drivers and plots this delta over the course of a lap, represented by distance.
    Turn numbers are used as x-axis ticks for easy reference.

    Parameters:
        lap_1_telemetry (pandas.DataFrame): Telemetry data for the first driver for a specific lap.
                                            Must contain 'Distance' and the specified `variable` columns.
        lap_2_telemetry (pandas.DataFrame): Telemetry data for the second driver for a specific lap.
                                            Must contain 'Distance' and the specified `variable` columns.
        driver_1 (str): The abbreviation of the first driver. Used for plot labels.
        driver_2 (str): The abbreviation of the second driver. Used for plot labels.
        variable (str): The name of the telemetry column to calculate the delta for and plot
                        on the y-axis (e.g., "Speed", "Brake").
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number'
                                  and 'Distance' columns. These distances are used for x-axis ticks.
        y_unit (str, optional): The unit of the variable being plotted (e.g., "km/h", "%").
                                 Appears in the y-axis label. Defaults to None.
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

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]
    
    lap_1_features = lap_1["Segments"] if feature in lap_1["Segments"].columns else lap_1["Turns"]
    lap_2_features = lap_2["Segments"] if feature in lap_2["Segments"].columns else lap_2["Turns"]

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

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    lap_1_features = lap_1["Segments"] if feature in lap_1["Segments"].columns else lap_1["Turns"]
    lap_2_features = lap_2["Segments"] if feature in lap_2["Segments"].columns else lap_2["Turns"]

    x_column = lap_1_features.columns[0]

    comparison = lap_1_features[[x_column, feature]].copy()

    comparison = comparison.merge(
        lap_2_features[[x_column, feature]],
        on=x_column,
        how="left",
        suffixes=(f"_{driver_1}", f"_{driver_2}"),
    )

    comparison["Delta"] = (
        comparison[f"{feature}_{driver_1}"].fillna(0)
        - comparison[f"{feature}_{driver_2}"].fillna(0)
    )

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
    """Calculates Pearson and Spearman correlation between feature deltas."""

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    features_1 = lap_1["Segments"][
        [
            "Segment",
            x_feature,
            y_feature,
        ]
    ].copy()

    features_2 = lap_2["Segments"][
        [
            "Segment",
            x_feature,
            y_feature,
        ]
    ].copy()

    comparison = features_1.merge(
        features_2,
        on="Segment",
        suffixes=(
            f"_{driver_1}",
            f"_{driver_2}",
        ),
    )

    comparison["DeltaX"] = (
        comparison[f"{x_feature}_{driver_1}"]
        -
        comparison[f"{x_feature}_{driver_2}"]
    )

    comparison["DeltaY"] = (
        comparison[f"{y_feature}_{driver_1}"]
        -
        comparison[f"{y_feature}_{driver_2}"]
    )

    comparison = comparison.dropna()

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
            "Pearson": [round(pearson_corr, 3)],
            "PearsonP": [round(pearson_p, 5)],
            "Spearman": [round(spearman_corr, 3)],
            "SpearmanP": [round(spearman_p, 5)],
        }
    )


def scatterplot_features_relationship(
    lap_1,
    lap_2,
    x_feature,
    y_feature,
):
    """Generates a scatter plot showing the relationship between feature deltas."""

    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

    features_1 = lap_1["Segments"][
        [
            "Segment",
            x_feature,
            y_feature,
        ]
    ].copy()

    features_2 = lap_2["Segments"][
        [
            "Segment",
            x_feature,
            y_feature,
        ]
    ].copy()

    comparison = features_1.merge(
        features_2,
        on="Segment",
        suffixes=(
            f"_{driver_1}",
            f"_{driver_2}",
        ),
    )

    comparison["DeltaX"] = (
        comparison[f"{x_feature}_{driver_1}"]
        -
        comparison[f"{x_feature}_{driver_2}"]
    )

    comparison["DeltaY"] = (
        comparison[f"{y_feature}_{driver_1}"]
        -
        comparison[f"{y_feature}_{driver_2}"]
    )

    plt.figure(
        figsize=(7, 6)
    )

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
            str(int(row["Segment"])),
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

    plt.xlabel(
        f"Delta {x_feature.replace('_', ' ')} "
        f"({driver_1} - {driver_2})"
    )

    plt.ylabel(
        f"Delta {y_feature.replace('_', ' ')} "
        f"({driver_1} - {driver_2})"
    )

    plt.title(
        f"Delta {y_feature.replace('_', ' ')} vs "
        f"Delta {x_feature.replace('_', ' ')}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    fig = plt.gcf()

    plt.show()

    return fig