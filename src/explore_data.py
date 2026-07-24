import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def choose_color(driver):
    """
    Choose a color based on the driver's abbreviation.

    Parameters
    ----------
    driver : str
        Driver's abbreviation.

    Returns
    -------
    str
        Color associated with the driver.
    """

    colors = {
        "VER": "midnightblue",
        "TSU": "firebrick",
        "LEC": "red",
        "HAM": "yellow",
        "NOR": "darkorange",
        "PIA": "gray",
        "RUS": "darkturquoise",
        "ANT": "indianred",
        "ALO": "seagreen",
        "STR": "dimgray",
    }

    return colors.get(driver, "black")  # Default to black if driver not found
    

def plot_variable_comparison(lap_1_telemetry, lap_2_telemetry, variable, turns, driver_1= None, driver_2= None, y_unit=None):
    """
    Plot a telemetry variable against distance for two drivers.

    Parameters
    ----------
    lap_1_telemetry : pandas.DataFrame
        First driver's telemetry for a specific lap.

    lap_2_telemetry : pandas.DataFrame
        Second driver's telemetry for a specific lap.

    driver_1 : str
        First driver's abbreviation.

    driver_2 : str
        Second driver's abbreviation.

    variable : str
        Column to plot on the y-axis.
        
    turns : pandas.DataFrame
        DataFrame with turn information ('Number', 'Distance').
    """

    title = f"{variable} of {driver_1} and {driver_2} throughout the lap"

    plt.figure(figsize=(12, 6))

    if driver_1 is not None:
        plt.plot(
            lap_1_telemetry["Distance"],
            lap_1_telemetry[variable],
            label=driver_1,
            color=choose_color(driver_1),
            alpha=1,
            linewidth=0.8
        )

    if driver_2 is not None:
        plt.plot(
            lap_2_telemetry["Distance"],
            lap_2_telemetry[variable],
            label=driver_2,
            color=choose_color(driver_2),
            alpha=1,
            linewidth=0.8
        )

    plt.xlabel("Distance (m)")
    plt.ylabel(f"{variable} ({y_unit})" if y_unit is not None else variable)
    plt.title(title)

    plt.legend()
    plt.grid()

    # Set x-axis ticks to turn numbers
    plt.xticks(ticks=turns['Distance'], labels=turns['Number'])
    plt.xlabel("Turn Number")

    plt.show()

def plot_variable_delta(lap_1_telemetry, lap_2_telemetry, driver_1, driver_2, variable, turns, y_unit=None):
    """
    Plot the difference in a telemetry variable between two drivers against distance.

    Parameters
    ----------
    lap_1_telemetry : pandas.DataFrame
        First driver's telemetry for a specific lap.

    lap_2_telemetry : pandas.DataFrame
        Second driver's telemetry for a specific lap.

    variable : str
        Column to plot on the y-axis.

    driver_1 : str
        First driver's abbreviation.

    driver_2 : str
        Second driver's abbreviation.
    """

    delta_variable = lap_1_telemetry[variable] - lap_2_telemetry[variable]

    plt.figure(figsize=(12, 6))
    plt.plot(
        lap_1_telemetry["Distance"],
        delta_variable,
        label=f"{driver_1} - {driver_2}",
        color="skyblue"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel(f" {variable} Delta ({y_unit})" if y_unit is not None else f" {variable} Delta")
    plt.title(f"{variable} Delta between {driver_1} and {driver_2}")

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid()

    # Set x-axis ticks to turn numbers
    plt.xticks(ticks=turns['Distance'], labels=turns['Number'])
    plt.xlabel("Turn Number")

    plt.show()


# Feature analysis functions


def barplot_feature_comparison(lap_1_feature, lap_2_feature, driver_1, driver_2, turns, feature, y_unit=None, include_zero=False):

    turn_numbers = turns["Number"].tolist()

    if include_zero:

        turn_numbers = [0] + turn_numbers

    comparison = pd.DataFrame({
        "Turn": turn_numbers
    })

    comparison = comparison.merge(
        lap_1_feature[["Turn", feature]],
        on="Turn",
        how="left"
    )

    comparison = comparison.merge(
        lap_2_feature[["Turn", feature]],
        on="Turn",
        how="left",
        suffixes=(f"_{driver_1}", f"_{driver_2}")
    )

    x = np.arange(len(comparison))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        comparison[f"{feature}_{driver_1}"],
        width,
        label=driver_1,
        color=choose_color(driver_1)
    )

    plt.bar(
        x + width / 2,
        comparison[f"{feature}_{driver_2}"],
        width,
        label=driver_2,
        color=choose_color(driver_2)
    )

    plt.xticks(x, comparison["Turn"])

    plt.xlabel("Turn")

    if y_unit is not None:
        plt.ylabel(f"{feature} ({y_unit})")
    else:
        plt.ylabel(feature.replace("_", " "))

    plt.title(feature.replace("_", " "))

    plt.legend()

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
def barplot_feature_delta(lap_1_feature, lap_2_feature, driver_1, driver_2, turns, feature, y_unit=None, include_zero=False):

    turn_numbers = turns["Number"].tolist()

    if include_zero:

        turn_numbers = [0] + turn_numbers

    comparison = pd.DataFrame({
        "Turn": turn_numbers
    })

    comparison = comparison.merge(
        lap_1_feature[["Turn", feature]],
        on="Turn",
        how="left"
    )

    comparison = comparison.merge(
        lap_2_feature[["Turn", feature]],
        on="Turn",
        how="left",
        suffixes=(f"_{driver_1}", f"_{driver_2}")
    )

    comparison["Delta"] = (
        comparison[f"{feature}_{driver_1}"]
        - comparison[f"{feature}_{driver_2}"]
    )

    plt.figure(figsize=(10, 4))

    plt.axhline(0, color="black", linewidth=1)

    plt.bar(
        comparison["Turn"],
        comparison["Delta"],
        color="skyblue"
    )

    plt.xlabel("Turn")

    if y_unit is not None:
        plt.ylabel(f"{feature} ({y_unit})")
    else:
        plt.ylabel(feature.replace("_", " "))

    plt.title(f"{feature} ({driver_1} - {driver_2})")

    plt.xticks(comparison["Turn"])

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    

def scatterplot_gear_shifts(gear_shifts, driver, turns):

    plt.figure(figsize=(12, 5))

    upshifts = gear_shifts[
        gear_shifts["Direction"] == "up"
    ]

    downshifts = gear_shifts[
        gear_shifts["Direction"] == "down"
    ]

    plt.scatter(
        upshifts["Distance"],
        upshifts["GearTo"],
        label="Upshift",
        marker="^",
        s=70,
        color="green"
    )

    plt.scatter(
        downshifts["Distance"],
        downshifts["GearTo"],
        label="Downshift",
        marker="v",
        s=70,
        color="red"
    )

    plt.xticks(
        turns["Distance"],
        turns["Number"]
    )

    plt.yticks(
        range(1, 9)
    )

    plt.xlabel("Turn")

    plt.ylabel("Gear")

    plt.title(
        f"Gear Shifts - {driver}"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

def scatterplot_shift_rpm(gear_shifts, driver, turns):

    plt.figure(figsize=(14, 8))

    for _, shift in gear_shifts.iterrows():

        if shift["Direction"] == "up":

            color = "green"
            marker = "^"

        else:

            color = "red"
            marker = "v"

        plt.plot(
            [shift["Distance"], shift["Distance"]],
            [shift["RPMbefore"], shift["RPMafter"]],
            color=color,
            alpha=0.5,
            linewidth=1
        )

        plt.scatter(
            shift["Distance"],
            shift["RPMbefore"],
            color=color,
            marker=marker,
            s=70
        )

        plt.scatter(
            shift["Distance"],
            shift["RPMafter"],
            facecolors="none",
            edgecolors=color,
            marker=marker,
            s=70
        )

    plt.xticks(
        turns["Distance"],
        turns["Number"]
    )

    plt.ylim(
        7000,
        13000
    )

    plt.xlabel("Turn")

    plt.ylabel("RPM")

    plt.title(
        f"RPM Before and After Gear Shifts - {driver}"
    )

    plt.scatter(
        [],
        [],
        color="green",
        marker="^",
        label="Upshift"
    )

    plt.scatter(
        [],
        [],
        color="red",
        marker="v",
        label="Downshift"
    )

    plt.scatter(
        [],
        [],
        color="black",
        marker="^",
        label="RPM Before"
    )

    plt.scatter(
        [],
        [],
        facecolors="none",
        edgecolors="black",
        marker="^",
        label="RPM After"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()
    
    
def scatterplot_features_relationship(
    lap_1,
    lap_2,
    driver_1,
    driver_2,
    x_dataframe,
    x_feature,
    y_dataframe,
    y_feature
):

    x_comparison = lap_1[x_dataframe][
        ["Turn", x_feature]
    ].merge(
        lap_2[x_dataframe][
            ["Turn", x_feature]
        ],
        on="Turn",
        suffixes=(
            f"_{driver_1}",
            f"_{driver_2}"
        )
    )

    y_comparison = lap_1[y_dataframe][
        ["Turn", y_feature]
    ].merge(
        lap_2[y_dataframe][
            ["Turn", y_feature]
        ],
        on="Turn",
        suffixes=(
            f"_{driver_1}",
            f"_{driver_2}"
        )
    )

    comparison = x_comparison.merge(
        y_comparison,
        on="Turn"
    )

    comparison["DeltaX"] = (
        comparison[
            f"{x_feature}_{driver_1}"
        ]
        -
        comparison[
            f"{x_feature}_{driver_2}"
        ]
    )

    comparison["DeltaY"] = (
        comparison[
            f"{y_feature}_{driver_1}"
        ]
        -
        comparison[
            f"{y_feature}_{driver_2}"
        ]
    )

    plt.figure(figsize=(7, 6))

    plt.scatter(
        comparison["DeltaX"],
        comparison["DeltaY"],
        s=150,
        color="skyblue"
    )

    for _, row in comparison.iterrows():

        plt.text(
            row["DeltaX"],
            row["DeltaY"],
            str(int(row["Turn"])),
            ha="center",
            va="center"
        )

    plt.axhline(
        0,
        color="black",
        linewidth=1
    )

    plt.axvline(
        0,
        color="black",
        linewidth=1
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

    plt.show()
    
    
# Performance comparison functions

def analyze_segment_times(telemetry, turns):

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

        segment_times.append({
            "Turn": turn_number,
            "StartDistance": start_distance,
            "EndDistance": end_distance,
            "SegmentTime": end_time - start_time
        })

    return pd.DataFrame(
        segment_times
    ).reset_index(drop=True)
    

def barplot_segment_time_delta(segment_times_1, segment_times_2, driver_1, driver_2, turns):

    turn_numbers = [0] + turns["Number"].tolist()

    comparison = pd.DataFrame({
        "Turn": turn_numbers
    })

    comparison = comparison.merge(
        segment_times_1[["Turn", "SegmentTime"]],
        on="Turn",
        how="left"
    )

    comparison = comparison.merge(
        segment_times_2[["Turn", "SegmentTime"]],
        on="Turn",
        how="left",
        suffixes=(f"_{driver_1}", f"_{driver_2}")
    )

    comparison["Delta"] = (
        comparison[f"SegmentTime_{driver_1}"]
        -
        comparison[f"SegmentTime_{driver_2}"]
    )

    plt.figure(figsize=(10, 4))

    plt.axhline(
        0,
        color="black",
        linewidth=1
    )

    plt.bar(
        comparison["Turn"],
        comparison["Delta"],
        color="skyblue"
    )

    plt.xlabel("Turn")

    plt.ylabel("Time Delta (s)")

    plt.title(
        f"Segment Time Delta ({driver_1} - {driver_2})"
    )

    plt.xticks(
        comparison["Turn"]
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    return comparison["Delta"].tolist()