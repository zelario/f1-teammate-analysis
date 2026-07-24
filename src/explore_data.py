import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    fig = plt.gcf()
    plt.show()
    return fig

def barplot_feature_comparison(lap_1, lap_2, dataframe_name, feature, turns, y_unit=None, include_zero=False):
    """Generates a bar plot comparing a specific feature between two drivers.

    This function creates a grouped bar chart to compare a chosen feature (e.g., speed, brake)
    for two drivers across different turns of a lap. It can optionally include a 'zero'
    segment for comparison.

    Parameters:
        lap_1 (dict): A dictionary containing the first driver's data, including 'Telemetry' and 'Driver' keys.
        lap_2 (dict): A dictionary containing the second driver's data, including 'Telemetry' and 'Driver' keys.
        data (pandas.DataFrame): The combined telemetry data for both drivers.
        feature (str): The name of the feature column to plot on the y-axis.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number' column.
        y_unit (str, optional): The unit of the feature being plotted (e.g., "km/h", "%").
                                 Appears in the y-axis label. Defaults to None.
        include_zero (bool, optional): If True, includes a 'zero' segment in the comparison.
                                       Defaults to False.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number'
                                  column.
        feature (str): The name of the feature column to plot on the y-axis.
        y_unit (str, optional): The unit of the feature being plotted (e.g., "km/h", "%").
                                 Appears in the y-axis label. Defaults to None.
        include_zero (bool, optional): If True, includes a 'zero' segment in the comparison.
                                       Defaults to False.
    """
    
    lap_1_feature = lap_1[dataframe_name]
    lap_2_feature = lap_2[dataframe_name]
    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

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
    
    fig = plt.gcf()
    plt.show()
    return fig
    
    
def barplot_feature_delta(lap_1, lap_2, dataframe_name, feature, turns, y_unit=None, include_zero=False):
    """Generates a bar plot showing the delta of a specific feature between two drivers.

    This function calculates the difference (delta) of a chosen feature between two drivers
    and visualizes it as a bar chart across different turns of a lap. It can optionally
    include a 'zero' segment.

    Parameters:
        lap_1_feature (pandas.DataFrame): Feature data for the first driver.
                                         Must contain 'Turn' and the specified `feature` columns.
        lap_2_feature (pandas.DataFrame): Feature data for the second driver.
                                         Must contain 'Turn' and the specified `feature` columns.
        driver_1 (str): The abbreviation of the first driver. Used for plot labels.
        driver_2 (str): The abbreviation of the second driver. Used for plot labels.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number'
                                  column.
        feature (str): The name of the feature column to calculate the delta for and plot
                        on the y-axis.
        y_unit (str, optional): The unit of the feature being plotted (e.g., "km/h", "%").
                                 Appears in the y-axis label. Defaults to None.
        include_zero (bool, optional): If True, includes a 'zero' segment in the comparison.
                                       Defaults to False.
    """
    
    lap_1_feature = lap_1[dataframe_name]
    lap_2_feature = lap_2[dataframe_name]
    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

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
    
    fig = plt.gcf()
    plt.show()
    return fig
    

def scatterplot_gear_shifts(lap, turns):
    """Generates a scatter plot visualizing gear shifts throughout a lap.

    This function plots upshifts and downshifts against distance, indicating the gear
    selected after the shift. Turn numbers are used for contextualizing the shifts.

    Parameters:
        gear_shifts (pandas.DataFrame): DataFrame containing gear shift information.
                                        Must include 'Distance', 'Direction' (up/down), and 'GearTo' columns.
        driver (str): The abbreviation of the driver. Used for the plot title.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number'
                                  and 'Distance' columns. These distances are used for x-axis ticks.
    """
    
    gear_shifts = lap["GearShifts"]
    driver = lap["Driver"]

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

    fig = plt.gcf()
    plt.show()
    return fig

def scatterplot_shift_rpm(lap, turns):
    """Generates a scatter plot showing RPM before and after gear shifts.

    This function visualizes the change in RPM during upshifts and downshifts,
    plotting both the RPM value before and after each shift. It uses turn numbers
    for x-axis context.

    Parameters:
        gear_shifts (pandas.DataFrame): DataFrame containing gear shift information.
                                        Must include 'Distance', 'Direction', 'RPMbefore', and 'RPMafter' columns.
        driver (str): The abbreviation of the driver. Used for the plot title.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number'
                                  and 'Distance' columns. These distances are used for x-axis ticks.
    """

    gear_shifts = lap["GearShifts"]
    driver = lap["Driver"]

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

    fig = plt.gcf()
    plt.show()
    return fig
    
    
def scatterplot_features_relationship(lap_1, lap_2, x_dataframe, x_feature, y_dataframe, y_feature):
    """Generates a scatter plot to visualize the relationship between the deltas of two features.

    This function compares two different features between two drivers, calculates the delta
    for each feature, and then plots these deltas against each other. Each point represents
    a turn, allowing for an understanding of how changes in one feature's delta relate
    to changes in another's.

    Parameters:
        lap_1 (dict): Dictionary containing various dataframes for the first driver,
                      including those specified by `x_dataframe` and `y_dataframe`.
        lap_2 (dict): Dictionary containing various dataframes for the second driver,
                      including those specified by `x_dataframe` and `y_dataframe`.
        driver_1 (str): The abbreviation of the first driver. Used for plot labels.
        driver_2 (str): The abbreviation of the second driver. Used for plot labels.
        x_dataframe (str): The key in the `lap_1` and `lap_2` dictionaries that
                           corresponds to the dataframe containing `x_feature`.
        x_feature (str): The name of the feature from `x_dataframe` to be used on the x-axis.
        y_dataframe (str): The key in the `lap_1` and `lap_2` dictionaries that
                           corresponds to the dataframe containing `y_feature`.
        y_feature (str): The name of the feature from `y_dataframe` to be used on the y-axis.
    """
    
    driver_1 = lap_1["Driver"]
    driver_2 = lap_2["Driver"]

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

    fig = plt.gcf()
    plt.show()
    return fig

# Performance comparison functions

def analyze_segment_times(lap, turns):
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

        segment_times.append({
            "Turn": turn_number,
            "StartDistance": start_distance,
            "EndDistance": end_distance,
            "SegmentTime": end_time - start_time
        })

    return pd.DataFrame(
        segment_times
    ).reset_index(drop=True)
    
    
def barplot_segment_times_comparison(segment_times_1, segment_times_2, driver_1, driver_2, turns):
    """Generates a bar plot comparing segment times between two drivers.

    This function visualizes the time spent by two drivers in each segment of the track,
    defined by the turns, using a grouped bar chart.

    Parameters:
        segment_times_1 (pandas.DataFrame): Segment times data for the first driver.
                                            Must contain 'Turn' and 'SegmentTime' columns.
        segment_times_2 (pandas.DataFrame): Segment times data for the second driver.
                                            Must contain 'Turn' and 'SegmentTime' columns.
        driver_1 (str): The abbreviation of the first driver. Used for plot labels.
        driver_2 (str): The abbreviation of the second driver. Used for plot labels.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number' column.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the plot.
    """
    
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

    x = np.arange(len(comparison))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        comparison[f"SegmentTime_{driver_1}"],
        width,
        label=driver_1,
        color=choose_color(driver_1)
    )

    plt.bar(
        x + width / 2,
        comparison[f"SegmentTime_{driver_2}"],
        width,
        label=driver_2,
        color=choose_color(driver_2)
    )

    plt.xticks(x, comparison["Turn"])

    plt.xlabel("Turn")

    plt.ylabel("Segment Time (s)")

    plt.title("Segment Times Comparison")

    plt.legend()

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    fig = plt.gcf()
    plt.show()
    
    return fig

def barplot_segment_time_delta(segment_times_1, segment_times_2, driver_1, driver_2, turns):
    """Generates a bar plot showing the time delta for each track segment between two drivers.

    This function visualizes the difference in time spent by two drivers in each segment
    of the track, defined by the turns.

    Parameters:
        segment_times_1 (pandas.DataFrame): Segment times data for the first driver.
                                            Must contain 'Turn' and 'SegmentTime' columns.
        segment_times_2 (pandas.DataFrame): Segment times data for the second driver.
                                            Must contain 'Turn' and 'SegmentTime' columns.
        driver_1 (str): The abbreviation of the first driver. Used for plot labels.
        driver_2 (str): The abbreviation of the second driver. Used for plot labels.
        turns (pandas.DataFrame): DataFrame containing turn information, expected to have 'Number'
                                  column.

    Returns:
        tuple: A tuple containing a list of the calculated time deltas for each segment and the matplotlib figure object.
    """

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

    fig = plt.gcf()
    plt.show()

    return comparison["Delta"].tolist(), fig