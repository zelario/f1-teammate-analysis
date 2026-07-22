import matplotlib.pyplot as plt
import numpy as np


def plot_variable_comparison(lap_1_telemetry, lap_2_telemetry, variable, turns, driver_1= None, driver_2= None, ):
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

    y_label = variable

    title = f"{variable} of {driver_1} and {driver_2} throughout the lap"

    plt.figure(figsize=(12, 6))

    if driver_1 is not None:
        plt.plot(
            lap_1_telemetry["Distance"],
            lap_1_telemetry[variable],
            label=driver_1,
            color="green",
            alpha=1,
            linewidth=0.8
        )

    if driver_2 is not None:
        plt.plot(
            lap_2_telemetry["Distance"],
            lap_2_telemetry[variable],
            label=driver_2,
            color="red",
            alpha=1,
            linewidth=0.8
        )

    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()
    plt.grid()

    # Set x-axis ticks to turn numbers
    plt.xticks(ticks=turns['Distance'], labels=turns['Number'])
    plt.xlabel("Turn Number")

    plt.show()

def plot_variable_delta(lap_1_telemetry, lap_2_telemetry, driver_1, driver_2, variable, turns):
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
    plt.ylabel(f" {variable} Delta")
    plt.title(f"{variable} Delta between {driver_1} and {driver_2}")

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid()

    # Set x-axis ticks to turn numbers
    plt.xticks(ticks=turns['Distance'], labels=turns['Number'])
    plt.xlabel("Turn Number")

    plt.show()
    

# Feature analysis functions

def barplot_feature_comparison(lap_1_feature, lap_2_feature, driver_1, driver_2, turns, feature, ylabel=None):

    comparison = turns[["Number"]].rename(columns={"Number": "Turn"})

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
        label=driver_1
    )

    plt.bar(
        x + width / 2,
        comparison[f"{feature}_{driver_2}"],
        width,
        label=driver_2
    )

    plt.xticks(x, comparison["Turn"])

    plt.xlabel("Turn")

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.title(feature.replace("_", " "))

    plt.legend()

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
def barplot_feature_delta(lap_1_feature, lap_2_feature, driver_1, driver_2, turns, feature, ylabel=None):

    comparison = turns[["Number"]].rename(columns={"Number": "Turn"})

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
        comparison["Delta"]
    )

    plt.xlabel("Turn")

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.title(f"{feature} ({driver_1} - {driver_2})")

    plt.xticks(comparison["Turn"])

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    

def scatterplot_features_relationship(lap_1_feature, lap_2_feature, driver_1, driver_2, x_feature, y_feature):

    plt.figure(figsize=(7, 6))

    plt.scatter(
        lap_1_feature[x_feature],
        lap_1_feature[y_feature],
        label=driver_1,
        s=70
    )

    plt.scatter(
        lap_2_feature[x_feature],
        lap_2_feature[y_feature],
        label=driver_2,
        s=70
    )

    for _, row in lap_1_feature.iterrows():
        plt.text(
            row[x_feature],
            row[y_feature],
            str(int(row["Turn"]))
        )

    for _, row in lap_2_feature.iterrows():
        plt.text(
            row[x_feature],
            row[y_feature],
            str(int(row["Turn"]))
        )

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()