import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def plot_variable_comparison(lap_1_telemetry, lap_2_telemetry, driver_1, driver_2, variable):
    """
    Plot a telemetry variable against distance for two drivers.

    Parameters
    ----------
    lap_1_telemetry : pandas.DataFrame
        First driver's telemetry for a specific lap.

    lap_2_telemetry : pandas.DataFrame
        Second driver's telemetry for a specific lap.

    variable : str
        Column to plot on the y-axis.

    y_label : str, optional
        Label for the y-axis.

    driver2 : str
        Second driver's abbreviation.

    variable : str
        Column to plot on the y-axis.

    y_label : str, optional
        Label for the y-axis.

    title : str, optional
        Plot title.
    """

    y_label = variable

    title = f"{variable} of {driver_1} and {driver_2} throughout the lap"

    plt.figure(figsize=(12, 6))

    plt.plot(
        lap_1_telemetry["Distance"],
        lap_1_telemetry[variable],
        label=driver_1,
        color="green"
    )

    plt.plot(
        lap_2_telemetry["Distance"],
        lap_2_telemetry[variable],
        label=driver_2,
        color="red"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()
    plt.grid()

    plt.show()
    
def plot_variable_delta(lap_1_telemetry, lap_2_telemetry, driver_1, driver_2, variable):
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
    """

    delta_variable = lap_1_telemetry[variable] - lap_2_telemetry[variable]

    plt.figure(figsize=(12, 6))
    plt.plot(
        lap_1_telemetry["Distance"],
        delta_variable,
        label=f"{driver_1} - {driver_2}",
        color="green"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel(f" {variable} Delta")
    plt.title(f"{variable} Delta between {driver_1} and {driver_2}")

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid()

    plt.show()