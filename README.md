# F1 Teammate Analysis

## How do two teammates extract performance from the same car differently?

A Formula 1 data analysis project that investigates the performance differences between teammates driving the same car.

The project uses telemetry and session data from the `FastF1` Python library to compare qualifying laps and analyse how differences in driving behaviour contribute to differences in lap time.

The analysis focuses on identifying where performance differences occur throughout a lap and which driving characteristics contribute to them.

## Objectives

The main objective is to answer the following question:

> **How can two drivers, driving the same car, produce different lap times?**

To investigate this, the project analyses differences in:

* Speed
* Throttle application
* Braking
* RPM
* Gear selection
* Lap time
* Sector performance
* Track position

## Methodology

The project follows a modular data analysis pipeline:

```text
Data Preparation
      ↓
Data Exploration
      ↓
Feature Engineering
      ↓
Feature Analysis
      ↓
Performance Comparison
      ↓
Conclusions
```

### 1. Data Collection

Telemetry and session data are collected using the `FastF1` Python library.

The analysis focuses on qualifying sessions, allowing teammates to be compared under similar competitive conditions.

### 2. Data Preparation

The raw telemetry data is prepared for analysis through the following steps:

* Selecting relevant telemetry channels
* Cleaning the data
* Converting variables to appropriate data types
* Handling missing values
* Structuring the data for further processing
* Data alignement between both drivers

### 3. Data Exploration


The aligned telemetry is initially explored through descriptive statistics and direct visual comparisons.

The analysis examines how the following variables evolve throughout the lap:

* Lap time
* Speed
* Throttle application
* RPM
* Gear selection
* Braking

Differences between the two drivers are also analysed across the circuit to identify areas where performance diverges.

### 4. Feature Engineering

The telemetry data is transformed into higher-level performance features designed to capture specific driving events and behaviours.

These include:

* Braking zones
* Acceleration zones
* Upshifts
* Downshifts
* Braking distance
* Braking duration
* Entry speed
* Minimum speed
* Acceleration performance

These features provide more interpretable metrics for analysing driver behaviour.

### 5. Feature Analysis

The engineered features are analysed to identify differences in driving behaviour between teammates.

The analysis focuses on questions such as:

* Where does each driver begin braking?
* How do braking distances differ?
* Who carries more speed into corners?
* Who reaches full throttle earlier?
* How do gear-shift patterns differ?
* Where does one driver gain or lose performance relative to the other?

### 6. Performance Comparison

The results from the telemetry and feature analyses are combined to identify the key factors contributing to differences in lap time.

The final comparison aims to explain not only which driver was faster, but where and how the performance difference emerged throughout the lap.


## Project Structure

```text
F1-Teammate-Analysis/
│
├── data/
│   ├── raw/
│   ├── unprocessed/
│   └── processed/
│
├── notebooks/
│   ├── data_preparation.ipynb
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── feature_engineering/
│   └── visualization/
│
├── requirements.txt
└── README.md
```

## Technologies

* Python
* Pandas
* NumPy
* SciPy
* Matplotlib
* FastF1

## Project Status

The project is currently under development.

Future analysis will focus on comparing specific performance areas and identifying the driving characteristics that explain differences in lap time between teammates.
