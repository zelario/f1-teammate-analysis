import fastf1
import pandas as pd
from pathlib import Path

def load_driver_cache(year, grand_prix, driver, segment=None):
    """Loads a driver's cached data payload from a pickle file.

    This function attempts to load a pre-saved data payload for a specific driver,
    event, and session segment from the local cache (`/data` directory). The cache
    file is identified by a key constructed from the year, Grand Prix name, segment,
    and driver abbreviation.

    Parameters:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., "Japanese Grand Prix").
        driver (str or dict): The driver's abbreviation (e.g., "VER") or a dictionary
                              containing driver information.
        segment (str, optional): The session segment (e.g., "Q1", "Q2", "Q3").
                                 Defaults to "session".

    Returns:
        dict or None: The loaded data payload as a dictionary if the cache file exists,
                      otherwise None.
    """
    
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data"

    # Accept either a driver string or a driver payload dict
    if isinstance(driver, dict):
        driver_key = driver.get("Driver") or driver.get("driver") or driver.get("Abbreviation") or driver.get("abbr")
    else:
        driver_key = driver

    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment if segment is not None else "session"
    key = f"{year}_{gp}_{seg}_{driver_key}"
    out_path = cache_dir / f"{key}.pkl"

    if out_path.exists():
        return pd.read_pickle(out_path)

    return None

def save_driver_cache(year, grand_prix, driver, data_payload, segment=None):
    """Persists a driver's data payload to a pickle file in the local cache.

    This function serializes and saves the driver's telemetry and metadata payload
    into the `/data` directory. The cache file name is structured using the year,
    Grand Prix name, session segment, and driver abbreviation.

    Parameters:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., "Japanese Grand Prix").
        driver (str): The driver's abbreviation (e.g., "VER").
        data_payload (dict): The dictionary containing telemetry and metadata to be cached.
        segment (str, optional): The session segment (e.g., "Q1", "Q2", "Q3").
                                 Defaults to "session".

    Returns:
        Path: The absolute path to the newly created cache file.
    """
    
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data"
    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment if segment is not None else "session"
    key = f"{year}_{gp}_{seg}_{driver}"
    out_path = cache_dir / f"{key}.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving payload to {out_path}")
    pd.to_pickle(data_payload, out_path)
    print(f"Saved: {out_path}")

    return out_path


def load_cache(year, grand_prix, segment, driver_1, driver_2):
    """Loads cached data payloads for both teammates in a specific session.

    This function attempts to load the cached data payloads for two drivers for a
    given year, Grand Prix, and segment. If either driver's cached payload is
    missing, it returns a tuple of (None, None).

    Parameters:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., "Japanese Grand Prix").
        segment (str): The session segment (e.g., "Q1", "Q2", "Q3").
        driver_1 (str): The first driver's abbreviation (e.g., "VER").
        driver_2 (str): The second driver's abbreviation (e.g., "PER").

    Returns:
        tuple (dict or None, dict or None): A tuple containing the data payloads of
                                            driver_1 and driver_2 respectively. Both
                                            elements are None if either is not cached.
    """
    
    lap_1 = load_driver_cache(year, grand_prix, driver_1, segment)
    lap_2 = load_driver_cache(year, grand_prix, driver_2, segment)
    
    if lap_1 is None or lap_2 is None:
        return None, None
    
    print(f"\nProcessed data for {year} {grand_prix} {driver_1} and {driver_2} found in cache.")
    print(f"Driver 1: {lap_1['Driver']}, Driver 2: {lap_2['Driver']}, Segment: {segment}")
    
    return lap_1, lap_2


def save_cache(year, grand_prix, segment, lap_1, lap_2):
    """Persists the data payloads for both teammates to the cache.

    This function extracts the driver abbreviations from each lap data dictionary
    and invokes `save_driver_cache` to serialize and save the payloads for both
    teammates under the designated session segment.

    Parameters:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., "Japanese Grand Prix").
        segment (str): The session segment (e.g., "Q1", "Q2", "Q3").
        lap_1 (dict): The data payload for the first driver, including 'Driver' key.
        lap_2 (dict): The data payload for the second driver, including 'Driver' key.

    Returns:
        None
    """
    driver_1 = lap_1.get("Driver")
    driver_2 = lap_2.get("Driver")
    save_driver_cache(year, grand_prix, driver_1, lap_1, segment)
    save_driver_cache(year, grand_prix, driver_2, lap_2, segment)
    print(f"\nProcessed data for {year} {grand_prix} saved to cache.")
    print(f"Driver 1: {driver_1}, Driver 2: {driver_2}, Segment: {segment}")

def get_drivers(session, team):
    """Retrieves the abbreviations of the two drivers competing for a team.

    This function searches the session results to identify and return the two
    drivers associated with the specified team.

    Parameters:
        session (fastf1.core.Session): The active FastF1 session object.
        team (str): The name of the team (e.g., "Red Bull Racing").

    Returns:
        tuple (str or None, str or None): A tuple containing the abbreviations of the two
                                          drivers (e.g., ("VER", "PER")). Elements can be
                                          None if the team has fewer than two drivers.
    """
    
    try:
        results = session.results
    except Exception:
        results = None

    team_drivers = results[
        results["TeamName"] == team
    ]["Abbreviation"].tolist()

    driver1 = team_drivers[0] if len(team_drivers) > 0 else None
    driver2 = team_drivers[1] if len(team_drivers) > 1 else None

    return driver1, driver2

def get_segment(session, driver1, driver2):
    """Identifies the highest shared session segment completed by both teammates.

    This function checks qualifying results to determine the best segment (Q3, Q2,
    or Q1) that both driver1 and driver2 have registered times in.

    Parameters:
        session (fastf1.core.Session): The active FastF1 session object.
        driver1 (str): The first driver's abbreviation (e.g., "VER").
        driver2 (str): The second driver's abbreviation (e.g., "PER").

    Returns:
        str or None: The name of the highest shared segment (e.g., "Q3"), or None
                     if no shared segment is found.
    """

    try:
        results = session.results
    except Exception:
        results = None

    segment = None
    if results is not None:
        try:
            r1 = results[results['Abbreviation'] == driver1].squeeze()
            r2 = results[results['Abbreviation'] == driver2].squeeze()
            for seg in ['Q3', 'Q2', 'Q1']:
                if seg in results.columns and pd.notna(r1.get(seg)) and pd.notna(r2.get(seg)):
                    segment = seg
                    break
        except Exception:
            segment = None

    return segment


def get_turns(session):
    """Extracts and sorts the circuit corner/turn information for a session.

    This function retrieves the circuit details from the session, extracts the corner
    numbers and their respective distances from the start/finish line, and sorts
    them in ascending order of distance.

    Parameters:
        session (fastf1.core.Session): The active FastF1 session object.

    Returns:
        pandas.DataFrame: A DataFrame containing sorted 'Number' and 'Distance' columns
                          for all corners on the circuit.
    """
    
    circuit_info = session.get_circuit_info()

    turns = circuit_info.corners[
        ["Number", "Distance"]
    ].copy()

    turns = turns.sort_values(
        by="Distance"
    ).reset_index(drop=True)

    return turns


def load_driver_data(year, grand_prix, driver, segment, session=None, results=None):
    """Loads and compiles telemetry and metadata for a single driver's lap.

    This function retrieves qualifying lap telemetry and metadata for a driver.
    If a segment and results are provided, it matches the driver's lap closest
    to their recorded segment time. Otherwise, it falls back to their fastest
    lap. It then returns a payload dict containing telemetry, lap time, compound,
    and tire age details.

    Parameters:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., "Japanese Grand Prix").
        driver (str): The driver's abbreviation (e.g., "VER").
        segment (str): The session segment (e.g., "Q1", "Q2", "Q3").
        session (fastf1.core.Session, optional): A loaded FastF1 session object.
                                                 If None, it is fetched and loaded.
        results (pandas.DataFrame, optional): Session results DataFrame. If None,
                                              retrieved from the loaded session.

    Returns:
        dict: A dictionary containing 'Driver', 'LapTime', 'LapStartTime',
              'TyreCompound', 'TyreAge', and 'Telemetry' (DataFrame).
    """

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data"
    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment
    key = f"{year}_{gp}_{seg}_{driver}"
    out_path = cache_dir / f"{key}.pkl"

    # prepare session/results if not provided
    if session is None:
        session = fastf1.get_session(year, grand_prix, "Q")
        session.load()

    if results is None:
        try:
            results = session.results
        except Exception:
            results = None

    # get all laps for driver
    driver_laps_all = session.laps.pick_drivers(driver)

    # If a segment is specified and results available, pick the lap whose LapTime is closest to that segment time
    lap = None
    if segment is not None and results is not None:
        try:
            row = results[results['Abbreviation'] == driver].squeeze()
            val = row.get(segment)
            if pd.notna(val):
                target_td = pd.to_timedelta(val)
                try:
                    diffs = (driver_laps_all['LapTime'] - target_td).abs()
                    idx = diffs.idxmin()
                    lap = driver_laps_all.loc[idx]
                except Exception:
                    lap = None
        except Exception:
            lap = None

    # If no lap found by segment match, fall back to the fastest lap (or first available)
    if lap is None:
        try:
            lap = driver_laps_all.pick_fastest()
        except Exception:
            try:
                lap = driver_laps_all.nsmallest(1, 'LapTime').iloc[0]
            except Exception:
                try:
                    lap = driver_laps_all.iloc[0]
                except Exception:
                    lap = None

    if lap is None:
        raise RuntimeError(f"No lap found for driver {driver}")

    telemetry = lap.get_telemetry()

    lap_data = {
        'Driver': driver,
        'LapTime': str(lap.get('LapTime')) if 'LapTime' in lap.index else None,
        'LapStartTime': str(lap.get('Time')) if 'Time' in lap.index else None,
        'TyreCompound': lap.get('Compound') if 'Compound' in lap.index else lap.get('TyreCompound') if 'TyreCompound' in lap.index else None,
        'TyreAge': lap.get('TyreLife') if 'TyreLife' in lap.index else None,
        'Telemetry': telemetry,
    }

    return lap_data


def load_teammates_data(year, grand_prix, segment, driver1, driver2):
    """Loads qualifying data payloads for both teammates, utilizing cache if available.

    This function first attempts to load cached payloads for both drivers. If either
    is missing, it loads the qualifying session, fetches the results, and compiles
    the payloads using `load_driver_data`.

    Parameters:
        year (int): The year of the Grand Prix.
        grand_prix (str): The name of the Grand Prix (e.g., "Japanese Grand Prix").
        segment (str): The session segment (e.g., "Q1", "Q2", "Q3").
        driver1 (str): The first driver's abbreviation (e.g., "VER").
        driver2 (str): The second driver's abbreviation (e.g., "PER").

    Returns:
        tuple (dict, dict): A tuple containing the compiled or cached data payloads for
                            driver1 and driver2 respectively.
    """
    
    driver1_data = load_driver_cache(year, grand_prix, driver1, segment=segment)
    driver2_data = load_driver_cache(year, grand_prix, driver2, segment=segment)
    
    if driver1_data is not None and driver2_data is not None:
        return driver1_data, driver2_data
    
    session = fastf1.get_session(year, grand_prix, "Q")
    session.load()

    try:
        results = session.results
    except Exception:
        results = None

    p1 = load_driver_data(year, grand_prix, driver1, segment=segment, session=session, results=results)
    p2 = load_driver_data(year, grand_prix, driver2, segment=segment, session=session, results=results)

    return p1, p2



if __name__ == "__main__":
    pass