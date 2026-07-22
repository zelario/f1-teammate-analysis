import fastf1
import pandas as pd
from pathlib import Path

def load_driver_cache(year, grand_prix, driver, segment=None):
    """Return a cached payload for a driver if it exists.

    The cache key uses year, grand prix, segment, and driver abbreviation.
    Returns the cached payload dict, or None when the file is missing.
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
    """Persist a driver's data payload to the cache.

    The cache key uses year, grand prix, segment, and driver abbreviation.
    Returns the output path after the payload is saved successfully.
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
    
    lap_1 = load_driver_cache(year, grand_prix, driver_1, segment)
    lap_2 = load_driver_cache(year, grand_prix, driver_2, segment)
    
    if lap_1 is None or lap_2 is None:
        return None, None
    
    print(f"\nProcessed data for {year} {grand_prix} {driver_1} and {driver_2} found in cache.")
    print(f"Driver 1: {lap_1['Driver']}, Driver 2: {lap_2['Driver']}, Segment: {segment}")
    
    return lap_1, lap_2


def save_cache(year, grand_prix, segment, lap_1, lap_2):
    driver_1 = lap_1.get("Driver")
    driver_2 = lap_2.get("Driver")
    save_driver_cache(year, grand_prix, driver_1, lap_1, segment)
    save_driver_cache(year, grand_prix, driver_2, lap_2, segment)
    print(f"\nProcessed data for {year} {grand_prix} saved to cache.")
    print(f"Driver 1: {driver_1}, Driver 2: {driver_2}, Segment: {segment}")

def get_drivers(session, team):
    """Return the two drivers for a team in a session.

    Returns a tuple of (driver1, driver2).
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
    """Find the best shared segment (Q3 > Q2 > Q1) for both teammates in a session.

    Returns the segment name (e.g., 'Q3') or None if no shared segment is found.
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
    
    circuit_info = session.get_circuit_info()

    turns = circuit_info.corners[
        ["Number", "Distance"]
    ].copy()

    turns = turns.sort_values(
        by="Distance"
    ).reset_index(drop=True)

    return turns


def load_driver_data(year, grand_prix, driver, segment, session=None, results=None):
    """Load telemetry and metadata for a single driver.

    Behavior:
    - Select a single lap for the driver in the qualifying session.
    - If `segment` (e.g. 'Q3') and `results` are provided, try to match the segment time among the driver's laps.
    - Otherwise, fall back to the driver's fastest lap.
    - Build payload: {LapTime, LapStartTime, TyreCompound, TyreAge, telemetry}
    - Save the payload to `data/{year}_{grand_prix_safe}_{segment}_{DRIVER}.pkl`.

    Returns the payload dict, cached or freshly built.
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
    """Load qualifying payloads for both teammates.

    The function first checks the unprocessed cache for both drivers using the
    provided segment. If either payload is missing, it loads the qualifying
    session and builds the payloads with `load_driver_data`.

    Returns a tuple with both payload dicts.
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