import fastf1
import pandas as pd
from pathlib import Path

def check_unprocessed_cache(year, grand_prix, driver, segment=None):
    """Return a cached payload for a driver if it exists.

    The cache key uses year, grand prix, segment, and driver abbreviation.
    Returns the cached payload dict, or None when the file is missing.
    """
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "unprocessed"
    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment
    key = f"{year}_{gp}_{seg}_{driver}"
    out_path = cache_dir / f"{key}.pkl"

    if out_path.exists():
        return pd.read_pickle(out_path)

    return None


def save_unprocessed_cache(year, grand_prix, driver, lap_data, segment):
    """Persist a driver's payload to the unprocessed cache.

    The cache key uses year, grand prix, segment, and driver abbreviation.
    Returns the output path after the payload is saved successfully.
    """
    
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "unprocessed"
    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment 
    key = f"{year}_{gp}_{seg}_{driver}"
    out_path = cache_dir / f"{key}.pkl"

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving payload to {out_path}")
    pd.to_pickle(lap_data, out_path)
    print(f"Saved: {out_path}")

    return out_path


def find_drivers_and_segment(year, grand_prix, team):
    """Find the best shared segment (Q3 > Q2 > Q1) for both teammates in a session.

    Returns the segment name (e.g., 'Q3') or None if no shared segment is found.
    """
    session = fastf1.get_session(year, grand_prix, "Q")
    session.load()

    try:
        results = session.results
    except Exception:
        results = None

    team_column = "Team" if results is not None and "Team" in results.columns else "TeamName"
    if results is None or team_column not in results.columns:
        raise KeyError("No team column found in session results")

    team_drivers = results[
        results[team_column] == team
    ]["Abbreviation"].tolist()

    driver1 = team_drivers[0] if len(team_drivers) > 0 else None
    driver2 = team_drivers[1] if len(team_drivers) > 1 else None

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

    return driver1, driver2, segment

def load_driver_data(year, grand_prix, driver, segment, session=None, results=None):
    """Load telemetry and metadata for a single driver.

    Behavior:
    - Return a cached payload from `data/unprocessed` when available.
    - Select a single lap for the driver in the qualifying session.
    - If `segment` (e.g. 'Q3') and `results` are provided, try to match the segment time among the driver's laps.
    - Otherwise, fall back to the driver's fastest lap.
    - Build payload: {LapTime, LapStartTime, TyreCompound, TyreAge, telemetry}
    - Save the payload to `data/unprocessed/{year}_{grand_prix_safe}_{segment}_{DRIVER}.pkl`.

    Returns the payload dict, cached or freshly built.
    """

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "unprocessed"
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
    
    driver1_data = check_unprocessed_cache(year, grand_prix, driver1, segment=segment)
    driver2_data = check_unprocessed_cache(year, grand_prix, driver2, segment=segment)
    
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