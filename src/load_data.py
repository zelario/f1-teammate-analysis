import fastf1
import pandas as pd
from pathlib import Path


def load_driver_telemetry(year, grand_prix, driver, segment=None, session=None, results=None):
    """Load telemetry and metadata for a single driver.

    Behavior:
    - Select a single lap for the driver in the qualifying session.
    - If `segment` (e.g. 'Q3') and `results` are provided, try to match the segment time among the driver's laps.
    - Otherwise, fall back to the driver's fastest lap.
    - Build payload: {LapTime, LapStartTime, TyreCompound, TyreAge, telemetry}
    - Save to `data/unprocessed/{year}_{grand_prix_safe}_{segment}_{DRIVER}.pkl` if the folder exists.

    Returns the payload dict.
    """
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "unprocessed"
    gp = str(grand_prix).strip().replace(" ", "_").lower()
    seg = segment if segment is not None else 'session'
    key = f"{year}_{gp}_{seg}_{driver}"
    out_path = cache_dir / f"{key}.pkl"

    if out_path.exists():
        return pd.read_pickle(out_path)

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

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving payload to {out_path}")
        pd.to_pickle(lap_data, out_path)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to save payload to {out_path}: {e}")

    return lap_data


def load_teammates_telemetry(year, grand_prix, driver_1, driver_2):
    """Choose qualifying segment both drivers participated in and save each driver's payload.

    Logic:
    - Load session.results and check Q3,Q2,Q1 for both drivers (highest first).
    - Use that segment when calling `load_driver_telemetry` so filenames include the segment.

    Returns (payload1, payload2)
    """
    
    session = fastf1.get_session(year, grand_prix, "Q")
    session.load()

    try:
        results = session.results
    except Exception:
        results = None

    segment = None
    if results is not None:
        try:
            r1 = results[results['Abbreviation'] == driver_1].squeeze()
            r2 = results[results['Abbreviation'] == driver_2].squeeze()
            for seg in ['Q3', 'Q2', 'Q1']:
                if seg in results.columns and pd.notna(r1.get(seg)) and pd.notna(r2.get(seg)):
                    segment = seg
                    break
        except Exception:
            segment = None

    p1 = load_driver_telemetry(year, grand_prix, driver_1, segment=segment, session=session, results=results)
    p2 = load_driver_telemetry(year, grand_prix, driver_2, segment=segment, session=session, results=results)

    return p1, p2


if __name__ == "__main__":
    load_teammates_telemetry(2024, "Japanese Grand Prix", "VER", "PER")
