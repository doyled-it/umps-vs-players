"""Download every 2026 regular-season pitch from Baseball Savant's Statcast search.

Savant caps a single CSV export at 25,000 rows, so we request two-day windows
(a full slate is ~4,700 pitches/day) and fail loudly if a window hits the cap.

Output: data/raw/statcast_2026.parquet
"""

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

SEASON_START = date(2026, 3, 25)
SEASON_END = date(2026, 9, 27)
WINDOW_DAYS = 2
ROW_CAP = 25_000

OUT = Path("data/raw/statcast_2026.parquet")
CHUNK_DIR = Path("data/raw/statcast_chunks")

URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
    "&hfGT=R%7C&player_type=pitcher&game_date_gt={start}&game_date_lt={end}"
)

KEEP = [
    "game_pk", "game_date", "at_bat_number", "pitch_number", "description", "type",
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "home_team", "away_team", "batter", "pitcher", "fielder_2", "player_name",
    "stand", "p_throws", "pitch_type", "release_speed",
    "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
    "on_1b", "on_2b", "on_3b", "bat_score", "fld_score",
    "home_score", "away_score", "bat_win_exp", "home_win_exp",
    "delta_run_exp", "delta_home_win_exp", "events", "game_type",
]


def fetch_window(start: date) -> Path:
    end = min(start + timedelta(days=WINDOW_DAYS - 1), SEASON_END)
    path = CHUNK_DIR / f"{start.isoformat()}.parquet"
    if path.exists():
        return path
    url = URL.format(start=start.isoformat(), end=end.isoformat())
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
            print(f"retry {start} after {e}", file=sys.stderr)
    text = r.text
    if not text.strip():
        df = pd.DataFrame(columns=KEEP)
    else:
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    if len(df) >= ROW_CAP:
        raise RuntimeError(f"window {start}..{end} hit the {ROW_CAP}-row cap; shrink WINDOW_DAYS")
    df = df[[c for c in KEEP if c in df.columns]]
    df.to_parquet(path, index=False)
    print(f"{start}..{end}: {len(df):,} pitches", file=sys.stderr)
    return path


def main() -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    starts = []
    d = SEASON_START
    while d <= SEASON_END:
        starts.append(d)
        d += timedelta(days=WINDOW_DAYS)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch_window, starts))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df[df["game_type"] == "R"]
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} pitches from {df.game_pk.nunique():,} games")


if __name__ == "__main__":
    main()
