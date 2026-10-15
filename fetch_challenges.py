"""Pull every 2026 regular-season ABS challenge from the MLB Stats API.

Each challenged pitch in a game's play-by-play carries a `reviewDetails` block:
who challenged (`player`), for which team (`challengeTeamId`) and whether the
call was overturned. The pitch's `details.call` is the call *after* review, so
an overturned "Ball" was originally called a strike and vice versa.

When the challenged pitch ends the plate appearance (strike three, ball four),
the feed puts `reviewDetails` on the play instead of the pitch; those are
attached to the play's last pitch and marked `source = "play"`.

Outputs:
  data/games_2026.csv       one row per game (teams, home-plate umpire)
  data/challenges_2026.csv  one row per challenged pitch
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
START, END = "2026-03-25", "2026-09-27"

session = requests.Session()


def get(url: str) -> dict:
    for attempt in range(5):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 4:
                raise
            print(f"retry {url} after {e}", file=sys.stderr)
            time.sleep(2 ** (attempt + 1))


def schedule() -> pd.DataFrame:
    d = get(f"{API}/schedule?sportId=1&gameType=R&startDate={START}&endDate={END}&hydrate=officials")
    rows = []
    for day in d["dates"]:
        for g in day["games"]:
            hp = next((o["official"] for o in g.get("officials", []) if o["officialType"] == "Home Plate"), {})
            rows.append({
                "game_pk": g["gamePk"],
                "official_date": g.get("officialDate"),
                "state": g["status"]["codedGameState"],
                "home_id": g["teams"]["home"]["team"]["id"],
                "home_name": g["teams"]["home"]["team"]["name"],
                "away_id": g["teams"]["away"]["team"]["id"],
                "away_name": g["teams"]["away"]["team"]["name"],
                "hp_umpire_id": hp.get("id"),
                "hp_umpire": hp.get("fullName"),
            })
    games = pd.DataFrame(rows)
    # Postponed and suspended games show up on more than one date; keep the
    # completed listing.
    games = games[games["state"] == "F"].drop_duplicates("game_pk", keep="last")
    return games.reset_index(drop=True)


def review_row(game_pk: int, play: dict, ev: dict, rd: dict, source: str) -> dict:
    about, matchup = play["about"], play["matchup"]
    return {
        "game_pk": game_pk,
        "at_bat_number": about["atBatIndex"] + 1,
        "pitch_number": ev.get("pitchNumber"),
        "inning": about["inning"],
        "half": about["halfInning"],
        "batter_id": matchup["batter"]["id"],
        "pitcher_id": matchup["pitcher"]["id"],
        "final_call": ev["details"].get("call", {}).get("code"),
        "final_call_desc": ev["details"].get("call", {}).get("description"),
        "review_type": rd.get("reviewType"),
        "overturned": rd.get("isOverturned"),
        "in_progress": rd.get("inProgress"),
        "challenge_team_id": rd.get("challengeTeamId"),
        "challenger_id": (rd.get("player") or {}).get("id"),
        "challenger_name": (rd.get("player") or {}).get("fullName"),
        "start_time": ev.get("startTime"),
        "source": source,
    }


def game_challenges(game_pk: int) -> list[dict]:
    d = get(f"{API}/game/{game_pk}/playByPlay")
    rows = []
    for play in d["allPlays"]:
        pitches = [ev for ev in play["playEvents"] if ev.get("isPitch")]
        for ev in pitches:
            if ev.get("reviewDetails"):
                rows.append(review_row(game_pk, play, ev, ev["reviewDetails"], "pitch"))
        play_rd = play.get("reviewDetails")
        if play_rd and pitches:
            last = pitches[-1]
            for rd in [play_rd, *(play_rd.get("additionalReviews") or [])]:
                if rd.get("reviewType") != "MJ":
                    continue
                if (last.get("reviewDetails") or {}).get("reviewType") == "MJ":
                    continue  # already recorded on the pitch itself
                rows.append(review_row(game_pk, play, last, rd, "play"))
    return rows


def main() -> None:
    Path("data").mkdir(exist_ok=True)
    games = schedule()
    games.to_csv("data/games_2026.csv", index=False)
    print(f"{len(games):,} final regular-season games", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(game_challenges, games["game_pk"]))
    ch = pd.DataFrame([r for rows in results for r in rows])
    ch.to_csv("data/challenges_2026.csv", index=False)
    print(f"wrote data/challenges_2026.csv: {len(ch):,} challenged pitches")
    print(ch["review_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
