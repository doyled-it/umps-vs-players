"""Catchers vs. umpires: who knows the ABS strike zone better? (2026 regular season)

Inputs (from fetch_statcast.py and fetch_challenges.py):
  data/raw/statcast_2026.parquet   every pitch, with location at the middle of
                                   the plate and each batter's ABS zone
  data/challenges_2026.csv         every challenged pitch and its result
  data/games_2026.csv              teams per game

Outputs: results/*.csv and results/summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import optimize, stats

OUT = Path("results")
HALF_WIDTH_IN = 8.5  # 17-inch plate
KEYS = ["game_pk", "at_bat_number", "pitch_number"]
CALLED = {"called_strike": "S", "ball": "B", "blocked_ball": "B"}
RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------- data prep

def load() -> pd.DataFrame:
    sc = pd.read_parquet("data/raw/statcast_2026.parquet")
    games = pd.read_csv("data/games_2026.csv")[["game_pk", "home_id", "away_id", "hp_umpire_id", "hp_umpire"]]
    ch = pd.read_csv("data/challenges_2026.csv")
    ch = ch[ch["review_type"] == "MJ"]

    df = sc[sc["description"].isin(CALLED)].copy()
    df["final"] = df["description"].map(CALLED)
    df = df.merge(games, on="game_pk", how="inner")
    cols = ["final_call", "overturned", "challenge_team_id", "challenger_id", "challenger_name"]
    df = df.merge(ch[KEYS + cols], on=KEYS, how="left")
    df["challenged"] = df["final_call"].notna()

    unmatched = len(ch) - int(df["challenged"].sum())
    # The feed and Statcast should agree on the post-review call.
    feed_final = df["final_call"].map({"C": "S", "B": "B"})
    bad = df["challenged"] & (feed_final != df["final"])
    df = df[~bad].copy()

    df["overturned"] = df["overturned"].fillna(False).astype(bool)
    flip = {"S": "B", "B": "S"}
    df["orig"] = np.where(df["overturned"], df["final"].map(flip), df["final"])

    top = df["inning_topbot"] == "Top"
    df["def_id"] = np.where(top, df["home_id"], df["away_id"])
    df["off_id"] = np.where(top, df["away_id"], df["home_id"])
    df["side"] = np.select(
        [df["challenge_team_id"] == df["def_id"], df["challenge_team_id"] == df["off_id"]],
        ["defense", "offense"], default="")
    df["role"] = np.select(
        [~df["challenged"],
         df["challenger_id"] == df["batter"],
         df["challenger_id"] == df["fielder_2"],
         df["challenger_id"] == df["pitcher"]],
        ["", "batter", "catcher", "pitcher"], default="other")

    df = df.sort_values(KEYS).reset_index(drop=True)
    df = add_challenges_left(df)
    df.attrs["dropped_inconsistent"] = int(bad.sum())
    df.attrs["unmatched_challenges"] = unmatched
    return df


def add_challenges_left(df: pd.DataFrame) -> pd.DataFrame:
    """Failed challenges each team had used before this pitch (teams start with 2
    and keep the ones they win; the pool is shared by offense and defense)."""
    failed = df["challenged"] & ~df["overturned"]
    for team in ("home", "away"):
        f = (failed & (df["challenge_team_id"] == df[f"{team}_id"])).astype(int)
        df[f"lost_{team}"] = f.groupby(df["game_pk"]).cumsum() - f
    top = df["inning_topbot"] == "Top"
    df["def_lost"] = np.where(top, df["lost_home"], df["lost_away"])
    df["off_lost"] = np.where(top, df["lost_away"], df["lost_home"])
    return df


def add_abs_margin(df: pd.DataFrame, radius_in: float) -> pd.DataFrame:
    """Signed inches by which the ball missed (+) or caught (-) the ABS zone.

    Statcast's 2026 plate_x/plate_z are measured at the middle of the plate,
    where ABS judges the pitch, and sz_top/sz_bot are the batter's ABS zone.
    A pitch is a strike if any part of the ball touches the zone.
    """
    x = df["plate_x"].abs() * 12
    z = df["plate_z"] * 12
    dx = x - HALF_WIDTH_IN
    dz_top = z - df["sz_top"] * 12
    dz_bot = df["sz_bot"] * 12 - z
    dz = np.maximum(dz_top, dz_bot)
    outside = np.hypot(np.clip(dx, 0, None), np.clip(dz, 0, None))
    inside = np.maximum(dx, dz)
    center_to_edge = np.where((dx > 0) | (dz > 0), outside, inside)
    df["margin"] = center_to_edge - radius_in
    df["abs_call"] = np.where(df["margin"] <= 0, "S", "B")
    df["edge"] = np.where(dx >= dz, "side", np.where(dz_top >= dz_bot, "top", "bottom"))
    return df


def calibrate_radius(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """ABS settled every challenged pitch, so its post-review call is ground
    truth. Pick the ball radius that best reproduces those rulings."""
    c = df[df["challenged"] & df["plate_x"].notna()].copy()
    rows = []
    for r in np.round(np.arange(0.0, 2.51, 0.01), 2):
        pred = add_abs_margin(c.copy(), r)["abs_call"]
        rows.append({"radius_in": r, "agreement": (pred == c["final"]).mean()})
    grid = pd.DataFrame(rows)
    best = grid["agreement"].max()
    r = float(grid.loc[grid["agreement"] >= best - 1e-12, "radius_in"].median())
    return r, grid


# ------------------------------------------------------------ small helpers

def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate_row(label: str, k: int, n: int, **extra) -> dict:
    lo, hi = wilson(k, n)
    return {"group": label, "n": int(n), "k": int(k), "rate": k / n if n else np.nan,
            "ci_lo": lo, "ci_hi": hi, **extra}


# ------------------------------------------------- psychometric curve fits

BIN_EDGES = np.arange(-6.0, 6.0001, 0.1)
BIN_MID = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2


def _nll(theta, x, n, k):
    mu, log_s, a, b = theta
    floor = stats.norm.cdf(a) * 0.5
    ceil = 0.5 + stats.norm.cdf(b) * 0.5
    p = floor + (ceil - floor) * stats.norm.cdf((mu - x) / np.exp(log_s))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()


def fit_curve(x, n, k, init=None) -> dict:
    """P(yes | margin) = floor + (ceil - floor) * Phi((mu - margin) / sigma).

    sigma is how blurry the person's zone is, in inches: the spread of their
    read of where the pitch crossed. mu is where they say yes half the time
    (relative to the floor and ceiling); ceil < 1 captures saying no for
    reasons other than location (e.g. saving a challenge).
    """
    m = n > 0
    x, n, k = x[m], n[m], k[m]
    starts = [init] if init is not None else [
        [mu0, 0.0, -3.0, b0] for mu0 in (-2.0, -0.5, 0.5) for b0 in (3.0, 0.0, -2.0)]
    best = None
    for theta0 in starts:
        res = optimize.minimize(_nll, theta0, args=(x, n, k), method="Nelder-Mead",
                                options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    mu, log_s, a, b = best.x
    return {"mu": mu, "sigma": float(np.exp(log_s)),
            "floor": float(stats.norm.cdf(a) * 0.5), "ceil": float(0.5 + stats.norm.cdf(b) * 0.5),
            "theta": best.x}


def curve_with_ci(sub: pd.DataFrame, y: pd.Series, reps: int = 200) -> dict:
    """Fit on 0.1-inch bins; 95% CI by resampling whole games."""
    s = sub[(sub["margin"] >= BIN_EDGES[0]) & (sub["margin"] < BIN_EDGES[-1])]
    yy = y.loc[s.index].astype(int)
    b = np.digitize(s["margin"], BIN_EDGES) - 1
    g_codes, g_idx = np.unique(s["game_pk"], return_inverse=True)
    N = np.zeros((len(g_codes), len(BIN_MID)))
    K = np.zeros_like(N)
    np.add.at(N, (g_idx, b), 1)
    np.add.at(K, (g_idx, b), yy.to_numpy())
    point = fit_curve(BIN_MID, N.sum(0), K.sum(0))
    sigmas, mus = [], []
    for _ in range(reps):
        w = RNG.multinomial(len(g_codes), np.full(len(g_codes), 1 / len(g_codes)))
        f = fit_curve(BIN_MID, w @ N, w @ K, init=point["theta"])
        sigmas.append(f["sigma"])
        mus.append(f["mu"])
    del point["theta"]
    point.update({
        "sigma_lo": float(np.percentile(sigmas, 2.5)), "sigma_hi": float(np.percentile(sigmas, 97.5)),
        "mu_lo": float(np.percentile(mus, 2.5)), "mu_hi": float(np.percentile(mus, 97.5)),
        "n_pitches": int(len(s)), "n_yes": int(yy.sum()),
    })
    return point


def shared_misread_sensitivity(pool: pd.DataFrame, ump_fit: dict, rhos=(0.0, 0.2, 0.4, 0.6)) -> list[dict]:
    """We only see the catcher's read on pitches the umpire called balls. If the
    catcher's misreads line up with the umpire's (correlation rho), the pitches
    that fooled the umpire also tend to fool the catcher, which makes the
    catcher look blurrier than they are. Refit the catcher's blur under a
    bivariate-normal model of the two reads for a range of rho."""
    s = pool[(pool["margin"] >= BIN_EDGES[0]) & (pool["margin"] < BIN_EDGES[-1])]
    b = np.digitize(s["margin"], BIN_EDGES) - 1
    n = np.bincount(b, minlength=len(BIN_MID)).astype(float)
    k = np.bincount(b, weights=(s["role"] == "catcher").astype(int), minlength=len(BIN_MID))
    keep = n > 0
    x, n, k = BIN_MID[keep], n[keep], k[keep]
    su, mu_u = ump_fit["sigma"], ump_fit["mu"]
    bu = (mu_u - x) / su  # umpire called ball: their read was beyond this

    def nll(theta, rho):
        mu_c, log_s, bc = theta
        a = (mu_c - x) / np.exp(log_s)
        mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
        both_below = np.array([mvn.cdf([ai, bi]) for ai, bi in zip(a, bu)])
        cond = (stats.norm.cdf(a) - both_below) / np.clip(stats.norm.sf(bu), 1e-12, None)
        p = np.clip((0.5 + 0.5 * stats.norm.cdf(bc)) * cond, 1e-9, 1 - 1e-9)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()

    out = []
    for rho in rhos:
        r = optimize.minimize(nll, [-1.5, np.log(1.8), 1.0], args=(rho,), method="Nelder-Mead",
                              options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-4})
        out.append({"rho": rho, "catcher_sigma": float(np.exp(r.x[1])), "catcher_mu": float(r.x[0]),
                    "catcher_ceil": float(0.5 + 0.5 * stats.norm.cdf(r.x[2])),
                    "neg_log_lik": float(r.fun)})
    return out


def clone_umpire(pool: pd.DataFrame, catcher_fit: dict, sigma: float) -> dict:
    """Put an umpire-eyed challenger in the catcher's crouch: same ball calls,
    same number of challenges, same ceiling, but the umpire's blur. Assumes
    its read is independent of the plate umpire's."""
    m = pool["margin"].to_numpy()
    target = (pool["role"] == "catcher").sum()
    ceil = catcher_fit["ceil"]

    def p(mu):
        return ceil * stats.norm.cdf((mu - m) / sigma)

    mu = optimize.brentq(lambda mu: p(mu).sum() - target, -10, 10)
    probs = p(mu)
    return {"sigma": sigma, "mu": mu, "challenges": float(probs.sum()),
            "win_rate": float(probs[m <= 0].sum() / probs.sum())}


# ----------------------------------------------------------------- analysis

def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    summary = {
        "called_pitches": int(len(df)),
        "games": int(df["game_pk"].nunique()),
        "challenges": int(df["challenged"].sum()),
        "dropped_inconsistent": df.attrs["dropped_inconsistent"],
        "unmatched_challenges": df.attrs["unmatched_challenges"],
    }

    # 1. Head-to-head: every challenge is a pitch where the challenger and the
    #    umpire disagreed, and ABS decided who was right.
    ch = df[df["challenged"]]
    roles = []
    for role in ["catcher", "batter", "pitcher", "other"]:
        r = ch[ch["role"] == role]
        roles.append(rate_row(role, r["overturned"].sum(), len(r),
                              on_ball_calls=int((r["orig"] == "B").sum()),
                              on_strike_calls=int((r["orig"] == "S").sum())))
    roles.append(rate_row("all", ch["overturned"].sum(), len(ch)))
    by_role = pd.DataFrame(roles)
    by_role.to_csv(OUT / "challenges_by_role.csv", index=False)

    # Rule check: in regulation a team with two failed challenges is out.
    reg = ch[ch["inning"] <= 9]
    summary["regulation_challenges_after_two_losses"] = int(
        ((reg["side"] == "defense") & (reg["def_lost"] >= 2)).sum()
        + ((reg["side"] == "offense") & (reg["off_lost"] >= 2)).sum())

    # 2. Rebuild the ABS call for every pitch and check it against ABS rulings.
    radius, grid = calibrate_radius(df)
    grid.to_csv(OUT / "abs_radius_calibration.csv", index=False)
    df = add_abs_margin(df, radius)
    loc = df[df["plate_x"].notna() & df["sz_top"].notna()].copy()
    chl = loc[loc["challenged"]]
    summary["abs_model"] = {
        "ball_radius_in": radius,
        "agreement_with_abs_rulings": float((chl["abs_call"] == chl["final"]).mean()),
        "challenged_pitches_checked": int(len(chl)),
        "pitches_without_location": int(len(df) - len(loc)),
    }

    # 3. Umpire accuracy on every called pitch (the original call, pre-review).
    loc["ump_right"] = loc["orig"] == loc["abs_call"]
    ump = [rate_row("all called pitches", loc["ump_right"].sum(), len(loc))]
    for call, label in [("B", "called ball"), ("S", "called strike")]:
        s = loc[loc["orig"] == call]
        ump.append(rate_row(label, s["ump_right"].sum(), len(s)))
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 99)]:
        s = loc[(loc["margin"].abs() >= lo) & (loc["margin"].abs() < hi)]
        ump.append(rate_row(f"|margin| {lo}-{hi} in", s["ump_right"].sum(), len(s)))
    pd.DataFrame(ump).to_csv(OUT / "umpire_accuracy.csv", index=False)

    # 4. Are catchers just guessing? At the same distance from the edge,
    #    compare how often a ball call was really a strike (what a random
    #    challenge would win) with how often the catcher's challenges won.
    balls = loc[loc["orig"] == "B"]
    strikes = loc[loc["orig"] == "S"]
    bands = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)]
    rows = []
    for who, pool in [("catcher", balls), ("batter", strikes)]:
        right_if = "S" if who == "catcher" else "B"
        for lo, hi in bands:
            s = pool[(pool["margin"].abs() >= lo) & (pool["margin"].abs() < hi)]
            c = s[s["role"] == who]
            rows.append({
                "challenger": who, "band_in": f"{lo}-{hi if hi < 99 else '+'}",
                "calls": len(s),
                "base_rate_wrong_call": (s["abs_call"] == right_if).mean(),
                "challenges": len(c),
                "challenge_win_rate": c["overturned"].mean() if len(c) else np.nan,
                "umpire_accuracy_all_calls_in_band": loc.loc[
                    (loc["margin"].abs() >= lo) & (loc["margin"].abs() < hi), "ump_right"].mean(),
            })
    pd.DataFrame(rows).to_csv(OUT / "difficulty_matched.csv", index=False)

    # 5. Who reads the zone more sharply? Fit each person's yes-curve against
    #    the true distance from the edge and compare the blur (sigma, inches).
    #    Umpire: P(calls strike). Catcher: P(challenges | ball call, team has a
    #    challenge). Batter: P(challenges | strike call, team has a challenge).
    has_def = (balls["def_lost"] < 2) | (balls["inning"] >= 10)
    has_off = (strikes["off_lost"] < 2) | (strikes["inning"] >= 10)
    cat_pool = balls[has_def & balls["role"].isin(["", "catcher"])]
    bat_pool = strikes[has_off & strikes["role"].isin(["", "batter"])]
    fits = {
        "umpire": curve_with_ci(loc, (loc["orig"] == "S")),
        "catcher": curve_with_ci(cat_pool, cat_pool["role"] == "catcher"),
        # The batter's "yes" is "that was a ball", so flip the axis.
        "batter": curve_with_ci(bat_pool.assign(margin=-bat_pool["margin"]), bat_pool["role"] == "batter"),
    }
    # Catchers' threshold moves with the stakes far more than an umpire's,
    # and mixing thresholds blurs a curve. Compare like with like by count.
    def stakes(s):
        return np.where((s["strikes"] == 2) | (s["balls"] == 3), "2 strikes or 3 balls", "other counts")
    by_count = {}
    for v in ["2 strikes or 3 balls", "other counts"]:
        u = loc[stakes(loc) == v]
        c = cat_pool[stakes(cat_pool) == v]
        by_count[v] = {
            "umpire": curve_with_ci(u, u["orig"] == "S", reps=100),
            "catcher": curve_with_ci(c, c["role"] == "catcher", reps=100),
        }
    by_edge = {}
    for edge in ["side", "top", "bottom"]:
        u = loc[loc["edge"] == edge]
        c = cat_pool[cat_pool["edge"] == edge]
        by_edge[edge] = {
            "umpire": curve_with_ci(u, u["orig"] == "S", reps=100),
            "catcher": curve_with_ci(c, c["role"] == "catcher", reps=100),
        }
    summary["sharpness"] = fits
    summary["sharpness_by_count"] = by_count
    summary["sharpness_by_edge"] = by_edge
    summary["shared_misread_sensitivity"] = shared_misread_sensitivity(cat_pool, fits["umpire"])
    summary["clone"] = {
        "catcher_actual_win_rate": float(cat_pool.loc[cat_pool["role"] == "catcher", "overturned"].mean()),
        "catcher_model": clone_umpire(cat_pool, fits["catcher"], fits["catcher"]["sigma"]),
        "umpire_eyes": clone_umpire(cat_pool, fits["catcher"], fits["umpire"]["sigma"]),
    }

    # Empirical curves for plotting.
    emp = []
    for who, pool, yes, flip_x in [
        ("umpire", loc, loc["orig"] == "S", False),
        ("catcher", cat_pool, cat_pool["role"] == "catcher", False),
        ("batter", bat_pool, bat_pool["role"] == "batter", True),
    ]:
        x = -pool["margin"] if flip_x else pool["margin"]
        b = pd.cut(x, np.arange(-4, 4.001, 0.25))
        g = pd.DataFrame({"bin": b, "yes": yes.astype(int)}).groupby("bin", observed=True)["yes"].agg(["size", "sum"])
        for interval, r in g.iterrows():
            emp.append({"who": who, "margin_mid": interval.mid, "n": int(r["size"]), "yes": int(r["sum"])})
    pd.DataFrame(emp).to_csv(OUT / "curves_empirical.csv", index=False)

    # 6. What catchers left on the table: umpire balls that ABS calls strikes.
    miss = balls[balls["abs_call"] == "S"].copy()
    miss["available"] = (miss["def_lost"] < 2) | (miss["inning"] >= 10)
    left = []
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 99)]:
        s = miss[(-miss["margin"] >= lo) & (-miss["margin"] < hi)]
        a = s[s["available"]]
        left.append({
            "missed_by_in": f"{lo}-{hi if hi < 99 else '+'}",
            "umpire_misses": len(s),
            "with_challenge_available": len(a),
            "catcher_challenged": int((a["role"] == "catcher").sum()),
            "pitcher_challenged": int((a["role"] == "pitcher").sum()),
            "share_challenged_by_defense": a["role"].isin(["catcher", "pitcher", "other"]).mean(),
        })
    pd.DataFrame(left).to_csv(OUT / "missed_calls_left_on_table.csv", index=False)

    # 7. Catcher win rate by game situation (strategy, not just eyesight).
    cc = loc[loc["role"] == "catcher"].copy()
    cc["inning_grp"] = pd.cut(cc["inning"], [0, 3, 6, 9, 30], labels=["1-3", "4-6", "7-9", "10+"])
    cc["count"] = cc["balls"].astype(str) + "-" + cc["strikes"].astype(str)
    cc["two_left"] = np.where(cc["def_lost"] == 0, "2 left", "1 left")
    sit = []
    for col in ["inning_grp", "two_left", "count"]:
        for v, s in cc.groupby(col, observed=True):
            sit.append(rate_row(f"{col}={v}", s["overturned"].sum(), len(s),
                                median_abs_margin=float(s["margin"].abs().median())))
    pd.DataFrame(sit).to_csv(OUT / "catcher_by_situation.csv", index=False)

    # How wrong was the loser? Distribution of misses on catcher challenges.
    cc[KEYS + ["fielder_2", "margin", "edge", "overturned", "balls", "strikes", "inning"]].to_csv(
        OUT / "catcher_challenge_margins.csv", index=False)
    won, lost = cc[cc["overturned"]], cc[~cc["overturned"]]
    summary["catcher_challenges_detail"] = {
        "won_median_inches_inside": float((-won["margin"]).median()),
        "lost_median_inches_outside": float(lost["margin"].median()),
        "lost_within_1in": float((lost["margin"] < 1).mean()),
        "won_by_1in_or_more": float((won["margin"] <= -1).mean()),
    }

    # 8. Per catcher.
    names = catcher_names(df["fielder_2"].dropna().astype(int).unique())
    opp = miss[miss["available"]]
    per = []
    for cid, s in loc.groupby("fielder_2"):
        c = s[s["role"] == "catcher"]
        o = opp[opp["fielder_2"] == cid]
        big = o[o["margin"] <= -1]
        per.append({
            "catcher_id": int(cid), "catcher": names.get(int(cid), str(cid)),
            "called_pitches": len(s), "challenges": len(c), "won": int(c["overturned"].sum()),
            "win_rate": c["overturned"].mean() if len(c) else np.nan,
            "ump_misses_available": len(o),
            "share_of_misses_challenged": (o["role"] == "catcher").mean() if len(o) else np.nan,
            "misses_1in_plus": len(big),
            "share_of_1in_plus_challenged": (big["role"] == "catcher").mean() if len(big) else np.nan,
        })
    per = pd.DataFrame(per).sort_values("challenges", ascending=False)
    per.to_csv(OUT / "catchers.csv", index=False)

    summary["by_role"] = by_role.to_dict(orient="records")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


def catcher_names(ids) -> dict:
    names = {}
    ids = list(map(int, ids))
    for i in range(0, len(ids), 100):
        chunk = ",".join(map(str, ids[i:i + 100]))
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={chunk}", timeout=60)
        r.raise_for_status()
        names.update({p["id"]: p["fullName"] for p in r.json()["people"]})
    return names


if __name__ == "__main__":
    main()
