"""Data for the zone visualizations (site/who-knows-the-zone.html).

Every pitch is placed on one standard zone so batters of different heights
line up: horizontal position is inches from the middle of the plate (catcher's
view), and height keeps each pitch's exact distance to the nearer of the top or
bottom edge of that batter's ABS zone, measured against an average batter's
zone. Distances to the edge, which decide ABS calls, are preserved exactly.

Writes results/zone_viz.json and builds site/who-knows-the-zone.html from
site/template.html.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import analyze as A

RADIUS = 1.45
X_RANGE = (-20, 20)
CELL = 1.0
MIN_CELL_N = 25
PITCH_GROUPS = {"FF": "Four-seam", "SI": "Sinker", "FC": "Cutter",
                "SL": "Breaking", "ST": "Breaking", "CU": "Breaking", "KC": "Breaking", "SV": "Breaking",
                "CH": "Offspeed", "FS": "Offspeed", "FO": "Offspeed"}


def standard_coords(df: pd.DataFrame, bot: float, top: float) -> pd.DataFrame:
    z = df["plate_z"] * 12
    b, t = df["sz_bot"] * 12, df["sz_top"] * 12
    nearer_bottom = (z - b) < (t - z)
    df["zx"] = df["plate_x"] * 12
    df["zz"] = np.where(nearer_bottom, bot + (z - b), top + (z - t))
    return df


def grid(df: pd.DataFrame, yes: pd.Series, z_range) -> list:
    xb = np.floor((df["zx"] - X_RANGE[0]) / CELL).astype(int)
    zb = np.floor((df["zz"] - z_range[0]) / CELL).astype(int)
    nx, nz = int((X_RANGE[1] - X_RANGE[0]) / CELL), int((z_range[1] - z_range[0]) / CELL)
    ok = (xb >= 0) & (xb < nx) & (zb >= 0) & (zb < nz)
    g = pd.DataFrame({"i": xb[ok], "j": zb[ok], "y": yes[ok].astype(int)}).groupby(["i", "j"])["y"].agg(["size", "sum"])
    g = g[g["size"] >= MIN_CELL_N]
    return [[int(i), int(j), int(r["size"]), round(r["sum"] / r["size"], 4)] for (i, j), r in g.iterrows()]


def benchmark(pool: pd.DataFrame, ump: dict, rho: float, ceil: float, sigmas) -> list:
    """Expected head-to-head win rate for a challenger with blur sigma making the
    catchers' number of challenges on the catchers' pitches (ball calls with a
    challenge available), with misreads correlated rho with the umpire's."""
    s = pool[(pool["margin"] >= A.BIN_EDGES[0]) & (pool["margin"] < A.BIN_EDGES[-1])]
    b = np.digitize(s["margin"], A.BIN_EDGES) - 1
    n = np.bincount(b, minlength=len(A.BIN_MID)).astype(float)
    keep = n > 0
    x, n = A.BIN_MID[keep], n[keep]
    target = (s["role"] == "catcher").sum()
    bu = (ump["mu"] - x) / ump["sigma"]
    mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    out = []
    for sig in sigmas:
        def p(mu):
            a = (mu - x) / sig
            if rho == 0:
                cond = stats.norm.cdf(a)
            else:
                both = mvn.cdf(np.column_stack([a, bu]))
                cond = (stats.norm.cdf(a) - both) / np.clip(stats.norm.sf(bu), 1e-12, None)
            return ceil * np.clip(cond, 0, 1) * n
        mu = optimize.brentq(lambda mu: p(mu).sum() - target, -15, 15, xtol=1e-4)
        pr = p(mu)
        out.append([round(float(sig), 3), round(float(pr[x < 0].sum() / pr.sum()), 4)])
    return out


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    df = A.add_abs_margin(A.load(), RADIUS)
    loc = df[df["plate_x"].notna() & df["sz_top"].notna()].copy()
    bot, top = round(loc["sz_bot"].mean() * 12, 1), round(loc["sz_top"].mean() * 12, 1)
    loc = standard_coords(loc, bot, top)
    z_range = (int(np.floor(bot - 14)), int(np.ceil(top + 12)))

    balls = loc[loc["orig"] == "B"]
    has_def = (balls["def_lost"] < 2) | (balls["inning"] >= 10)
    cat_pool = balls[has_def & balls["role"].isin(["", "catcher"])]

    names = A.catcher_names(loc["fielder_2"].dropna().astype(int).unique())
    cc = loc[loc["role"] == "catcher"]
    name_list = sorted({names.get(int(i), str(int(i))) for i in cc["fielder_2"]})
    name_idx = {n: k for k, n in enumerate(name_list)}
    challenges = [
        [round(r.zx, 1), round(r.zz, 1), int(r.overturned), name_idx[names.get(int(r.fielder_2), str(int(r.fielder_2)))],
         f"{int(r.balls)}-{int(r.strikes)}", int(r.inning), round(float(r.margin), 2), r.edge]
        for r in cc.itertuples()
    ]

    miss = balls[(balls["abs_call"] == "S") & has_def]
    misses = [[round(r.zx, 1), round(r.zz, 1), int(r.role in ("catcher", "pitcher")), round(float(-r.margin), 2)]
              for r in miss.itertuples()]

    ump = summary["sharpness"]["umpire"]
    sens = {s["rho"]: s for s in A.shared_misread_sensitivity(cat_pool, ump, rhos=(0.0, 0.4))}
    sigmas = np.round(np.arange(0.4, 3.01, 0.1), 2)
    bench = {
        "independent": benchmark(cat_pool, ump, 0.0, sens[0.0]["catcher_ceil"], sigmas),
        "shared": benchmark(cat_pool, ump, 0.4, sens[0.4]["catcher_ceil"], sigmas),
        "umpire_sigma": ump["sigma"],
        "catcher_sigma_independent": sens[0.0]["catcher_sigma"],
        "catcher_sigma_shared": sens[0.4]["catcher_sigma"],
        "equal_eyes_independent": benchmark(cat_pool, ump, 0.0, sens[0.0]["catcher_ceil"], [ump["sigma"]])[0][1],
        "equal_eyes_shared": benchmark(cat_pool, ump, 0.4, sens[0.4]["catcher_ceil"], [ump["sigma"]])[0][1],
        "catcher_actual": float(cc["overturned"].mean()),
    }

    by_type = []
    lp = loc[(loc["margin"] >= -6) & (loc["margin"] < 6)]
    for g in ["Four-seam", "Sinker", "Cutter", "Breaking", "Offspeed"]:
        u = lp[lp["pitch_type"].map(PITCH_GROUPS) == g]
        c = cat_pool[cat_pool["pitch_type"].map(PITCH_GROUPS) == g]
        fu = A.curve_with_ci(u, u["orig"] == "S", reps=60)
        fc = A.curve_with_ci(c, c["role"] == "catcher", reps=60)
        by_type.append({"group": g, "umpire": [fu["sigma"], fu["sigma_lo"], fu["sigma_hi"]],
                        "catcher": [fc["sigma"], fc["sigma_lo"], fc["sigma_hi"]], "challenges": fc["n_yes"]})
    by_edge = [{"group": e.capitalize() if e != "side" else "Sides",
                "umpire": [v["umpire"]["sigma"], v["umpire"]["sigma_lo"], v["umpire"]["sigma_hi"]],
                "catcher": [v["catcher"]["sigma"], v["catcher"]["sigma_lo"], v["catcher"]["sigma_hi"]],
                "challenges": v["catcher"]["n_yes"]}
               for e, v in summary["sharpness_by_edge"].items()]

    data = {
        "zone": {"half_width": A.HALF_WIDTH_IN, "bot": bot, "top": top, "radius": RADIUS},
        "x_range": X_RANGE, "z_range": z_range, "cell": CELL,
        "ump_grid": grid(loc, loc["orig"] == "S", z_range),
        "catcher_grid": grid(cat_pool, cat_pool["role"] == "catcher", z_range),
        "challenges": challenges, "names": name_list,
        "misses": misses,
        "benchmark": bench,
        "sharpness": {k: [v["sigma"], v["sigma_lo"], v["sigma_hi"]] for k, v in summary["sharpness"].items()},
        "by_type": by_type, "by_edge": by_edge,
        "totals": {
            "games": summary["games"], "called": summary["called_pitches"],
            "ump_accuracy": float((loc["orig"] == loc["abs_call"]).mean()),
            "abs_agreement": summary["abs_model"]["agreement_with_abs_rulings"],
            "catcher_challenges": int(len(cc)), "catcher_won": int(cc["overturned"].sum()),
            "misses_available": int(len(miss)),
            "misses_challenged": int(miss["role"].isin(["catcher", "pitcher"]).sum()),
            "big_misses": int((miss["margin"] <= -2).sum()),
            "big_misses_challenged": int((miss.loc[miss["margin"] <= -2, "role"].isin(["catcher", "pitcher"])).sum()),
        },
    }
    Path("results/zone_viz.json").write_text(json.dumps(data, separators=(",", ":")))
    build_site()
    print(json.dumps(bench, indent=1)[:1500])
    print(json.dumps({"by_type": by_type, "totals": data["totals"], "bot": bot, "top": top, "z_range": z_range}, indent=1))


def build_site() -> None:
    """Inline the data into the page template (artifacts can't fetch it)."""
    tpl = Path("site/template.html").read_text()
    data = Path("results/zone_viz.json").read_text()
    Path("site/who-knows-the-zone.html").write_text(tpl.replace("/*__DATA__*/null", data))


if __name__ == "__main__":
    import sys

    build_site() if "--build-only" in sys.argv else main()
