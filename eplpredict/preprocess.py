"""Turn raw crawled matches into model features.

Example:
    python -m eplpredict.preprocess --inputs data/raw_2012_13.csv data/raw_2013_14.csv

Pass the inputs from the oldest to the newest season. The row order is kept,
so the chronological train/test split in the training scripts stays valid.

Added columns
    home_idx, away_idx            team index from config.TEAM_INDEX
    <stat>_share                  home / (home + away) for possession, shots, touches,
                                  passes, tackles and clearances
    home_sot_ratio, away_sot_ratio  shots on target / shots
    result                        0 = home loss, 1 = draw, 2 = home win
"""
import argparse

import numpy as np
import pandas as pd

from .config import LEGACY_COLUMNS, PROCESSED_CSV, RAW_STATS, team_to_index

SHARE_STATS = ["possession", "shots", "touches", "passes", "tackles", "clearances"]
SHARE_NAMES = {"shots": "shot", "touches": "touch", "passes": "pass",
               "tackles": "tackle", "clearances": "clearance", "possession": "possession"}


def safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float) -> pd.Series:
    """numerator / denominator, with `default` where the denominator is 0."""
    ratio = numerator / denominator.replace(0, np.nan)
    return ratio.fillna(default)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=LEGACY_COLUMNS).copy()

    required = ["home_team", "away_team", "goal_diff"]
    required += [f"{side}_{stat}" for stat in RAW_STATS for side in ("home", "away")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)  # rows with a missing statistic are dropped

    teams = set(df["home_team"]) | set(df["away_team"])
    unknown = sorted(t for t in teams if _safe_index(t) is None)
    if unknown:
        raise ValueError(f"Teams without an index in config.TEAM_INDEX: {unknown}")

    df["home_idx"] = df["home_team"].map(team_to_index)
    df["away_idx"] = df["away_team"].map(team_to_index)

    # A share is 0.5 when both teams have 0 events.
    for stat in SHARE_STATS:
        total = df[f"home_{stat}"] + df[f"away_{stat}"]
        df[f"{SHARE_NAMES[stat]}_share"] = safe_ratio(df[f"home_{stat}"], total, 0.5)

    df["home_sot_ratio"] = safe_ratio(df["home_shots_on_target"], df["home_shots"], 0.0)
    df["away_sot_ratio"] = safe_ratio(df["away_shots_on_target"], df["away_shots"], 0.0)

    df["result"] = np.select([df["goal_diff"] > 0, df["goal_diff"] < 0], [2, 0], default=1)
    return df


def _safe_index(name: str):
    try:
        return team_to_index(name)
    except KeyError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Raw CSV files, oldest season first.")
    parser.add_argument("--out", default=str(PROCESSED_CSV))
    args = parser.parse_args()

    raw = pd.concat([pd.read_csv(p) for p in args.inputs], ignore_index=True)
    processed = preprocess(raw)
    processed.to_csv(args.out, index=False)

    counts = processed["result"].value_counts(normalize=True).sort_index()
    print(f"{len(raw)} raw rows -> {len(processed)} rows saved to {args.out}")
    print("Class share  loss {:.3f}  draw {:.3f}  win {:.3f}".format(
        *(counts.get(c, 0.0) for c in (0, 1, 2))))


if __name__ == "__main__":
    main()
