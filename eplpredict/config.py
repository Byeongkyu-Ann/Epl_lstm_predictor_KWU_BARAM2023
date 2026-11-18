"""Shared paths, column names and constants."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
ASSET_DIR = ROOT / "assets"

PROCESSED_CSV = DATA_DIR / "processed_matches.csv"

# ---------------------------------------------------------------------------
# Team index
# ---------------------------------------------------------------------------
# Integer index per team. 35 is the strongest team and 1 is the weakest.
# This index is the only team-related input of the models.
TEAM_INDEX = {
    "Man City": 35, "Liverpool": 34, "Arsenal": 33, "Spurs": 32,
    "Chelsea": 31, "Man Utd": 30, "Newcastle": 29, "Aston Villa": 28,
    "Brighton": 27, "West Ham": 26, "Crystal Palace": 25, "Brentford": 24,
    "Wolves": 23, "Everton": 22, "Nott'm Forest": 21, "Leicester": 20,
    "Fulham": 19, "Burnley": 18, "Sheffield Utd": 17, "Bournemouth": 16,
    "Watford": 15, "Huddersfield": 14, "Leeds": 13, "West Brom": 12,
    "Southampton": 11, "Middlesbrough": 10, "Norwich": 9, "Swansea": 8,
    "Stoke": 7, "Cardiff": 6, "QPR": 5, "Reading": 4, "Wigan": 3,
    "Sunderland": 2, "Hull": 1,
}


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


_TEAM_LOOKUP = {_key(name): idx for name, idx in TEAM_INDEX.items()}


def team_to_index(name: str) -> int:
    """Map a team name to its index. Spaces and apostrophes are ignored."""
    try:
        return _TEAM_LOOKUP[_key(name)]
    except KeyError:
        raise KeyError(f"Unknown team '{name}'. Known teams: {sorted(TEAM_INDEX)}") from None


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
# Raw statistics written by the crawler. Each one has a home_ and an away_ column.
RAW_STATS = ["possession", "shots_on_target", "shots", "touches",
             "passes", "tackles", "clearances"]

# Column names used by the first crawler version. preprocess.py still accepts them.
LEGACY_COLUMNS = {
    "Hometeam": "home_team", "Awayteam": "away_team", "new_score": "goal_diff",
    "Match_week": "match_week",
    "H_possesion": "home_possession", "A_possesion": "away_possession",
    "H_shot": "home_shots", "A_shot": "away_shots",
    "H_shot_target": "home_shots_on_target", "A_shot_target": "away_shots_on_target",
    "H_touch": "home_touches", "A_touch": "away_touches",
    "H_passes": "home_passes", "A_passes": "away_passes",
    "H_tackles": "home_tackles", "A_tackles": "away_tackles",
    "H_clearance": "home_clearances", "A_clearance": "away_clearances",
}

# Stage 1 targets: share of each statistic that belongs to the home team.
# Key = model name. Value = column of the processed CSV.
STAGE1_TARGETS = {
    "possession": "possession_share",
    "shot": "shot_share",
    "touch": "touch_share",
    "pass": "pass_share",
    "clearance": "clearance_share",
}

TEAM_FEATURES = ["home_idx", "away_idx"]
STAGE1_FEATURES = TEAM_FEATURES
STAGE2_FEATURES = TEAM_FEATURES + [STAGE1_TARGETS[k] for k in
                                   ("possession", "shot", "touch", "pass", "clearance")]

# Result classes
LOSE, DRAW, WIN = 0, 1, 2
RESULT_LABELS = {LOSE: "HomeTeam Lose !!", DRAW: "Draw !!", WIN: "HomeTeam Win !!"}

# Checkpoint names. Names in parentheses are the file names of the original build.
CHECKPOINTS = {
    "possession": ("stage1_possession.pt", "Possesion_model_1.pth"),
    "shot": ("stage1_shot.pt", "Shot_model_1.pth"),
    "touch": ("stage1_touch.pt", "Touch_model_1.pth"),
    "pass": ("stage1_pass.pt", "Pass_model_1.pth"),
    "clearance": ("stage1_clearance.pt", "Clearance_model_1.pth"),
    "result": ("stage2_result.pt", "final_predict_model_2.pth"),
}
