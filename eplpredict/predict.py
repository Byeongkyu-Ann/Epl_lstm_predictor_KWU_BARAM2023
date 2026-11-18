"""Two-stage inference without any GUI code.

Example:
    python -m eplpredict.predict "Man Utd" "Man City"
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import CHECKPOINTS, MODEL_DIR, RESULT_LABELS, STAGE1_TARGETS, team_to_index
from .model import load_checkpoint

STAGE1_ORDER = ["possession", "shot", "touch", "pass", "clearance"]


@dataclass
class Prediction:
    home: str
    away: str
    home_shares: dict   # home share of each statistic in [0, 1]
    result: int         # 0 = home loss, 1 = draw, 2 = home win
    scores: list        # softmax of the Stage 2 outputs

    @property
    def label(self) -> str:
        return RESULT_LABELS[self.result]


def _resolve(model_dir: Path, key: str) -> Path:
    """Use the new checkpoint name if it exists, otherwise the name of the original build."""
    new_name, legacy_name = CHECKPOINTS[key]
    for name in (new_name, legacy_name):
        if (model_dir / name).exists():
            return model_dir / name
    raise FileNotFoundError(f"Neither {new_name} nor {legacy_name} found in {model_dir}. "
                            "Train the models first or copy the checkpoints there.")


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        model_dir = Path(model_dir)
        self.stage1 = {k: load_checkpoint(_resolve(model_dir, k)) for k in STAGE1_ORDER}
        self.stage2 = load_checkpoint(_resolve(model_dir, "result"))

    @torch.no_grad()
    def predict(self, home: str, away: str) -> Prediction:
        teams = torch.tensor([[[team_to_index(home), team_to_index(away)]]], dtype=torch.float32)
        shares = {k: float(self.stage1[k](teams).item()) for k in STAGE1_ORDER}
        shares = {k: min(max(v, 0.0), 1.0) for k, v in shares.items()}

        features = torch.cat([teams, torch.tensor([[[shares[k] for k in STAGE1_ORDER]]])], dim=2)
        scores = torch.softmax(self.stage2(features), dim=1)[0]
        return Prediction(home, away, shares, int(scores.argmax()), scores.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home")
    parser.add_argument("away")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    result = Predictor(args.model_dir).predict(args.home, args.away)
    print(f"{result.home} (home) vs {result.away} (away)")
    for key in STAGE1_ORDER:
        share = result.home_shares[key] * 100
        print(f"  {key:<10} {share:5.1f}% : {100 - share:5.1f}%")
    print(f"  result     {result.label}")


if __name__ == "__main__":
    main()
