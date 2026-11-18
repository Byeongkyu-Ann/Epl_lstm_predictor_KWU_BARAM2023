"""Stage 1: predict the home share of each match statistic from the two teams.

One single-output LSTM is trained per statistic.

Example:
    python -m eplpredict.train_stage1 --epochs 500
"""
import argparse

import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn

from .config import MODEL_DIR, PROCESSED_CSV, STAGE1_FEATURES, STAGE1_TARGETS
from .data import chronological_split, make_sequences
from .model import LSTMModel, save_checkpoint
from .training import fit, get_device, predict, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(PROCESSED_CSV))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seq-length", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    df = pd.read_csv(args.data)
    train_df, test_df = chronological_split(df, args.train_ratio)
    print(f"train {len(train_df)} rows, test {len(test_df)} rows, device {device}")

    for name, column in STAGE1_TARGETS.items():
        print(f"[{name}] target = {column}")
        x_tr, y_tr = make_sequences(train_df[STAGE1_FEATURES].values,
                                    train_df[[column]].values.astype("float32"), args.seq_length)
        x_te, y_te = make_sequences(test_df[STAGE1_FEATURES].values,
                                    test_df[[column]].values.astype("float32"), args.seq_length)

        model = LSTMModel(len(STAGE1_FEATURES), args.hidden_size, args.num_layers, output_size=1)
        fit(model, x_tr, y_tr, nn.MSELoss(), args.lr, args.epochs, device)

        pred = predict(model, x_te, device).numpy()
        baseline = float(((y_te - y_tr.mean()) ** 2).mean())
        mse = float(((y_te - pred) ** 2).mean())
        print(f"  test R2 {r2_score(y_te, pred):.4f}  MSE {mse:.5f}  "
              f"(train-mean baseline MSE {baseline:.5f})")
        save_checkpoint(model, MODEL_DIR / f"stage1_{name}.pt")


if __name__ == "__main__":
    main()
