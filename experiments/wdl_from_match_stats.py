"""Reference experiment: predict win / draw / loss from the true match statistics.

The model sees both teams' possession, shots, shots on target, touches, passes,
tackles and clearances of the finished match. It shows how much of the result
is explained by in-match statistics and acts as a reference for the two-stage
pipeline, which has to predict those statistics first.

Run from the repository root:
    python -m experiments.wdl_from_match_stats --epochs 500
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from eplpredict.config import PROCESSED_CSV
from eplpredict.data import chronological_split, make_sequences
from eplpredict.model import LSTMModel
from eplpredict.training import fit, get_device, predict, seed_everything

STATS = ["possession", "shots", "shots_on_target", "touches", "passes", "tackles", "clearances"]
FEATURES = [f"{side}_{s}" for s in STATS for side in ("home", "away")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(PROCESSED_CSV))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seq-length", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=30)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    df = pd.read_csv(args.data)
    train_df, test_df = chronological_split(df, args.train_ratio)

    scaler = MinMaxScaler().fit(train_df[FEATURES].values)  # fitted on training rows only
    x_tr, y_tr = make_sequences(scaler.transform(train_df[FEATURES].values),
                                train_df["result"].values, args.seq_length)
    x_te, y_te = make_sequences(scaler.transform(test_df[FEATURES].values),
                                test_df["result"].values, args.seq_length)
    y_tr, y_te = y_tr.astype("int64"), y_te.astype("int64")

    model = LSTMModel(len(FEATURES), args.hidden_size, args.num_layers, output_size=3)
    fit(model, x_tr, y_tr, nn.CrossEntropyLoss(), args.lr, args.epochs, device)

    pred = predict(model, x_te, device).argmax(dim=1).numpy()
    majority = int(np.bincount(y_tr, minlength=3).argmax())
    print(f"test accuracy {accuracy_score(y_te, pred):.4f}  "
          f"(always predicting class {majority}: {accuracy_score(y_te, np.full_like(y_te, majority)):.4f})")


if __name__ == "__main__":
    main()
