"""Stage 2: predict the match result from the two teams and the Stage 1 statistics.

The model is trained on the true statistics of each match. At prediction time
the app feeds it the Stage 1 predictions instead.

Example:
    python -m eplpredict.train_stage2 --epochs 1000
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn

from .config import MODEL_DIR, PROCESSED_CSV, STAGE2_FEATURES
from .data import chronological_split, make_sequences
from .model import LSTMModel, save_checkpoint
from .training import fit, get_device, predict, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(PROCESSED_CSV))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seq-length", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=30)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="Save the training loss curve.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    df = pd.read_csv(args.data)
    train_df, test_df = chronological_split(df, args.train_ratio)
    print(f"train {len(train_df)} rows, test {len(test_df)} rows, device {device}")
    print(f"features: {STAGE2_FEATURES}")

    x_tr, y_tr = make_sequences(train_df[STAGE2_FEATURES].values,
                                train_df["result"].values, args.seq_length)
    x_te, y_te = make_sequences(test_df[STAGE2_FEATURES].values,
                                test_df["result"].values, args.seq_length)
    y_tr, y_te = y_tr.astype("int64"), y_te.astype("int64")

    model = LSTMModel(len(STAGE2_FEATURES), args.hidden_size, args.num_layers, output_size=3)
    history = fit(model, x_tr, y_tr, nn.CrossEntropyLoss(), args.lr, args.epochs, device)

    pred = predict(model, x_te, device).argmax(dim=1).numpy()
    majority = int(np.bincount(y_tr, minlength=3).argmax())
    print(f"test accuracy {accuracy_score(y_te, pred):.4f}  "
          f"(always predicting class {majority}: {accuracy_score(y_te, np.full_like(y_te, majority)):.4f})")
    print("confusion matrix (rows = true loss/draw/win, columns = predicted):")
    print(confusion_matrix(y_te, pred, labels=[0, 1, 2]))

    save_checkpoint(model, MODEL_DIR / "stage2_result.pt")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(history)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("Stage 2 training loss")
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "stage2_loss.png", dpi=150)


if __name__ == "__main__":
    main()
