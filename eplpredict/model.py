"""LSTM model and checkpoint helpers."""
import sys
from pathlib import Path

import torch
from torch import nn


class LSTMModel(nn.Module):
    """Stacked LSTM followed by a linear head on the last hidden state.

    The forward pass returns raw outputs. Use them as logits for classification
    and as regression values for share prediction.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # h0 and c0 default to zeros
        return self.fc(out[:, -1, :])

    def config(self) -> dict:
        return dict(input_size=self.input_size, hidden_size=self.hidden_size,
                    num_layers=self.num_layers, output_size=self.output_size)


def save_checkpoint(model: LSTMModel, path: Path) -> None:
    """Save weights together with the constructor arguments."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config(), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> LSTMModel:
    """Load a checkpoint written by save_checkpoint or a legacy full-model pickle.

    Legacy files were saved with torch.save(model) from a script where the class
    lived in __main__. Unpickling them needs the class under that name.
    """
    main = sys.modules["__main__"]
    if not hasattr(main, "LSTMModel"):
        main.LSTMModel = LSTMModel
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        model = LSTMModel(**obj["config"])
        model.load_state_dict(obj["state_dict"])
    return model.to(device).eval()
