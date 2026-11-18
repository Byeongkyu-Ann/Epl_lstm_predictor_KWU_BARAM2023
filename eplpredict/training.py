"""Small full-batch training loop used by all training scripts."""
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit(model: nn.Module, x: np.ndarray, y: np.ndarray, loss_fn: nn.Module,
        lr: float, epochs: int, device: torch.device, log_every: int = 100) -> list:
    """Train on the full dataset every epoch with Adam. Returns the loss history."""
    model.to(device).train()
    inputs = torch.from_numpy(x).float().to(device)
    targets = torch.from_numpy(y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        loss = loss_fn(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if epoch % log_every == 0 or epoch == epochs:
            print(f"  epoch {epoch}/{epochs}  loss {loss.item():.5f}")
    return history


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> torch.Tensor:
    model.to(device).eval()
    return model(torch.from_numpy(x).float().to(device)).cpu()
