"""Lightweight PyTorch training loop (no Lightning dependency)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str = None,
    verbose: bool = True,
) -> dict:
    """Generic training loop for PyTorch regression models.

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader or None
    epochs       : int
    lr           : float
    weight_decay : float
    grad_clip    : float   Gradient clipping max norm (0 = disabled).
    device       : str     'cpu' | 'cuda' | None (auto).
    verbose      : bool

    Returns
    -------
    history : dict with 'train_loss' and (if val_loader) 'val_loss'.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.MSELoss()
    history: dict[str, list] = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr += loss.item() * len(xb)
        history["train_loss"].append(tr / len(train_loader.dataset))
        sched.step()

        if val_loader is not None:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += crit(model(xb), yb).item() * len(xb)
            history["val_loss"].append(vl / len(val_loader.dataset))

        if verbose and ep % 10 == 0:
            msg = f"Epoch {ep:3d}/{epochs} | train {history['train_loss'][-1]:.5f}"
            if val_loader:
                msg += f" | val {history['val_loss'][-1]:.5f}"
            print(msg)

    return history
