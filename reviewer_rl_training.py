import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from reviewer_rl import ReviewerDataset, Reviewer

@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.1          # used only if val_dataset is not provided
    early_stopping_patience: int = 15
    use_amp: bool = True               # mixed precision, only kicks in on CUDA
    num_workers: int = 0
    device: Optional[str] = None       # auto-detected if None
    log_every: int = 1                 # print every N epochs


def _warmup_cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then cosine decay to 0, as a multiplier on the base LR."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_reviewer(
    reviewer: Reviewer,
    train_dataset: ReviewerDataset | Subset[ReviewerDataset],
    val_dataset: ReviewerDataset | Subset[ReviewerDataset],
    config: TrainConfig = field(default_factory=TrainConfig),
) -> tuple[dict, dict]:
    """
    Trains `model` to predict a scalar quality score from an embedding.

    Assumes each dataset item is a tuple (embedding, target), where target is
    a scalar (or shape (1,)) float. Adjust the unpacking in the loop below if
    your Dataset yields something else (e.g. a dict).

    Returns a history dict with per-epoch train/val losses and the best
    validation loss achieved, and leaves `model` loaded with its best weights.
    """
    if isinstance(config, TrainConfig) is False:
        config = TrainConfig()

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reviewer = reviewer.to(device)


    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(reviewer.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _warmup_cosine_lr(step, total_steps, warmup_steps)
    )

    use_amp = config.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(reviewer.state_dict())
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "lr": [], "best_val_loss": None}

    global_step = 0
    for epoch in range(config.epochs):
        reviewer.train()
        running_loss = 0.0
        for embeddings, targets in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = reviewer(embeddings).view(-1, 1)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(reviewer.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * embeddings.size(0)
            global_step += 1

        train_loss = running_loss / len(train_dataset)

        # --- validation ---
        reviewer.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for embeddings, targets in val_loader:
                embeddings = embeddings.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).float().view(-1, 1)
                preds = reviewer(embeddings).view(-1, 1)
                val_running_loss += criterion(preds, targets).item() * embeddings.size(0)
        val_loss = val_running_loss / len(val_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if (epoch + 1) % config.log_every == 0:
            print(f"epoch {epoch+1:>4}/{config.epochs} | "
                  f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        # --- early stopping / checkpointing on best val loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(reviewer.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for "
                      f"{config.early_stopping_patience} epochs). Best val_loss: {best_val_loss:.4f}")
                break

    last_state = reviewer.state_dict()
    reviewer.load_state_dict(best_state)
    history["best_val_loss"] = best_val_loss
    return history, last_state