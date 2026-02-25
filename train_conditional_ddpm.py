import argparse
import copy
import os
import random
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from conditional_ddpm_module import (
    Config,
    ConditionalUNet,
    Diffusion,
    normalize_img,
)


# =========================================================
# Dataset
# =========================================================

class InpaintingNpyDataset(Dataset):
    def __init__(self, corrupted: np.ndarray, clean: np.ndarray):
        if corrupted.ndim == 3:
            corrupted = corrupted[..., None]
        if clean.ndim == 3:
            clean = clean[..., None]

        self.corrupted = corrupted.astype(np.float32)
        self.clean = clean.astype(np.float32)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        corrupted = self.corrupted[idx]
        clean = self.clean[idx]

        mask = (corrupted != 0).astype(np.float32)

        corrupted_t = torch.from_numpy(corrupted).permute(2, 0, 1)
        clean_t = torch.from_numpy(clean).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).permute(2, 0, 1)

        corrupted_t = normalize_img(corrupted_t)
        clean_t = normalize_img(clean_t)

        return corrupted_t, clean_t, mask_t




# =========================================================
# Utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    clean_path: str,
    corrupted_path: str,
    batch_size: int,
    val_split: float,
    seed: int,
):

    clean = np.load(clean_path)
    corrupted = np.load(corrupted_path)

    c_train, c_val, y_train, y_val = train_test_split(
        corrupted, clean, test_size=val_split, random_state=seed
    )

    train_ds = InpaintingNpyDataset(c_train, y_train)
    val_ds = InpaintingNpyDataset(c_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k], alpha=1 - decay)
            else:
                v.copy_(msd[k])


def save_checkpoint(path, model, ema_model, optimizer, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def evaluate(model, diffusion, loader, device, use_amp):
    model.eval()
    losses = []

    with torch.no_grad():
        for corrupted, clean, mask in loader:

            corrupted = corrupted.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            cond = torch.cat([corrupted, mask], dim=1)

            t = torch.randint(0, diffusion.config.timesteps, (clean.size(0),), device=device)
            x_t, noise = diffusion.q_sample(clean, t)

            with autocast(enabled=use_amp):
                pred_noise = model(x_t, t, cond)
                loss = F.mse_loss(pred_noise, noise)

            losses.append(loss.item())

    return float(np.mean(losses))

# =========================================================
# Train
# =========================================================


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        device=device,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    set_seed(config.seed)

    train_loader, val_loader = create_dataloaders(
        args.clean_path,
        args.corrupted_path,
        config.batch_size,
        args.val_split,
        config.seed,
    )

    model = ConditionalUNet(
        dim=config.dim,
        dim_mults=config.dim_mults,
        in_channels=3,
        out_channels=1,
    ).to(device)

    ema_model = copy.deepcopy(model).eval()

    diffusion = Diffusion(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    global_step = 0

    for epoch in range(config.epochs):

        model.train()
        losses = []

        for corrupted, clean, mask in train_loader:

            corrupted = corrupted.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            cond = torch.cat([corrupted, mask], dim=1)

            t = torch.randint(0, config.timesteps, (clean.size(0),), device=device)
            x_t, noise = diffusion.q_sample(clean, t)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pred_noise = model(x_t, t, cond)
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, config.ema_decay)

            losses.append(loss.item())
            global_step += 1

        scheduler.step()

        train_loss = np.mean(losses)
        val_loss = evaluate(ema_model, diffusion, val_loader, device, use_amp)

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}"
        )

        save_checkpoint(
            args.checkpoint,
            model,
            ema_model,
            optimizer,
            epoch,
            global_step,
        )

# =========================================================
# CLI
# =========================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--clean-path", default="cifar10_grayscale_32x32.npy")
    p.add_argument("--corrupted-path", default="cifar10_grayscale_32x32_corrupted.npy")

    p.add_argument("--checkpoint", default="checkpoints/ddpm.pt")

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)

    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.2)

    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()
    train(args)
