import argparse
import copy
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from conditional_ddpm_module import (
    Config,
    ConditionalUNet,
    Diffusion,
    normalize_img,
    denormalize_img,
    ddpm_step,
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


def create_loaders(clean_path, corrupted_path, batch_size, val_split, seed):

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


# =========================================================
# Train
# =========================================================

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(device=device, timesteps=args.timesteps)

    set_seed(args.seed)

    train_loader, val_loader = create_loaders(
        args.clean_path,
        args.corrupted_path,
        args.batch_size,
        args.val_split,
        args.seed,
    )

    model = ConditionalUNet(
        dim=config.dim,
        dim_mults=config.dim_mults,
        in_channels=3,
        out_channels=1,
    ).to(device)

    ema_model = copy.deepcopy(model).eval()

    diffusion = Diffusion(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = GradScaler(enabled=device.type == "cuda")

    global_step = 0

    for epoch in range(args.epochs):

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

            with autocast(enabled=device.type == "cuda"):
                pred_noise = model(x_t, t, cond)
                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(0.999).add_(p.data, alpha=1 - 0.999)

            losses.append(loss.item())
            global_step += 1

        print(f"Epoch {epoch+1} | Loss {np.mean(losses):.6f}")

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
