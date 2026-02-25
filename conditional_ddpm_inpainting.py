import argparse
import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    device: torch.device
    image_size: int = 32
    channels: int = 1
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"
    dim: int = 48
    dim_mults: Tuple[int, ...] = (1, 2, 4)
    batch_size: int = 128
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    seed: int = 42
    num_workers: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_img(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def denormalize_img(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


class InpaintingNpyDataset(Dataset):
    def __init__(self, corrupted: np.ndarray, clean: np.ndarray):
        if corrupted.ndim == 3:
            corrupted = corrupted[..., None]
        if clean.ndim == 3:
            clean = clean[..., None]

        self.corrupted = corrupted.astype(np.float32)
        self.clean = clean.astype(np.float32)

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int):
        corrupted = self.corrupted[idx]
        clean = self.clean[idx]

        mask = (corrupted != 0).astype(np.float32)

        corrupted_t = torch.from_numpy(corrupted).permute(2, 0, 1)
        clean_t = torch.from_numpy(clean).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).permute(2, 0, 1)

        corrupted_t = normalize_img(corrupted_t)
        clean_t = normalize_img(clean_t)

        return corrupted_t, clean_t, mask_t


def create_dataloaders(
    clean_path: str,
    corrupted_path: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    limit: Optional[int] = None,
):
    clean = np.load(clean_path)
    corrupted = np.load(corrupted_path)

    if limit is not None:
        clean = clean[:limit]
        corrupted = corrupted[:limit]

    c_train, c_val, y_train, y_val = train_test_split(
        corrupted, clean, test_size=val_split, random_state=seed
    )

    train_ds = InpaintingNpyDataset(c_train, y_train)
    val_ds = InpaintingNpyDataset(c_val, y_val)

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, "b (h d) x y -> b h d (x y)", h=self.heads)
        k = rearrange(k, "b (h d) x y -> b h d (x y)", h=self.heads)
        v = rearrange(v, "b (h d) x y -> b h d (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h d (x y) -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.Conv2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        in_channels: int = 3,
        out_channels: int = 1,
        groups: int = 8,
    ):
        super().__init__()
        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=groups),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, groups=groups),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=groups)

        self.ups = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, groups=groups),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=groups),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_res = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim, groups=groups)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_noisy, cond], dim=1)
        x = self.init_conv(x)
        residual = x
        t_emb = self.time_mlp(t)

        skips = []
        for block1, block2, attn, down in self.downs:
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block1, block2, attn, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            x = up(x)

        x = torch.cat([x, residual], dim=1)
        x = self.final_res(x, t_emb)
        return self.final_conv(x)


class Diffusion:
    def __init__(self, config: Config):
        self.config = config
        if config.schedule == "cosine":
            betas = cosine_beta_schedule(config.timesteps)
        else:
            betas = linear_beta_schedule(config.timesteps, config.beta_start, config.beta_end)

        self.betas = betas.to(config.device)
        self.alphas = (1.0 - self.betas).to(config.device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(config.device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        xt = a * x0 + b * noise
        return xt, noise


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k], alpha=1.0 - decay)
            else:
                v.copy_(msd[k])


def ddpm_step(
    x_t: torch.Tensor,
    pred_noise: torch.Tensor,
    t: int,
    diffusion: Diffusion,
) -> torch.Tensor:
    beta_t = diffusion.betas[t]
    alpha_t = diffusion.alphas[t]
    alpha_bar_t = diffusion.alpha_bars[t]

    # mean = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * pred_noise)
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
    )

    if t > 0:
        sigma = torch.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        return mean + sigma * noise
    return mean


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[torch.device] = None,
) -> Dict[str, int]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    ema_model.load_state_dict(ckpt["ema_model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {"epoch": ckpt.get("epoch", 0), "step": ckpt.get("step", 0)}


def evaluate(
    model: nn.Module,
    diffusion: Diffusion,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for corrupted, clean, mask in loader:
            corrupted = corrupted.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            cond = torch.cat([corrupted, mask], dim=1)

            t = torch.randint(0, diffusion.config.timesteps, (clean.size(0),), device=device)
            x_t, noise = diffusion.q_sample(clean, t)
            with autocast(enabled=use_amp):
                pred_noise = model(x_t, t, cond)
                loss = F.mse_loss(pred_noise, noise)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def train(args: argparse.Namespace, config: Config) -> None:
    set_seed(config.seed)

    train_loader, val_loader = create_dataloaders(
        clean_path=args.clean_path,
        corrupted_path=args.corrupted_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=args.val_split,
        seed=config.seed,
        limit=args.limit,
    )

    model = ConditionalUNet(
        dim=config.dim,
        dim_mults=config.dim_mults,
        in_channels=3,
        out_channels=1,
    ).to(config.device)
    ema_model = copy.deepcopy(model).eval()
    diffusion = Diffusion(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    use_amp = config.device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        state = load_checkpoint(
            args.checkpoint_path,
            model,
            ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=config.device,
        )
        start_epoch = state["epoch"] + 1
        global_step = state["step"]

    for epoch in range(start_epoch, config.epochs):
        model.train()
        losses = []

        for corrupted, clean, mask in train_loader:
            corrupted = corrupted.to(config.device, non_blocking=True)
            clean = clean.to(config.device, non_blocking=True)
            mask = mask.to(config.device, non_blocking=True)

            cond = torch.cat([corrupted, mask], dim=1)
            t = torch.randint(0, config.timesteps, (clean.size(0),), device=config.device)
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

        train_loss = float(np.mean(losses)) if losses else float("inf")
        val_loss = evaluate(ema_model, diffusion, val_loader, config.device, use_amp)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{config.epochs} | train loss: {train_loss:.6f} | "
            f"val loss: {val_loss:.6f} | lr: {lr:.7f}"
        )

        save_checkpoint(
            args.checkpoint_path,
            model,
            ema_model,
            optimizer,
            scheduler,
            epoch,
            global_step,
        )

        if args.preview_every > 0 and ((epoch + 1) % args.preview_every == 0):
            sample(
                args=args,
                config=config,
                checkpoint_path=args.checkpoint_path,
                corrupted_input_path=args.preview_corrupted_path or args.corrupted_path,
                output_path=os.path.join(args.sample_dir, f"preview_epoch_{epoch + 1}.npy"),
                save_trajectory=False,
            )


def sample(
    args: argparse.Namespace,
    config: Config,
    checkpoint_path: Optional[str] = None,
    corrupted_input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    save_trajectory: Optional[bool] = None,
) -> None:
    checkpoint_path = checkpoint_path or args.checkpoint_path
    corrupted_input_path = corrupted_input_path or args.sample_corrupted_path
    output_path = output_path or args.sample_output
    if save_trajectory is None:
        save_trajectory = args.save_trajectory

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ConditionalUNet(
        dim=config.dim,
        dim_mults=config.dim_mults,
        in_channels=3,
        out_channels=1,
    ).to(config.device)
    ema_model = copy.deepcopy(model).eval()

    load_checkpoint(
        checkpoint_path,
        model,
        ema_model,
        optimizer=None,
        scheduler=None,
        map_location=config.device,
    )
    ema_model.eval()

    diffusion = Diffusion(config)

    corrupted_np = np.load(corrupted_input_path)
    if corrupted_np.ndim == 2:
        corrupted_np = corrupted_np[None, ..., None]
    elif corrupted_np.ndim == 3:
        if corrupted_np.shape[-1] == 1:
            corrupted_np = corrupted_np[None, ...]
        else:
            corrupted_np = corrupted_np[..., None]

    if args.sample_index >= len(corrupted_np):
        raise IndexError(f"sample_index {args.sample_index} out of range for {len(corrupted_np)} samples")

    corrupted = corrupted_np[args.sample_index].astype(np.float32)
    mask = (corrupted != 0).astype(np.float32)

    corrupted_t = torch.from_numpy(corrupted).permute(2, 0, 1).unsqueeze(0).to(config.device)
    mask_t = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(config.device)

    corrupted_t = normalize_img(corrupted_t)
    cond = torch.cat([corrupted_t, mask_t], dim=1)

    x = torch.randn((1, 1, config.image_size, config.image_size), device=config.device)

    trajectory_dir = None
    if save_trajectory:
        trajectory_dir = os.path.join(args.sample_dir, "trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(config.timesteps - 1, -1, -1):
            t = torch.full((1,), i, device=config.device, dtype=torch.long)
            pred_noise = ema_model(x, t, cond)
            x = ddpm_step(x, pred_noise, i, diffusion)

            # Inpainting projection: keep known pixels fixed at every reverse step.
            x = mask_t * corrupted_t + (1.0 - mask_t) * x

            if trajectory_dir is not None and (i % args.trajectory_stride == 0 or i == 0):
                frame = denormalize_img(x).squeeze().detach().cpu().numpy()
                np.save(os.path.join(trajectory_dir, f"t_{i:04d}.npy"), frame)

    final_img = denormalize_img(x).squeeze().detach().cpu().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, final_img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional DDPM for 32x32 grayscale inpainting")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--sample", action="store_true", help="Run sampling")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if available")

    parser.add_argument("--clean-path", type=str, default="cifar10_grayscale_32x32.npy")
    parser.add_argument("--corrupted-path", type=str, default="cifar10_grayscale_32x32_corrupted.npy")
    parser.add_argument("--sample-corrupted-path", type=str, default="cifar10_grayscale_32x32_corrupted.npy")

    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/ddpm_inpainting.pt")
    parser.add_argument("--sample-dir", type=str, default="sample_outputs")
    parser.add_argument("--sample-output", type=str, default="sample_outputs/final.npy")

    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--trajectory-stride", type=int, default=50)
    parser.add_argument("--preview-every", type=int, default=0)
    parser.add_argument("--preview-corrupted-path", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--dim-mults", type=str, default="1,2,4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dim_mults = tuple(int(x.strip()) for x in args.dim_mults.split(",") if x.strip())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(
        device=device,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule,
        dim=args.dim,
        dim_mults=dim_mults,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    if args.train:
        train(args, config)

    if args.sample:
        sample(args, config)

    if not args.train and not args.sample:
        raise ValueError("Nothing to do. Use --train and/or --sample.")


if __name__ == "__main__":
    main()
