import os
import argparse
import copy

import numpy as np
import torch

from conditional_ddpm_module import (
    Config,
    ConditionalUNet,
    Diffusion,
    normalize_img,
    denormalize_img,
    ddpm_step,
)


# =========================================================
# Inference
# =========================================================

def run_inference(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        device=device,
        timesteps=args.timesteps,
        schedule=args.schedule,
        dim=args.dim,
        dim_mults=tuple(int(x) for x in args.dim_mults.split(",")),
    )

    # Build model
    model = ConditionalUNet(
        dim=config.dim,
        dim_mults=config.dim_mults,
        in_channels=3,
        out_channels=1,
    ).to(device)

    ema_model = copy.deepcopy(model).eval()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    ema_model.load_state_dict(ckpt["ema_model"])
    ema_model.eval()

    diffusion = Diffusion(config)

    # =====================================================
    # Load input
    # =====================================================

    corrupted_np = np.load(args.input)

    if corrupted_np.ndim == 2:
        corrupted_np = corrupted_np[None, ..., None]
    elif corrupted_np.ndim == 3:
        corrupted_np = corrupted_np[..., None]

    corrupted = corrupted_np[args.index].astype(np.float32)
    mask = (corrupted != 0).astype(np.float32)

    corrupted_t = torch.from_numpy(corrupted).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(device)

    corrupted_t = normalize_img(corrupted_t)

    cond = torch.cat([corrupted_t, mask_t], dim=1)

    # =====================================================
    # Reverse diffusion
    # =====================================================

    x = torch.randn((1, 1, config.image_size, config.image_size), device=device)

    with torch.no_grad():
        for i in range(config.timesteps - 1, -1, -1):

            t = torch.full((1,), i, device=device, dtype=torch.long)

            pred_noise = ema_model(x, t, cond)

            x = ddpm_step(x, pred_noise, i, diffusion)

            # Inpainting projection
            x = mask_t * corrupted_t + (1.0 - mask_t) * x

    final = denormalize_img(x).squeeze().cpu().numpy()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, final)

    print("Saved:", args.output)


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="sample_outputs/final.npy")
    p.add_argument("--index", type=int, default=0)

    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", default="cosine")
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--dim-mults", default="1,2,4")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
    
    
    
    
'''
Example run:

python inference_ddpm_inpaint.py \
  --checkpoint checkpoints/ddpm_inpainting.pt \
  --input cifar10_grayscale_32x32_corrupted.npy \
  --output sample_outputs/final.npy
  
'''
    
    
    
    
    
