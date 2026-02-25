import numpy as np
import torch
from torchvision import datasets, transforms

# -----------------------------
# Config
# -----------------------------
out_path = "cifar10_grayscale_32x32.npy"

# -----------------------------
# Transform: RGB -> Grayscale
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # (1, 32, 32)
    transforms.ToTensor(),                        # float32 in [0,1]
])

# -----------------------------
# Download CIFAR-10 (train + test)
# -----------------------------
train_ds = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Combine
full_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])

# -----------------------------
# Convert to numpy array (N, 32, 32)
# -----------------------------
N = len(full_ds)
data = np.zeros((N, 32, 32), dtype=np.float32)

for i in range(N):
    img, _ = full_ds[i]         # img: (1, 32, 32)
    data[i] = img.squeeze(0).numpy()

# -----------------------------
# Save
# -----------------------------
np.save(out_path, data)

print(f"Saved: {out_path}")
print("Shape:", data.shape)


import numpy as np

labels = np.load("cifar10_grayscale_32x32.npy")
labels.shape

import numpy as np

# ============================================
# Config
# ============================================
input_path  = "cifar10_grayscale_32x32.npy"
output_path = "cifar10_grayscale_32x32_corrupted.npy"

missing_fraction = 0.4     # fraction of pixels to corrupt (0–1)
mode = "random_mask"       # options: random_mask, block_mask, gaussian_noise
seed = 1337

rng = np.random.default_rng(seed)

# ============================================
# Load data
# ============================================
data = np.load(input_path)        # (N, 32, 32)
assert data.ndim == 3

N, H, W = data.shape
corrupted = data.copy()

# ============================================
# Corruption functions
# ============================================
def apply_random_mask(img, frac):
    mask = rng.random(img.shape) < frac
    out = img.copy()
    out[mask] = 0.0
    return out

def apply_block_mask(img, frac):
    out = img.copy()
    block_area = int(frac * H * W)

    # approximate square block
    side = int(np.sqrt(block_area))
    side = max(1, min(side, H))

    x0 = rng.integers(0, H - side + 1)
    y0 = rng.integers(0, W - side + 1)

    out[x0:x0+side, y0:y0+side] = 0.0
    return out

def apply_gaussian_noise(img, sigma=0.3):
    noise = rng.normal(0.0, sigma, size=img.shape)
    out = img + noise
    return np.clip(out, 0.0, 1.0)

# ============================================
# Apply corruption
# ============================================
for i in range(N):
    img = corrupted[i]

    if mode == "random_mask":
        corrupted[i] = apply_random_mask(img, missing_fraction)

    elif mode == "block_mask":
        corrupted[i] = apply_block_mask(img, missing_fraction)

    elif mode == "gaussian_noise":
        corrupted[i] = apply_gaussian_noise(img)

    else:
        raise ValueError("Unknown mode")

# ============================================
# Save
# ============================================
np.save(output_path, corrupted)

print("Saved:", output_path)
print("Shape:", corrupted.shape)


mask = (corrupted == 0).astype(np.float32)
np.save("cifar10_masks.npy", mask)
