# ddpm-inpainting
conditional-ddpm-inpainting

# CFG-DDPM Inpainting and Reconstruction

Classifier-free conditional diffusion for image reconstruction and inpainting using PyTorch.

This repository implements a conditional denoising diffusion probabilistic model (DDPM) with classifier-free guidance (CFG) for recovering missing or corrupted image regions and reconstructing dense images from sparse or degraded inputs.

The core idea is to learn the conditional score function

    ∇x log p(x | c)

where:
- x = target (ground truth image)
- c = conditioning input (corrupted, sparse, masked, or partial image)

The model supports:

- Conditional reconstruction (dense from sparse)
- Image inpainting (mask-based missing regions)
- Classifier-free guidance sampling
- Paired numpy datasets
- U-Net with cross-attention conditioning
- Full training and sampling pipelines

---

## Conceptual Overview

### Diffusion Models

A DDPM learns to reverse a Markov noising process.

Forward process:

    q(x_t | x_0) = sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε

where ε ~ N(0, I).

The neural network learns to predict ε:

    ε_θ(x_t, t, c)

Training objective:

    L = || ε − ε_θ(x_t, t, c) ||²

After training, generation proceeds by iteratively denoising from Gaussian noise.

---

## Conditional Diffusion

Instead of modeling p(x), we model:

    p(x | c)

where c is an observed degraded version of the image.

Examples of conditioning:

- Sparse pixels
- Masked images
- Low-resolution inputs
- Corrupted measurements
- Physics-based projections

The network receives both:

- noisy sample x_t
- condition c

and predicts the noise consistent with the condition.

---

## Classifier-Free Guidance (CFG)

Classifier-free guidance removes the need for an external classifier.

Training:

With probability p, the condition is dropped (set to zero):

    ε_θ(x_t, t, ∅)

Otherwise:

    ε_θ(x_t, t, c)

Sampling combines both predictions:

    ε = ε_uncond + w (ε_cond − ε_uncond)

where w is the guidance strength.

Interpretation:

- w = 0 → unconditional generation
- w = 1 → standard conditional
- w > 1 → stronger adherence to condition

CFG increases fidelity and constraint consistency.

---

## Inpainting Mechanism

Inpainting is enforced using projection during reverse diffusion.

Known pixels are re-noised to timestep t:

    x_known_t = q(x_known, t)

Then merged:

    x_t ← mask * x_known_t + (1 − mask) * x_t

This ensures:

- Known regions remain consistent
- Unknown regions are generated
- Global coherence emerges naturally

This is equivalent to solving a constrained stochastic optimization problem.

---

## Architecture

The model uses a conditional U-Net with:

- Time embedding (sinusoidal)
- Residual blocks
- Linear attention
- Cross-attention conditioning
- Control encoder for conditioning image

The conditioning path encodes spatial structure and injects features at multiple resolutions.

---

## Repository Structure
