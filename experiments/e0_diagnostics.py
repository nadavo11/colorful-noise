"""E0: How non-white is the Colorful-Noise latent, really?

Measures radially-averaged PSD + DC offsets of:
  (a) white noise
  (b) VAE latents of the paper's input images
  (c) fft_radial_frequency_swap(noise, latent, alpha, gamma) across a grid

Outputs plots + a JSON summary to results/e0/.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/colorful_noise")
from utils import fft_radial_frequency_swap
from spectral_ops import radial_psd, whiteness

OUT = "/workspace/experiments/results/e0"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

# --- encode inputs with the SDXL VAE (fp32) --------------------------------
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae").to("cuda")

def encode(path, size=1024):
    img = Image.open(path).convert("RGB").resize((size, size))
    import numpy as np
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)[None] / 127.5 - 1.0
    with torch.no_grad():
        z = vae.encode(x.to("cuda")).latent_dist.sample() * vae.config.scaling_factor
    return z

inputs = sorted(glob.glob("/workspace/colorful_noise/inputs/*.png"))
latents = {os.path.basename(p): encode(p) for p in inputs}
H = W = 128

# --- (a)+(b): noise vs natural-image latents --------------------------------
noise = torch.randn(8, 4, H, W, device="cuda")
c_n, psd_n = radial_psd(noise)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].loglog(c_n[1:], psd_n.mean(0)[1:], "k-", lw=2, label="white noise")
for name, z in latents.items():
    c, psd = radial_psd(z)
    axes[0].loglog(c[1:], psd.mean(0)[1:], label=f"VAE latent: {name}")
axes[0].set(title="White noise vs VAE latents (channel-avg PSD)",
            xlabel="radial frequency", ylabel="power")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# --- (c): the paper's mixed latent across (alpha, gamma) --------------------
z = latents["cat_orange.png"]
summary = {"whiteness_white_noise": whiteness(noise),
           "dc_per_channel": {}}
alphas = [0.005, 0.015, 0.05, 0.15]
gammas = [0.05, 0.07, 0.2, 1.0]
n1 = torch.randn(1, 4, H, W, device="cuda")
for a in alphas:
    for g in ([0.05] if a != 0.015 else gammas):  # full gamma sweep at default alpha
        mix = fft_radial_frequency_swap(n1, z, p=a, temp=g)
        c, psd = radial_psd(mix)
        axes[1].loglog(c[1:], psd.mean(0)[1:], label=f"α={a}, γ={g}")
        summary[f"whiteness_alpha{a}_gamma{g}"] = whiteness(mix)
        summary[f"std_alpha{a}_gamma{g}"] = float(mix.std())
axes[1].loglog(c_n[1:], psd_n.mean(0)[1:], "k--", lw=2, label="white noise")
axes[1].set(title="Paper's mixed latent PSD (cat_orange)",
            xlabel="radial frequency", ylabel="power")
axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/psd_overview.png", dpi=130)

# --- DC / channel-mean analysis ---------------------------------------------
for name, z in latents.items():
    summary["dc_per_channel"][name] = {
        "latent_channel_means": z.mean(dim=(2, 3)).flatten().tolist(),
        "mix_channel_means_default": fft_radial_frequency_swap(
            n1, z, p=0.015, temp=0.05).mean(dim=(2, 3)).flatten().tolist(),
    }
summary["noise_channel_means"] = n1.mean(dim=(2, 3)).flatten().tolist()
summary["latent_std"] = {n: float(z.std()) for n, z in latents.items()}

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"saved -> {OUT}/psd_overview.png")
