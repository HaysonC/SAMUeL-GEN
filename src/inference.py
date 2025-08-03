import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model import DiffusionModel
from diffusion import Diffusion
from sample import p_sample_v
import matplotlib.pyplot as plt
from data_utils import prepare_data
from utils import save_sampling_animation, decode
from config import TRAINING_TIMESTEPS as TIMESTEPS

import os
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def get_device():
    device = input("Enter device (cuda, mps, cpu): ").strip()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available. Falling back to CPU.")
        device = "cpu"
    elif device not in ["cuda", "mps", "cpu"]:
        print("Invalid device choice. Defaulting to automatic device selection.")
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
    return device

def main():
    device = get_device()
    timesteps = TIMESTEPS
    diffusion = Diffusion(timesteps=timesteps).to(device)
    model_path = input("Enter model checkpoint path: ").strip() or "checkpoints/model2_best.pth"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Exiting.")
        return
    model = DiffusionModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model_state'], strict=False)
    print(f"Model loaded from {model_path} with device {device}")
    model.eval()
    output_dir = Path("./outputs/generated_vocals")
    os.makedirs(output_dir, exist_ok=True)
    path = input("Enter dataset path (or leave blank for kagglehub): ").strip()
    if not path:
        from kagglehub import dataset_download
        path = dataset_download("boyazhangnb/encodedsongs")
    accomp, vocals, norm_acc, norm_voc = prepare_data(path)
    idx = 0
    try:
        idx = int(input(f"Enter index of song to generate (0 to {len(vocals) - 1}): ").strip())
        if idx < 0 or idx >= len(vocals):
            raise ValueError("Index out of range.")
    except ValueError:
        print("Invalid input. Defaulting to index 0.")
        idx = 0
    vocals_sample = vocals[idx].unsqueeze(0).to(device)
    cond = vocals_sample

    noise = torch.randn_like(vocals_sample)
    print(f"Sampling with shape: {vocals_sample.shape}")
    # run sampling and record frames and statistics
    generated, frames, stats_means, stats_stds = p_sample_v(  # type: ignore
        model, diffusion, noise, cond,
        guidance_scale=3.0,
        return_frames=True,
        return_stats=True,
        debug=True,
        clamp_range=(accomp.min().item(), accomp.max().item())
    )
    print(f"Generated tensor stats before denorm: mean {generated.mean().item():.4f}, std {generated.std().item():.4f}, min {generated.min().item():.4f}, max {generated.max().item():.4f}")
    generated_accomp = norm_acc.denormalize(generated)
    # Free memory after sampling
    del noise
    del accomp
    del vocals
    import gc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    # Optionally show or save spectrogram/animation
    try:
        from latent_dit.utils import show_spectral
        show_spectral(generated_accomp, title="Generated Accompaniment Spectrogram")
    except ImportError:
        pass
    try:
        decoded_accomp = decode(generated_accomp, savePath=output_dir / "inference_output_accomp.wav", device=device)
    except ImportError as e:
        decoded_accomp = None
        print("WARNING: decode function not available. Skipping decoding.")
        print(f"Error: {e}")
    if frames is not None and input("Save sampling animation? 7(y/N): ").strip().lower() == 'y':
        print("Saving sampling animation...")
        save_sampling_animation(frames, save_path=output_dir / "sampling_animation.mp4", interval=50, cmap="magma")
    # plot and save sampling statistics
    stats_path = output_dir / 'p_sample_v_stats.png'
    plt.figure()
    plt.plot(stats_means, label='x mean')
    plt.plot(stats_stds, label='x std')
    plt.xlabel('Sampling step')
    plt.ylabel('Value')
    plt.title('p_sample_v x Statistics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(stats_path)
    plt.close()
    print(f"Saved sampling statistics plot to {stats_path}")
    print("Decoded generated accompaniment saved as 'inference_output_accomp.wav'.")

if __name__ == "__main__":
    main()
