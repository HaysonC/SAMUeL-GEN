from pathlib import Path
import numpy as np
import torch

class ZScoreNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Z-score normalization
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # Reverse Z-score normalization
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean
    


def load_encoded_songs(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    print("Loading encoded accompaniment and vocals from", root)
    root = Path(root)
    vocals = np.load(root / "latent_vocal.npy", mmap_mode='r')
    accomp = np.load(root / "latent_accomp.npy", mmap_mode='r')
    return torch.tensor(accomp, dtype=torch.float32), torch.tensor(vocals, dtype=torch.float32)


def prepare_data(root: str | Path):
    """
    Load and normalize latent accompaniment and vocals with tightened vocal normalization.

    Returns normalized tensors and their normalizers for inverse transforms.
    """

    accomp, vocals = load_encoded_songs(root)

    print(f"Accompaniment shape: {accomp.shape}, Vocals shape: {vocals.shape}")
    acc_mean = accomp.mean(dim=(0,2), keepdim=True)
    acc_std  = accomp.std(dim=(0,2), keepdim=True) + 1e-5
    voc_mean = vocals.mean(dim=(0,2), keepdim=True)
    voc_std  = vocals.std(dim=(0,2), keepdim=True) + 1e-5
    
    norm_acc = ZScoreNormalizer(acc_mean, acc_std)
    norm_voc = ZScoreNormalizer(voc_mean, voc_std)
    
    
    accomp_norm = norm_acc.normalize(accomp)
    vocals_norm = norm_voc.normalize(vocals)
    return accomp_norm, vocals_norm, norm_acc, norm_voc
