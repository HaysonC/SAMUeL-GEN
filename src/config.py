"""
# model2/config.py

Configuration settings for the UNet model. Including shapes for each stage, conditioning parameters, and attention strategies.

As well as the v-objective for training, and all training parameters.
"""

from typing import List, Tuple, Optional, Dict, Any, Union, Callable, Type, TypeVar
from dataclasses import dataclass, field

import torch
"""
Data Path
"""
DATA_KAGGLE_PATH: str = "boyazhangnb/encodedsongs"

"""
Configuration settings for the UNet model.


| Stage | Shape         | Notes                                 |
| ----- | ------------- | ------------------------------------- |
| Input | (B, 64, 1024) | Feature = 64, Temporal = 1024         |
| ↓     | (B, 128, 512) | Conv1D with stride 2                  |
| ↓     | (B, 256, 256) | ...                                   |
| ↓     | (B, 512, 128) | Bottleneck (add attention)            |
| ↑     | (B, 256, 256) | Upsample + concat skip connection     |
| ↑     | (B, 128, 512) | ...                                   |
| ↑     | (B, 64, 1024) | Final v prediction                    |
"""

MODEL_UNET_SHAPES: List[Tuple[int, int]] = [
    (64, 1024),  # Input
    (128, 512),  # Downsample
    (256, 256),  # Downsample
    (512, 128),  # Bottleneck
    (256, 256),  # Upsample
    (128, 512),  # Upsample
    (64, 1024),  # Output
]

MODEL_WINDOW_SIZE: int = 16 # Window size for local attention
# Training hyperparameters
TRAINING_TIMESTEPS: int = 800
TRAINING_LR: float = 3.5e-4
TRAINING_EPOCHS: int = 100

def get_dynamic_batch_size(target_gpu_mem_gb: float = 10.0, base_batch_size: int = 32, base_mem_gb: float = 8.0) -> int:
    """
    Dynamically estimate batch size based on available GPU memory.
    Assumes base_batch_size fits into base_mem_gb GPU RAM.
    """
    if torch.cuda.is_available():
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    elif torch.backends.mps.is_available():
        # MPS does not expose memory info, so fallback to base
        total_mem_gb = base_mem_gb
    else:
        # For CPU, use base batch size
        total_mem_gb = base_mem_gb

    # Scale batch size linearly with available memory
    batch_size = int(base_batch_size * (total_mem_gb / base_mem_gb))
    return max(1, batch_size)

TRAINING_BATCH_SIZE: int = 32 # keep batch size smaller to ensure batch diversity and compliance with SNR-weighted loss
print(f"Dynamic batch size set to: {TRAINING_BATCH_SIZE}")
TRAINING_COND_DROP_PROB: float = 0.1
TRAINING_LATENT_NOISE_LEVEL: float = 0.05
TRAINING_PATIENCE: int = 30
TRAINING_COND_DROP_PROB: float = 0.1  
TRAINING_CFG_SCALE: float = 2.0 
TRAINING_EMA_DECAY: float = 0.995
LOSS_WEIGHT_MODE: str = 'inverse' # 'raw', 'normalized', 'log', 'inverse'

SYSTEM_DEVICE: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

