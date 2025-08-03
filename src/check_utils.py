import numpy as np
import librosa
import torch

def check_wav_similarity(path1, path2, threshold=0.99):
    """
    Check if two wav files are similar or the same using normalized cross-correlation of their audio features.
    Returns True if similarity is above threshold, else False.
    """
    y1, sr1 = librosa.load(path1, sr=None)
    y2, sr2 = librosa.load(path2, sr=None)
    # Resample if needed
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1
    # Pad or trim to same length
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    # Compute normalized cross-correlation
    similarity = np.corrcoef(y1, y2)[0, 1]
    return similarity >= threshold, similarity


path1 = "outputs/generated_vocals/inference_output_accomp_e.wav"
path2 = "outputs/generated_vocals/inference_output_accomp.wav"
import torch.nn as nn

# Model checkpoint utilities

def load_model_from_checkpoint(checkpoint_path, model_class, map_location='cpu'):
    """
    Load a model from a checkpoint file.
    Args:
        checkpoint_path (str): Path to the checkpoint .pth file.
        model_class (type): The class of the model to instantiate.
        map_location (str or torch.device): Device mapping for loading.
    Returns:
        model (nn.Module): Model loaded with weights from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Support both nested and flat checkpoints
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model = model_class()
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow missing keys
    return model
def module_structure_diff(model1, model2):
    mods1 = dict(model1.named_modules())
    mods2 = dict(model2.named_modules())

    all_keys = set(mods1.keys()).union(mods2.keys())
    diffs = []

    for key in all_keys:
        m1 = mods1.get(key)
        m2 = mods2.get(key)

        if m1 is None:
            diffs.append((key, "🟢 New in model2: " + str(type(m2).__name__)))
        elif m2 is None:
            diffs.append((key, "🔴 Missing in model2: " + str(type(m1).__name__)))
        elif type(m1) != type(m2):
            diffs.append((key, f"🟡 Type mismatch: {type(m1).__name__} vs {type(m2).__name__}"))
        # else: identical
    return diffs

if __name__ == "__main__":
    print(check_wav_similarity(path1, path2))
