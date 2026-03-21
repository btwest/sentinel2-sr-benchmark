"""
Fast SEN2NAIP dataset that reads pre-extracted .npz files.
Run data/extract_sen2naip.py first to generate the files.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SEN2NAIPNpz(Dataset):
    def __init__(self, root: str | Path, phase: str = "train", val_fraction: float = 0.1):
        root = Path(root)
        all_files = sorted(root.glob("*.npz"))
        if not all_files:
            raise FileNotFoundError(f"No .npz files found in {root}. Run data/extract_sen2naip.py first.")

        split = int(len(all_files) * (1.0 - val_fraction))
        self.files = all_files[:split] if phase == "train" else all_files[split:]
        print(f"SEN2NAIPNpz {phase}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        lr = torch.from_numpy(data["lr"].astype(np.float32)) / 10000.0  # (4,130,130) [0,1]
        hr = torch.from_numpy(data["hr"].astype(np.float32)) / 10000.0  # (4,520,520) [0,1]
        return lr, hr
