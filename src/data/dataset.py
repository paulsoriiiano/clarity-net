import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from audio_utils import mix_at_snr, load_audio, slice_audio

class WaveformDataset(Dataset):
    def __init__(self, manifest_path, split, root_dir="."):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.records = manifest["splits"][split]
        self.root_dir = Path(root_dir)
        self.sample_rate = manifest["audio_config"]["sample_rate"]
        self.segment_length = manifest["audio_config"]["segment_samples"]

        # simple file cache so each file is only loaded/sliced once
        self._segment_cache = {}

    def __len__(self):
        return len(self.records)

    def _get_segments_for_file(self, rel_path):
        if rel_path not in self._segment_cache:
            full_path = self.root_dir / rel_path
            audio = load_audio(str(full_path), sr=self.sample_rate)
            self._segment_cache[rel_path] = slice_audio(audio, self.segment_length)
        return self._segment_cache[rel_path]

    def __getitem__(self, idx):
        row = self.records[idx]

        clean = self._get_segments_for_file(row["clean_path"])[row["clean_segment_idx"]]
        noise = self._get_segments_for_file(row["noise_path"])[row["noise_segment_idx"]]
        noisy = mix_at_snr(clean, noise, row["snr_db"])

        return {
            "pair_id": row["pair_id"],
            "clean_audio": torch.tensor(clean, dtype=torch.float32),
            "noisy_audio": torch.tensor(noisy, dtype=torch.float32),
            "snr": row["snr_db"],
        }