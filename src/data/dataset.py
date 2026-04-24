import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.audio_utils import mix_at_snr, load_audio, slice_audio, compute_stft, SEGMENT_SAMPLES, SAMPLE_RATE, SNR_MIN_DB, SNR_MAX_DB

class WaveformDataset(Dataset):
    """
    Dataset with deterministic augmentation for fair model comparison.
    
    Key features:
    - Fixed file splits (from MANIFEST)
    - Augmentation is deterministic given (epoch, index, augmentation_id)
    - Same augmented data across different model training runs
    - Different augmented data across epochs (for learning)
    """
    
    def __init__(self, clean_files, noise_files, mode='train',
                 snr_range=(SNR_MIN_DB, SNR_MAX_DB), 
                 augmentation_factor=1,
                 base_seed=42, 
                 epoch=0,
                 segment_length=SEGMENT_SAMPLES, 
                 sample_rate=SAMPLE_RATE):
        """
        Args:
            clean_files: List of clean audio file paths
            noise_files: List of noise file paths
            mode: 'train', 'val', or 'test'
            snr_range: (min_snr, max_snr) in dB
            augmentation_factor: Number of augmented versions per clean segment
            base_seed: Base random seed for reproducibility
            epoch: Current training epoch (changes augmentation)
        """
        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val', or 'test'"
        self.mode = mode
        self.snr_range = snr_range
        self.segment_length = segment_length,
        self.sample_rate = sample_rate

        self.base_seed = base_seed
        self.epoch = epoch
        self.augmentation_factor = augmentation_factor if mode == 'train' else 1
        
        # Load clean and noise segments (same as before)
        self.clean_segments = []
        self.noise_segments = []

        # Load clean speech
        print(f"[{mode.upper()}] Loading {len(clean_files)} clean files...")
        self.clean_segments = []
        for f in clean_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.clean_segments.extend(slice_audio(audio, segment_length))
        
        # Load noise
        print(f"[{mode.upper()}] Loading {len(noise_files)} noise files...")
        self.noise_segments = []
        for f in noise_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.noise_segments.extend(slice_audio(audio, segment_length))
        
        print(f"[{mode.upper()}] {len(self.clean_segments)} clean, {len(self.noise_segments)} noise segments")
        
        # For val/test, generate fixed pairs once
        if mode in ['val', 'test']:
            self.fixed_pairs = self._generate_fixed_pairs()
    
    def set_epoch(self, epoch):
        """Update epoch for deterministic augmentation."""
        self.epoch = epoch
    
    def _generate_fixed_pairs(self):
        pairs = []
        for clean_seg in self.clean_segments:
            noise_seg = random.choice(self.noise_segments)
            snr = random.uniform(*self.snr_range)
            noisy_seg = mix_at_snr(clean_seg, noise_seg, snr)
            pairs.append((clean_seg, noisy_seg, snr))
        return pairs
    
    def _get_deterministic_random(self, index):
        """
        Get a deterministic random state for this (epoch, index).
        
        This ensures:
        - Same index at same epoch = same augmentation (reproducible)
        - Same index at different epochs = different augmentation (variety)
        - Same across different model runs (fair comparison)
        """
        # Map augmented index to base clean index
        clean_idx = index % len(self.clean_segments)
        aug_id = index // len(self.clean_segments)  # Which augmentation (0 to augmentation_factor-1)
        
        # Create deterministic seed from: base_seed + epoch + clean_idx + aug_id
        seed = self.base_seed + (self.epoch * 1000000) + (clean_idx * 100) + aug_id
        
        # Create local random state (doesn't affect global random)
        rng = np.random.RandomState(seed)
        
        return rng, clean_idx
    
    def __len__(self):
        return len(self.clean_segments) * self.augmentation_factor
    
    def __getitem__(self, index):
        if self.mode in ['val', 'test']:
            # Fixed pairs (deterministic)
            clean_idx = index % len(self.clean_segments)
            clean, noisy, snr = self.fixed_pairs[clean_idx]
        else:
            # Deterministic augmentation for training
            rng, clean_idx = self._get_deterministic_random(index)
            
            clean = self.clean_segments[clean_idx]
            
            # Use local RNG for deterministic randomness
            noise_idx = rng.randint(0, len(self.noise_segments))
            noise = self.noise_segments[noise_idx]
            
            snr = rng.uniform(*self.snr_range)
            noisy = mix_at_snr(clean, noise, snr)
        
        # Compute complex STFT
        clean_mag, _ = compute_stft(clean)
        noisy_mag, noisy_phase = compute_stft(noisy)
        
        noisy_tensor = torch.tensor(noisy_mag, dtype=torch.float32).unsqueeze(0)
        clean_tensor = torch.tensor(clean_mag, dtype=torch.float32).unsqueeze(0)
        phase_tensor = torch.tensor(noisy_phase, dtype=torch.float32)
        
        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'phase': phase_tensor,
            'snr': snr
        }

print("✓ Dataset class defined")