import random
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from src.data.audio_utils import (
    mix_at_snr,
    load_audio,
    slice_audio,
    compute_stft,
    SEGMENT_SAMPLES,
    SAMPLE_RATE,
    SNR_MIN_DB,
    SNR_MAX_DB,
    N_FFT,
    HOP_LENGTH
)

def compute_complex_stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Compute complex STFT (returns complex values, not separated).
    
    Args:
        audio: Audio signal
        n_fft: FFT size
        hop_length: Hop length
    
    Returns:
        Complex STFT array
    """
    stft_complex = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return stft_complex


class WaveformDataset(Dataset):
    """
    Spectrogram dataset for baseline CNN-style models.

    Train mode:
        - deterministic augmentation across epochs via _get_deterministic_random
        - length = num_clean_segments * augmentation_factor

    Val/Test mode:
        - fixed deterministic noisy-clean pairs
        - can generate multiple fixed noisy versions per clean segment via eval_pairs_per_clean
        - length = len(self.fixed_pairs)

    Returns:
        {
            "noisy": [1, F, T] magnitude spectrogram,
            "clean": [1, F, T] magnitude spectrogram,
            "phase": [F, T] noisy phase,
            "snr": float,
            "clean_waveform": [samples],
            "noisy_waveform": [samples],
        }
    """

    def __init__(
        self,
        clean_files,
        noise_files,
        mode="train",
        snr_range=(SNR_MIN_DB, SNR_MAX_DB),
        augmentation_factor=1,
        eval_pairs_per_clean=1,
        base_seed=42,
        epoch=0,
        segment_length=SEGMENT_SAMPLES,
        sample_rate=SAMPLE_RATE,
    ):
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val', or 'test'"

        self.mode = mode
        self.snr_range = snr_range
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.base_seed = base_seed
        self.epoch = epoch

        self.augmentation_factor = augmentation_factor if mode == "train" else 1
        self.eval_pairs_per_clean = eval_pairs_per_clean if mode in ["val", "test"] else 1

        # Load clean segments
        print(f"[{mode.upper()}] Loading {len(clean_files)} clean files...")
        self.clean_segments = []
        for f in clean_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.clean_segments.extend(slice_audio(audio, segment_length))

        # Load noise segments
        print(f"[{mode.upper()}] Loading {len(noise_files)} noise files...")
        self.noise_segments = []
        for f in noise_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.noise_segments.extend(slice_audio(audio, segment_length))

        print(
            f"[{mode.upper()}] {len(self.clean_segments)} clean segments, "
            f"{len(self.noise_segments)} noise segments"
        )

        if mode in ["val", "test"]:
            print(
                f"[{mode.upper()}] Generating deterministic fixed pairs "
                f"(eval_pairs_per_clean={self.eval_pairs_per_clean})..."
            )
            self.fixed_pairs = self._generate_fixed_pairs()
            print(f"[{mode.upper()}] Total fixed pairs: {len(self.fixed_pairs)}")

    def set_epoch(self, epoch):
        """Update epoch for deterministic train-time augmentation."""
        self.epoch = epoch

    def _generate_fixed_pairs(self):
        """
        Generate deterministic fixed pairs for val/test.

        Each clean segment gets `eval_pairs_per_clean` fixed noisy versions.
        """
        rng = random.Random(self.base_seed)
        pairs = []

        for clean_seg in self.clean_segments:
            for _ in range(self.eval_pairs_per_clean):
                noise_seg = rng.choice(self.noise_segments)
                snr = rng.uniform(*self.snr_range)
                noisy_seg = mix_at_snr(clean_seg, noise_seg, snr)
                pairs.append((clean_seg, noisy_seg, snr))

        return pairs

    def _get_deterministic_random(self, index):
        """
        Deterministic RNG for train mode.

        Ensures:
        - same (epoch, index) -> same augmentation
        - different epochs -> different augmentation
        - reproducible across model runs
        """
        clean_idx = index % len(self.clean_segments)
        aug_id = index // len(self.clean_segments)

        seed = self.base_seed + (self.epoch * 1000000) + (clean_idx * 100) + aug_id
        rng = np.random.RandomState(seed)

        return rng, clean_idx

    def __len__(self):
        if self.mode in ["val", "test"]:
            return len(self.fixed_pairs)
        return len(self.clean_segments) * self.augmentation_factor

    def __getitem__(self, index):
        if self.mode in ["val", "test"]:
            clean, noisy, snr = self.fixed_pairs[index]
        else:
            rng, clean_idx = self._get_deterministic_random(index)

            clean = self.clean_segments[clean_idx]
            noise_idx = rng.randint(0, len(self.noise_segments))
            noise = self.noise_segments[noise_idx]
            snr = rng.uniform(*self.snr_range)
            noisy = mix_at_snr(clean, noise, snr)

        # Spectrogram representation
        clean_mag, _ = compute_stft(clean)
        noisy_mag, noisy_phase = compute_stft(noisy)

        noisy_tensor = torch.tensor(noisy_mag, dtype=torch.float32).unsqueeze(0)
        clean_tensor = torch.tensor(clean_mag, dtype=torch.float32).unsqueeze(0)
        phase_tensor = torch.tensor(noisy_phase, dtype=torch.float32)

        clean_waveform = torch.tensor(clean, dtype=torch.float32)
        noisy_waveform = torch.tensor(noisy, dtype=torch.float32)

        return {
            "noisy": noisy_tensor,
            "clean": clean_tensor,
            "phase": phase_tensor,
            "snr": float(snr),
            "clean_waveform": clean_waveform,
            "noisy_waveform": noisy_waveform,
        }


class FullSubNetDataset(Dataset):
    """
    Waveform dataset for FullSubNet.

    Train mode:
        - deterministic augmentation across epochs via _get_deterministic_random
        - optional SNR curriculum scheduling
        - length = num_clean_segments * augmentation_factor

    Val/Test mode:
        - fixed deterministic noisy-clean pairs
        - can generate multiple fixed noisy versions per clean segment via
          eval_pairs_per_clean
        - length = len(self.fixed_pairs)

    Returns:
        {
            "noisy": [samples],
            "clean": [samples],
            "noisy_waveform": [samples],
            "clean_waveform": [samples],
            "snr": float,
        }
    """

    def __init__(
        self,
        clean_files,
        noise_files,
        mode="train",
        snr_range=(SNR_MIN_DB, SNR_MAX_DB),
        augmentation_factor=1,
        eval_pairs_per_clean=1,
        base_seed=42,
        epoch=0,
        segment_length=SEGMENT_SAMPLES,
        sample_rate=SAMPLE_RATE,
        use_snr_curriculum=False,
        snr_curriculum=None,
    ):
        """
        Args:
            clean_files: List of clean audio file paths
            noise_files: List of noise audio file paths
            mode: 'train', 'val', or 'test'
            snr_range: Default SNR range for mixing
            augmentation_factor: Number of train augmentations per clean segment
            eval_pairs_per_clean: Number of fixed noisy versions per clean segment in val/test
            base_seed: Base seed for reproducibility
            epoch: Current epoch (used only in train mode)
            segment_length: Segment length in samples
            sample_rate: Sample rate
            use_snr_curriculum: Whether to update train SNR range by epoch
            snr_curriculum: List of tuples:
                [(start_epoch, end_epoch, (snr_min, snr_max)), ...]
        """
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val', or 'test'"

        self.mode = mode
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.base_seed = base_seed
        self.epoch = epoch

        self.default_snr_range = snr_range
        self.snr_range = snr_range

        self.use_snr_curriculum = use_snr_curriculum
        self.snr_curriculum = snr_curriculum

        self.augmentation_factor = augmentation_factor if mode == "train" else 1
        self.eval_pairs_per_clean = eval_pairs_per_clean if mode in ["val", "test"] else 1

        # Initialize epoch-dependent train SNR range if curriculum is enabled
        if self.mode == "train":
            self.snr_range = self._get_snr_range_for_epoch(self.epoch)

        # Load clean speech segments
        print(f"[FullSubNet-{mode.upper()}] Loading {len(clean_files)} clean files...")
        self.clean_segments = []
        for f in clean_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.clean_segments.extend(slice_audio(audio, segment_length))

        # Load noise segments
        print(f"[FullSubNet-{mode.upper()}] Loading {len(noise_files)} noise files...")
        self.noise_segments = []
        for f in noise_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.noise_segments.extend(slice_audio(audio, segment_length))

        print(
            f"[FullSubNet-{mode.upper()}] {len(self.clean_segments)} clean segments, "
            f"{len(self.noise_segments)} noise segments"
        )

        if self.mode == "train":
            print(f"[FullSubNet-TRAIN] Initial train SNR range: {self.snr_range}")

        # For val/test, generate deterministic fixed pairs once
        if mode in ["val", "test"]:
            print(
                f"[FullSubNet-{mode.upper()}] Generating deterministic fixed pairs "
                f"(eval_pairs_per_clean={self.eval_pairs_per_clean})..."
            )
            self.fixed_pairs = self._generate_fixed_pairs()
            print(f"[FullSubNet-{mode.upper()}] Total fixed pairs: {len(self.fixed_pairs)}")

    def _get_snr_range_for_epoch(self, epoch):
        """
        Return the train-time SNR range for the given epoch.

        Curriculum format:
            [(start_epoch, end_epoch, (snr_min, snr_max)), ...]
        """
        if not self.use_snr_curriculum or self.snr_curriculum is None:
            return self.default_snr_range

        for start_epoch, end_epoch, snr_range in self.snr_curriculum:
            if start_epoch <= epoch <= end_epoch:
                return snr_range

        return self.default_snr_range

    def set_epoch(self, epoch):
        """
        Update epoch and refresh train-time SNR range if curriculum is enabled.
        """
        self.epoch = epoch
        if self.mode == "train":
            self.snr_range = self._get_snr_range_for_epoch(epoch)

    def _generate_fixed_pairs(self):
        """
        Generate deterministic fixed pairs for val/test.

        Each clean segment gets `eval_pairs_per_clean` fixed noisy versions.
        """
        rng = random.Random(self.base_seed)
        pairs = []

        for clean_seg in self.clean_segments:
            for _ in range(self.eval_pairs_per_clean):
                noise_seg = rng.choice(self.noise_segments)
                snr = rng.uniform(*self.default_snr_range)
                noisy_seg = mix_at_snr(clean_seg, noise_seg, snr)
                pairs.append((clean_seg, noisy_seg, snr))

        return pairs

    def _get_deterministic_random(self, index):
        """
        Deterministic RNG for train mode.

        Ensures:
        - same (epoch, index) -> same augmentation
        - different epochs -> different augmentation
        - reproducible across runs
        """
        clean_idx = index % len(self.clean_segments)
        aug_id = index // len(self.clean_segments)

        seed = self.base_seed + (self.epoch * 1000000) + (clean_idx * 100) + aug_id
        rng = np.random.RandomState(seed)
        return rng, clean_idx

    def __len__(self):
        if self.mode in ["val", "test"]:
            return len(self.fixed_pairs)
        return len(self.clean_segments) * self.augmentation_factor

    def __getitem__(self, index):
        if self.mode in ["val", "test"]:
            clean, noisy, snr = self.fixed_pairs[index]
        else:
            rng, clean_idx = self._get_deterministic_random(index)

            clean = self.clean_segments[clean_idx]
            noise_idx = rng.randint(0, len(self.noise_segments))
            noise = self.noise_segments[noise_idx]

            snr = rng.uniform(*self.snr_range)
            noisy = mix_at_snr(clean, noise, snr)

        clean_tensor = torch.tensor(clean, dtype=torch.float32)
        noisy_tensor = torch.tensor(noisy, dtype=torch.float32)

        return {
            "noisy": noisy_tensor,
            "clean": clean_tensor,
            "noisy_waveform": noisy_tensor,
            "clean_waveform": clean_tensor,
            "snr": float(snr),
        }


class FullSubNetPlusDataset(Dataset):
    """
    Dataset for FullSubNet+ style models using magnitude + real + imaginary
    spectrogram information.

    Train mode:
        - deterministic augmentation across epochs via _get_deterministic_random
        - length = num_clean_segments * augmentation_factor

    Val/Test mode:
        - fixed deterministic noisy-clean pairs
        - can generate multiple fixed noisy versions per clean segment via
          eval_pairs_per_clean
        - length = len(self.fixed_pairs)

    Returns a dictionary with:
        noisy_mag:       [F, T]
        noisy_real:      [F, T]
        noisy_imag:      [F, T]
        clean_mag:       [F, T]
        clean_real:      [F, T]
        clean_imag:      [F, T]
        noisy_waveform:  [samples]
        clean_waveform:  [samples]
        snr:             float
    """

    def __init__(
        self,
        clean_files,
        noise_files,
        mode="train",
        snr_range=(SNR_MIN_DB, SNR_MAX_DB),
        augmentation_factor=1,
        eval_pairs_per_clean=1,
        base_seed=42,
        epoch=0,
        segment_length=SEGMENT_SAMPLES,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        center=True,
    ):
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val', or 'test'"

        self.mode = mode
        self.snr_range = snr_range
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.base_seed = base_seed
        self.epoch = epoch

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

        self.augmentation_factor = augmentation_factor if mode == "train" else 1
        self.eval_pairs_per_clean = eval_pairs_per_clean if mode in ["val", "test"] else 1

        # Load clean speech segments
        print(f"[FullSubNetPlus-{mode.upper()}] Loading {len(clean_files)} clean files...")
        self.clean_segments = []
        for f in clean_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.clean_segments.extend(slice_audio(audio, segment_length))

        # Load noise segments
        print(f"[FullSubNetPlus-{mode.upper()}] Loading {len(noise_files)} noise files...")
        self.noise_segments = []
        for f in noise_files:
            audio = load_audio(str(f), sr=sample_rate)
            self.noise_segments.extend(slice_audio(audio, segment_length))

        print(
            f"[FullSubNetPlus-{mode.upper()}] {len(self.clean_segments)} clean segments, "
            f"{len(self.noise_segments)} noise segments"
        )

        if mode in ["val", "test"]:
            print(
                f"[FullSubNetPlus-{mode.upper()}] Generating deterministic fixed pairs "
                f"(eval_pairs_per_clean={self.eval_pairs_per_clean})..."
            )
            self.fixed_pairs = self._generate_fixed_pairs()
            print(f"[FullSubNetPlus-{mode.upper()}] Total fixed pairs: {len(self.fixed_pairs)}")

    def set_epoch(self, epoch: int):
        """Update epoch for deterministic train-time augmentation."""
        self.epoch = epoch

    def _generate_fixed_pairs(self):
        """
        Generate deterministic fixed noisy-clean pairs for validation/test.
        """
        rng = random.Random(self.base_seed)
        pairs = []

        for clean_seg in self.clean_segments:
            for _ in range(self.eval_pairs_per_clean):
                noise_seg = rng.choice(self.noise_segments)
                snr = rng.uniform(*self.snr_range)
                noisy_seg = mix_at_snr(clean_seg, noise_seg, snr)
                pairs.append((clean_seg, noisy_seg, snr))

        return pairs

    def _get_deterministic_random(self, index: int):
        """
        Deterministic RNG for train mode so augmentation is reproducible.
        """
        clean_idx = index % len(self.clean_segments)
        aug_id = index // len(self.clean_segments)

        seed = self.base_seed + (self.epoch * 1000000) + (clean_idx * 100) + aug_id
        rng = np.random.RandomState(seed)

        return rng, clean_idx

    def _compute_complex_stft(self, audio: np.ndarray):
        """
        Compute complex STFT and return magnitude, real, and imaginary parts.
        """
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
        )

        mag = np.abs(stft).astype(np.float32)
        real = np.real(stft).astype(np.float32)
        imag = np.imag(stft).astype(np.float32)

        return mag, real, imag

    def _compute_cirm_target(self, clean_real, clean_imag, noisy_real, noisy_imag, eps=1e-8):
        """Compute complex ideal ratio mask (cIRM) target for FullSubNet+."""
        den = noisy_real ** 2 + noisy_imag ** 2 + eps
        mask_real = (clean_real * noisy_real + clean_imag * noisy_imag) / den
        mask_imag = (clean_imag * noisy_real - clean_real * noisy_imag) / den
        return mask_real.astype(np.float32), mask_imag.astype(np.float32)

    def __len__(self):
        if self.mode in ["val", "test"]:
            return len(self.fixed_pairs)
        return len(self.clean_segments) * self.augmentation_factor

    def __getitem__(self, index):
        if self.mode in ["val", "test"]:
            clean, noisy, snr = self.fixed_pairs[index]
        else:
            rng, clean_idx = self._get_deterministic_random(index)

            clean = self.clean_segments[clean_idx]
            noise_idx = rng.randint(0, len(self.noise_segments))
            noise = self.noise_segments[noise_idx]
            snr = rng.uniform(*self.snr_range)
            noisy = mix_at_snr(clean, noise, snr)

        clean_mag, clean_real, clean_imag = self._compute_complex_stft(clean)
        noisy_mag, noisy_real, noisy_imag = self._compute_complex_stft(noisy)

        target_mask_real, target_mask_imag = self._compute_cirm_target(clean_real, clean_imag, noisy_real, noisy_imag)

        return {
            "noisy_mag": torch.tensor(noisy_mag, dtype=torch.float32),
            "noisy_real": torch.tensor(noisy_real, dtype=torch.float32),
            "noisy_imag": torch.tensor(noisy_imag, dtype=torch.float32),
            "target_mask_imag": torch.tensor(target_mask_imag, dtype=torch.float32),
            "target_mask_real": torch.tensor(target_mask_real, dtype=torch.float32),
            "clean_mag": torch.tensor(clean_mag, dtype=torch.float32),
            "clean_real": torch.tensor(clean_real, dtype=torch.float32),
            "clean_imag": torch.tensor(clean_imag, dtype=torch.float32),
            "noisy_waveform": torch.tensor(noisy, dtype=torch.float32),
            "clean_waveform": torch.tensor(clean, dtype=torch.float32),
            "snr": float(snr),
        }



class ComplexWaveformDataset(Dataset):
    """
    Dataset for complex-STFT models such as DCCRN.

    Train mode:
        - deterministic augmentation across epochs via _get_deterministic_random
        - length = num_clean_segments * augmentation_factor

    Val/Test mode:
        - fixed deterministic noisy-clean pairs
        - can generate multiple fixed noisy versions per clean segment via
          eval_pairs_per_clean
        - length = len(self.fixed_pairs)

    Returns:
        {
            "noisy": [2, F, T]   complex STFT with channels [real, imag]
            "clean": [2, F, T]   complex STFT with channels [real, imag]
            "snr": float
            "clean_waveform": [samples]
            "noisy_waveform": [samples]
        }
    """

    def __init__(
        self,
        clean_files,
        noise_files,
        mode="train",
        snr_range=(SNR_MIN_DB, SNR_MAX_DB),
        augmentation_factor=1,
        eval_pairs_per_clean=1,
        base_seed=42,
        epoch=0,
        segment_length=SEGMENT_SAMPLES,
        sample_rate=SAMPLE_RATE,
    ):
        """
        Args:
            clean_files: List of clean audio file paths
            noise_files: List of noise file paths
            mode: 'train', 'val', or 'test'
            snr_range: (min_snr, max_snr) in dB
            augmentation_factor: Number of augmented versions per clean segment in train mode
            eval_pairs_per_clean: Number of fixed noisy versions per clean segment in val/test
            base_seed: Base random seed for reproducibility
            epoch: Current training epoch (changes train augmentation only)
            segment_length: Segment length in samples
            sample_rate: Audio sample rate
        """
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val', or 'test'"

        self.mode = mode
        self.snr_range = snr_range
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.base_seed = base_seed
        self.epoch = epoch

        self.augmentation_factor = augmentation_factor if mode == "train" else 1
        self.eval_pairs_per_clean = eval_pairs_per_clean if mode in ["val", "test"] else 1

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

        print(
            f"[{mode.upper()}] Base clean segments: {len(self.clean_segments)}, "
            f"noise segments: {len(self.noise_segments)}, "
            f"augmentation_factor: {self.augmentation_factor}"
        )

        # For val/test, generate fixed deterministic pairs once
        if mode in ["val", "test"]:
            print(
                f"[{mode.upper()}] Generating deterministic fixed pairs "
                f"(eval_pairs_per_clean={self.eval_pairs_per_clean})..."
            )
            self.fixed_pairs = self._generate_fixed_pairs()
            print(f"[{mode.upper()}] Total fixed pairs: {len(self.fixed_pairs)}")

    def set_epoch(self, epoch):
        """Update epoch for deterministic train-time augmentation."""
        self.epoch = epoch

    def _generate_fixed_pairs(self):
        """
        Generate deterministic fixed pairs for val/test.

        Each clean segment gets `eval_pairs_per_clean` fixed noisy versions.
        """
        rng = random.Random(self.base_seed)
        pairs = []

        for clean_seg in self.clean_segments:
            for _ in range(self.eval_pairs_per_clean):
                noise_seg = rng.choice(self.noise_segments)
                snr = rng.uniform(*self.snr_range)
                noisy_seg = mix_at_snr(clean_seg, noise_seg, snr)
                pairs.append((clean_seg, noisy_seg, snr))

        return pairs

    def _get_deterministic_random(self, index):
        """
        Get a deterministic random state for this (epoch, index).

        Ensures:
        - same index at same epoch -> same augmentation
        - same index at different epochs -> different augmentation
        - same across model runs -> fair comparison
        """
        clean_idx = index % len(self.clean_segments)
        aug_id = index // len(self.clean_segments)

        seed = self.base_seed + (self.epoch * 1000000) + (clean_idx * 100) + aug_id
        rng = np.random.RandomState(seed)

        return rng, clean_idx

    def __len__(self):
        if self.mode in ["val", "test"]:
            return len(self.fixed_pairs)
        return len(self.clean_segments) * self.augmentation_factor

    def __getitem__(self, index):
        if self.mode in ["val", "test"]:
            clean, noisy, snr = self.fixed_pairs[index]
        else:
            rng, clean_idx = self._get_deterministic_random(index)

            clean = self.clean_segments[clean_idx]

            noise_idx = rng.randint(0, len(self.noise_segments))
            noise = self.noise_segments[noise_idx]

            snr = rng.uniform(*self.snr_range)
            noisy = mix_at_snr(clean, noise, snr)

        # Compute complex STFTs
        clean_stft = compute_complex_stft(clean)
        noisy_stft = compute_complex_stft(noisy)

        # Split into real/imag channels
        clean_real = np.real(clean_stft)
        clean_imag = np.imag(clean_stft)
        noisy_real = np.real(noisy_stft)
        noisy_imag = np.imag(noisy_stft)

        clean_tensor = torch.stack(
            [
                torch.tensor(clean_real, dtype=torch.float32),
                torch.tensor(clean_imag, dtype=torch.float32),
            ],
            dim=0,
        )  # [2, F, T]

        noisy_tensor = torch.stack(
            [
                torch.tensor(noisy_real, dtype=torch.float32),
                torch.tensor(noisy_imag, dtype=torch.float32),
            ],
            dim=0,
        )  # [2, F, T]

        return {
            "noisy": noisy_tensor,
            "clean": clean_tensor,
            "snr": float(snr),
            "clean_waveform": torch.tensor(clean, dtype=torch.float32),
            "noisy_waveform": torch.tensor(noisy, dtype=torch.float32),
        }


print("✓ Dataset classes defined")