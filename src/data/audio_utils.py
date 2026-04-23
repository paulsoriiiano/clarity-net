import numpy as np
import librosa
from pathlib import Path

# Audio parameters
SAMPLE_RATE = 16000
SEGMENT_DURATION = 4.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
N_FFT = 512
HOP_LENGTH = 128
SNR_MIN_DB = 0
SNR_MAX_DB = 20

def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    return audio

def slice_audio(audio: np.ndarray, segment_length: int = SEGMENT_SAMPLES) -> list:
    """Slice audio into fixed-length segments."""
    segments = []
    for start in range(0, len(audio), segment_length):
        segment = audio[start:start + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        segments.append(segment)
    return segments

def mix_at_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Mix clean speech with noise at target SNR."""
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if clean_power == 0 or noise_power == 0:
        return clean.copy()
    desired_noise_power = clean_power / (10 ** (target_snr_db / 10))
    noise_scaled = noise * np.sqrt(desired_noise_power / noise_power)
    noisy = np.clip(clean + noise_scaled, -1.0, 1.0)
    return noisy

def compute_stft(audio: np.ndarray) -> tuple:
    """Compute STFT magnitude and phase."""
    stft_complex = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft_complex)
    phase = np.angle(stft_complex)
    return magnitude, phase

def extract_category_from_filename(filename: str) -> int:
    """Extract ESC-50 category number from filename."""
    try:
        parts = Path(filename).stem.split('-')
        if len(parts) >= 4:
            return int(parts[3])
    except (ValueError, IndexError):
        pass
    return None

def reconstruct_audio(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Reconstruct waveform from magnitude and phase spectrograms."""
    stft_complex = magnitude * np.exp(1j * phase)
    audio = librosa.istft(stft_complex, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return audio