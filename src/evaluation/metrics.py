"""
Unified metrics for Baseline CNN, DCCRN, and FullSubNet models.

Handles:
- Baseline CNN: Spectrogram-based (magnitude output)
- DCCRN: Complex spectrogram-based (real/imag output)
- FullSubNet: Raw waveform-based (time-domain output)
"""
import warnings
import librosa
import numpy as np

import torch
import torch.nn as nn

from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

# Try to import from audio_utils, but handle if not available
try:
    from src.data.audio_utils import N_FFT, HOP_LENGTH
except ImportError:
    N_FFT = 512
    HOP_LENGTH = 128

def compute_pesq(clean, enhanced, sr=16000, min_length=0.5):
    """
    Compute PESQ score with robust error handling.
    
    Args:
        clean: Clean audio signal
        enhanced: Enhanced audio signal
        sr: Sample rate (must be 8000 or 16000 for PESQ)
        min_length: Minimum length in seconds (default 0.5s)
    
    Returns:
        PESQ score (range -0.5 to 4.5) or None if computation fails
    """
    try:
        # Ensure both signals have same length
        min_len = min(len(clean), len(enhanced))
        
        # Check minimum length (PESQ needs at least 0.5 seconds)
        min_samples = int(min_length * sr)
        if min_len < min_samples:
            return None
        
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        # Check for silence (all zeros or very low energy)
        if np.max(np.abs(clean)) < 1e-6 or np.max(np.abs(enhanced)) < 1e-6:
            return None
        
        # Normalize BOTH by the same factor to preserve relative amplitudes
        max_val = max(np.max(np.abs(clean)), np.max(np.abs(enhanced)))
        clean = clean / (max_val + 1e-8)
        enhanced = enhanced / (max_val + 1e-8)
        
        # PESQ requires mode 'wb' (wideband) for 16kHz or 'nb' (narrowband) for 8kHz
        mode = 'wb' if sr == 16000 else 'nb'
        
        # Suppress PESQ warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pesq(sr, clean, enhanced, mode)
        
        return score
    
    except Exception as e:
        # Return None on any error (don't crash training)
        return None


def compute_stoi_score(clean, enhanced, sr=16000, min_length=0.4):
    """
    Compute STOI score with robust error handling.
    
    Args:
        clean: Clean audio signal
        enhanced: Enhanced audio signal
        sr: Sample rate
        min_length: Minimum length in seconds (default 0.4s for STOI)
    
    Returns:
        STOI score (range 0 to 1) or None if computation fails
    """
    try:
        # Ensure both signals have same length
        min_len = min(len(clean), len(enhanced))
        
        # Check minimum length (STOI needs at least 400ms)
        min_samples = int(min_length * sr)
        if min_len < min_samples:
            return None
        
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        # Check for silence
        if np.max(np.abs(clean)) < 1e-6 or np.max(np.abs(enhanced)) < 1e-6:
            return None
        
        # Normalize BOTH by the same factor to preserve relative amplitudes
        max_val = max(np.max(np.abs(clean)), np.max(np.abs(enhanced)))
        clean = clean / (max_val + 1e-8)
        enhanced = enhanced / (max_val + 1e-8)
        
        # Suppress STOI warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = stoi(clean, enhanced, sr, extended=False)
        
        # STOI should be between 0 and 1, but sometimes returns weird values
        if not (0 <= score <= 1):
            return None
        
        return score
    
    except Exception as e:
        # Return None on any error
        return None


def compute_si_snr_score(clean, enhanced, metric=None):
    """
    Compute SI-SNR score in dB using torchmetrics.
    Returns float or None on failure.
    """
    try:
        min_len = min(len(clean), len(enhanced))
        if min_len == 0:
            return None

        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        if np.max(np.abs(clean)) < 1e-6 or np.max(np.abs(enhanced)) < 1e-6:
            return None

        clean_t = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)       # [1, T]
        enhanced_t = torch.tensor(enhanced, dtype=torch.float32).unsqueeze(0)  # [1, T]

        if metric is None:
            metric = ScaleInvariantSignalNoiseRatio()

        score = metric(enhanced_t, clean_t)
        return float(score.item())

    except Exception:
        return None


def validate_with_metrics(model, loader, criterion, device,
                          max_pesq_samples=50, hop_length=None, n_fft=None, sr=16000):
    """
    Unified validation for:
    - Baseline CNN: magnitude spectrogram model
    - DCCRN: complex spectrogram model
    - FullSubNet: waveform model
    - FullSubNet+: explicit mag/real/imag inputs with complex mask output

    Reports:
    - loss
    - PESQ
    - STOI
    - SI-SNR

    Notes:
    - ``max_pesq_samples`` is kept for backward compatibility, but it now caps
      total metric attempts, not only successful PESQ samples.
    - Magnitude-model audio reconstruction mirrors ``evaluate_with_metrics()``:
      prefer ``batch["phase"]`` and fall back to ``batch["noisy_stft"]`` only
      when needed.
    """
    if hop_length is None:
        hop_length = HOP_LENGTH
    if n_fft is None:
        n_fft = N_FFT

    def _to_numpy(x):
        """Safely move tensor-like values to a NumPy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _istft(stft_matrix):
        """Reconstruct waveform from an STFT matrix."""
        return librosa.istft(stft_matrix, hop_length=hop_length, n_fft=n_fft)

    def _phase_from_batch(batch, i):
        """Return a complex unit phase matrix for magnitude-only reconstruction."""
        if "phase" in batch:
            phase = _to_numpy(batch["phase"][i])
            if np.iscomplexobj(phase):
                return phase
            return np.exp(1j * phase)

        if "noisy_stft" in batch:
            noisy_stft = _to_numpy(batch["noisy_stft"][i])
            if np.iscomplexobj(noisy_stft):
                return librosa.magphase(noisy_stft)[1]

        raise KeyError(
            "Magnitude validation needs batch['phase'] or complex batch['noisy_stft'] "
            "to reconstruct enhanced audio."
        )

    def _clean_waveform_from_batch(batch, i, fallback_stft=None):
        """Prefer dataset waveform; otherwise reconstruct from a fallback STFT."""
        if "clean_waveform" in batch:
            return _to_numpy(batch["clean_waveform"][i])
        if fallback_stft is not None:
            return _istft(fallback_stft)
        raise KeyError("batch['clean_waveform'] is required for metric validation.")

    def _score_one_pair(clean_audio, enhanced_audio, si_snr_metric):
        """Compute all validation metrics for one clean/enhanced pair."""
        pesq_score = compute_pesq(clean_audio, enhanced_audio, sr)
        stoi_score = compute_stoi_score(clean_audio, enhanced_audio, sr)
        si_snr_score = compute_si_snr_score(clean_audio, enhanced_audio, si_snr_metric)
        return pesq_score, stoi_score, si_snr_score

    model.eval()
    total_loss = 0.0
    pesq_scores = []
    stoi_scores = []
    si_snr_scores = []
    pesq_failed = 0
    stoi_failed = 0
    si_snr_failed = 0
    metric_attempts = 0

    # Reuse one torchmetrics instance, matching evaluate_with_metrics().
    si_snr_metric = ScaleInvariantSignalNoiseRatio()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):

            # ------------------------------------------------------------------
            # FullSubNet+ branch
            # ------------------------------------------------------------------
            is_fullsubnet_plus = all(
                k in batch for k in [
                    "noisy_mag", "noisy_real", "noisy_imag",
                    "clean_mag", "clean_real", "clean_imag"
                ]
            )

            if is_fullsubnet_plus:
                noisy_mag = batch["noisy_mag"].to(device)      # [B, F, T]
                noisy_real = batch["noisy_real"].to(device)    # [B, F, T]
                noisy_imag = batch["noisy_imag"].to(device)    # [B, F, T]

                clean_mag = batch["clean_mag"].to(device)      # [B, F, T]
                clean_real = batch["clean_real"].to(device)    # [B, F, T]
                clean_imag = batch["clean_imag"].to(device)    # [B, F, T]

                pred_mask = model(noisy_mag, noisy_real, noisy_imag)   # [B, 2, F, T]
                enh_real, enh_imag = model.apply_mask(pred_mask, noisy_real, noisy_imag)
                enh_mag = torch.sqrt(enh_real.pow(2) + enh_imag.pow(2) + 1e-8)

                try:
                    loss = criterion(
                        enh_real,
                        enh_imag,
                        clean_real,
                        clean_imag,
                        enhanced_mag=enh_mag,
                        clean_mag=clean_mag,
                        pred_mask=pred_mask,
                    )
                except TypeError:
                    loss = criterion(
                        (enh_real, enh_imag, enh_mag),
                        (clean_real, clean_imag, clean_mag),
                    )

                total_loss += loss.item()

                if metric_attempts < max_pesq_samples:
                    batch_size = noisy_mag.shape[0]
                    max_remaining = max_pesq_samples - metric_attempts

                    for i in range(min(batch_size, max_remaining)):
                        metric_attempts += 1
                        try:
                            enhanced_stft = (
                                _to_numpy(enh_real[i]) + 1j * _to_numpy(enh_imag[i])
                            )
                            clean_stft = (
                                _to_numpy(clean_real[i]) + 1j * _to_numpy(clean_imag[i])
                            )
                            enhanced_audio = _istft(enhanced_stft)
                            clean_audio = _clean_waveform_from_batch(
                                batch, i, fallback_stft=clean_stft
                            )

                            pesq_score, stoi_score, si_snr_score = _score_one_pair(
                                clean_audio, enhanced_audio, si_snr_metric
                            )

                            if pesq_score is not None:
                                pesq_scores.append(pesq_score)
                            else:
                                pesq_failed += 1

                            if stoi_score is not None:
                                stoi_scores.append(stoi_score)
                            else:
                                stoi_failed += 1

                            if si_snr_score is not None:
                                si_snr_scores.append(si_snr_score)
                            else:
                                si_snr_failed += 1

                        except Exception:
                            pesq_failed += 1
                            stoi_failed += 1
                            si_snr_failed += 1
                            continue

            # ------------------------------------------------------------------
            # Existing branches
            # ------------------------------------------------------------------
            else:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)

                is_waveform = (noisy.dim() == 2)                          # [B, T]
                is_complex = (noisy.dim() == 4 and noisy.shape[1] == 2)   # [B, 2, F, T]
                is_magnitude = (noisy.dim() == 4 and noisy.shape[1] == 1) # [B, 1, F, T]

                enhanced = model(noisy)

                if is_waveform and enhanced.dim() == 3 and enhanced.size(1) == 1:
                    enhanced_for_loss = enhanced.squeeze(1)
                else:
                    enhanced_for_loss = enhanced

                loss = criterion(enhanced_for_loss, clean)
                total_loss += loss.item()

                if metric_attempts < max_pesq_samples:
                    batch_size = noisy.shape[0]
                    max_remaining = max_pesq_samples - metric_attempts

                    for i in range(min(batch_size, max_remaining)):
                        metric_attempts += 1
                        try:
                            if is_waveform:
                                clean_audio = _to_numpy(clean[i])
                                enhanced_audio = _to_numpy(enhanced[i]).squeeze()

                            elif is_complex:
                                enhanced_real = _to_numpy(enhanced[i, 0])
                                enhanced_imag = _to_numpy(enhanced[i, 1])
                                enhanced_stft = enhanced_real + 1j * enhanced_imag
                                enhanced_audio = _istft(enhanced_stft)

                                clean_real = _to_numpy(clean[i, 0])
                                clean_imag = _to_numpy(clean[i, 1])
                                clean_stft = clean_real + 1j * clean_imag
                                clean_audio = _clean_waveform_from_batch(
                                    batch, i, fallback_stft=clean_stft
                                )

                            elif is_magnitude:
                                enhanced_mag_i = _to_numpy(enhanced[i, 0])
                                clean_mag_i = _to_numpy(clean[i, 0])
                                phase = _phase_from_batch(batch, i)

                                enhanced_stft = enhanced_mag_i * phase
                                clean_stft = clean_mag_i * phase

                                enhanced_audio = _istft(enhanced_stft)
                                clean_audio = _clean_waveform_from_batch(
                                    batch, i, fallback_stft=clean_stft
                                )

                            else:
                                pesq_failed += 1
                                stoi_failed += 1
                                si_snr_failed += 1
                                continue

                            pesq_score, stoi_score, si_snr_score = _score_one_pair(
                                clean_audio, enhanced_audio, si_snr_metric
                            )

                            if pesq_score is not None:
                                pesq_scores.append(pesq_score)
                            else:
                                pesq_failed += 1

                            if stoi_score is not None:
                                stoi_scores.append(stoi_score)
                            else:
                                stoi_failed += 1

                            if si_snr_score is not None:
                                si_snr_scores.append(si_snr_score)
                            else:
                                si_snr_failed += 1

                        except Exception:
                            pesq_failed += 1
                            stoi_failed += 1
                            si_snr_failed += 1
                            continue

            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    avg_pesq = np.mean(pesq_scores) if len(pesq_scores) > 0 else 0.0
    avg_stoi = np.mean(stoi_scores) if len(stoi_scores) > 0 else 0.0
    avg_si_snr = np.mean(si_snr_scores) if len(si_snr_scores) > 0 else 0.0
    std_si_snr = np.std(si_snr_scores) if len(si_snr_scores) > 0 else 0.0

    if pesq_failed > 0 or stoi_failed > 0 or si_snr_failed > 0:
        print(
            f"    [Metrics] PESQ: {len(pesq_scores)}/{metric_attempts} valid, "
            f"STOI: {len(stoi_scores)}/{metric_attempts} valid, "
            f"SI-SNR: {len(si_snr_scores)}/{metric_attempts} valid"
        )

    return {
        "loss": avg_loss,
        "pesq": avg_pesq,
        "stoi": avg_stoi,
        "si_snr": avg_si_snr,
        "si_snr_std": std_si_snr,
        "pesq_count": len(pesq_scores),
        "stoi_count": len(stoi_scores),
        "si_snr_count": len(si_snr_scores),
    }


def evaluate_with_metrics(model, test_loader, device, desc="Evaluation",
                          hop_length=None, n_fft=None, sr=16000):
    """
    Unified evaluation for:
    - Baseline CNN
    - DCCRN
    - FullSubNet
    - FullSubNet+

    Reports:
    - MSE
    - PESQ
    - STOI
    - SI-SNR
    """
    if hop_length is None:
        hop_length = HOP_LENGTH
    if n_fft is None:
        n_fft = N_FFT

    model.eval()

    total_mse = 0.0
    pesq_scores = []
    stoi_scores = []
    si_snr_scores = []
    audio_samples = []
    max_samples = 5

    si_snr_metric = ScaleInvariantSignalNoiseRatio()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=desc)):

            # ------------------------------------------------------------------
            # FullSubNet+ branch
            # ------------------------------------------------------------------
            is_fullsubnet_plus = all(
                k in batch for k in [
                    "noisy_mag", "noisy_real", "noisy_imag",
                    "clean_mag", "clean_real", "clean_imag"
                ]
            )

            if is_fullsubnet_plus:
                noisy_mag = batch["noisy_mag"].to(device)
                noisy_real = batch["noisy_real"].to(device)
                noisy_imag = batch["noisy_imag"].to(device)

                clean_mag = batch["clean_mag"].to(device)
                clean_real = batch["clean_real"].to(device)
                clean_imag = batch["clean_imag"].to(device)

                snr = batch["snr"]

                pred_mask = model(noisy_mag, noisy_real, noisy_imag)  # [B, 2, F, T]
                enh_real, enh_imag = model.apply_mask(pred_mask, noisy_real, noisy_imag)
                enh_mag = torch.sqrt(enh_real.pow(2) + enh_imag.pow(2) + 1e-8)

                # Use complex MSE for reporting
                mse_real = nn.functional.mse_loss(enh_real, clean_real)
                mse_imag = nn.functional.mse_loss(enh_imag, clean_imag)
                mse = 0.5 * (mse_real + mse_imag)
                total_mse += mse.item()

                batch_size = noisy_mag.shape[0]

                for i in range(batch_size):
                    try:
                        enhanced_stft = enh_real[i].cpu().numpy() + 1j * enh_imag[i].cpu().numpy()
                        noisy_stft = noisy_real[i].cpu().numpy() + 1j * noisy_imag[i].cpu().numpy()
                        clean_stft = clean_real[i].cpu().numpy() + 1j * clean_imag[i].cpu().numpy()

                        enhanced_audio = librosa.istft(
                            enhanced_stft, hop_length=hop_length, n_fft=n_fft
                        )
                        noisy_audio = batch["noisy_waveform"][i].cpu().numpy()
                        clean_audio = batch["clean_waveform"][i].cpu().numpy()

                    except Exception:
                        continue

                    pesq_score = compute_pesq(clean_audio, enhanced_audio, sr)
                    stoi_score = compute_stoi_score(clean_audio, enhanced_audio, sr)
                    si_snr_score = compute_si_snr_score(clean_audio, enhanced_audio, si_snr_metric)

                    if pesq_score is not None:
                        pesq_scores.append(pesq_score)
                    if stoi_score is not None:
                        stoi_scores.append(stoi_score)
                    if si_snr_score is not None:
                        si_snr_scores.append(si_snr_score)

                    if len(audio_samples) < max_samples:
                        audio_samples.append({
                            "clean": clean_audio,
                            "noisy": noisy_audio,
                            "enhanced": enhanced_audio,
                            "clean_spec": clean_stft,
                            "noisy_spec": noisy_stft,
                            "enhanced_spec": enhanced_stft,
                            "enhanced_mag": enh_mag[i].cpu().numpy(),
                            "clean_mag": clean_mag[i].cpu().numpy(),
                            "noisy_mag": noisy_mag[i].cpu().numpy(),
                            "snr": snr[i].item() if isinstance(snr, torch.Tensor) else snr,
                            "pesq": pesq_score,
                            "stoi": stoi_score,
                            "si_snr": si_snr_score,
                        })

            # ------------------------------------------------------------------
            # Existing branches
            # ------------------------------------------------------------------
            else:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                snr = batch["snr"]

                is_waveform = (noisy.dim() == 2)
                is_complex = (noisy.dim() == 4 and noisy.shape[1] == 2)
                is_magnitude = (noisy.dim() == 4 and noisy.shape[1] == 1)

                enhanced = model(noisy)

                if is_waveform and enhanced.dim() == 3 and enhanced.size(1) == 1:
                    enhanced_for_mse = enhanced.squeeze(1)
                else:
                    enhanced_for_mse = enhanced

                mse = nn.functional.mse_loss(enhanced_for_mse, clean)
                total_mse += mse.item()

                batch_size = noisy.shape[0]

                for i in range(batch_size):
                    try:
                        if is_waveform:
                            clean_audio = clean[i].cpu().numpy()
                            noisy_audio = noisy[i].cpu().numpy()
                            enhanced_audio = enhanced[i].squeeze().cpu().numpy()

                        elif is_complex:
                            enhanced_real = enhanced[i, 0].cpu().numpy()
                            enhanced_imag = enhanced[i, 1].cpu().numpy()
                            enhanced_stft = enhanced_real + 1j * enhanced_imag
                            enhanced_audio = librosa.istft(
                                enhanced_stft, hop_length=hop_length, n_fft=n_fft
                            )

                            clean_real = clean[i, 0].cpu().numpy()
                            clean_imag = clean[i, 1].cpu().numpy()
                            clean_stft = clean_real + 1j * clean_imag
                            clean_audio = batch["clean_waveform"][i].cpu().numpy()

                            noisy_real_i = noisy[i, 0].cpu().numpy()
                            noisy_imag_i = noisy[i, 1].cpu().numpy()
                            noisy_stft = noisy_real_i + 1j * noisy_imag_i
                            noisy_audio = batch["noisy_waveform"][i].cpu().numpy()

                        elif is_magnitude:
                            if "phase" not in batch:
                                continue

                            phase = batch["phase"][i].cpu().numpy()
                            noisy_mag_i = noisy[i, 0].cpu().numpy()
                            clean_mag_i = clean[i, 0].cpu().numpy()
                            enhanced_mag_i = enhanced[i, 0].cpu().numpy()

                            noisy_stft = noisy_mag_i * np.exp(1j * phase)
                            enhanced_stft = enhanced_mag_i * np.exp(1j * phase)

                            noisy_audio = batch["noisy_waveform"][i].cpu().numpy()
                            clean_audio = batch["clean_waveform"][i].cpu().numpy()
                            enhanced_audio = librosa.istft(
                                enhanced_stft, hop_length=hop_length, n_fft=n_fft
                            )
                        else:
                            continue

                    except Exception:
                        continue

                    pesq_score = compute_pesq(clean_audio, enhanced_audio, sr)
                    stoi_score = compute_stoi_score(clean_audio, enhanced_audio, sr)
                    si_snr_score = compute_si_snr_score(clean_audio, enhanced_audio, si_snr_metric)

                    if pesq_score is not None:
                        pesq_scores.append(pesq_score)
                    if stoi_score is not None:
                        stoi_scores.append(stoi_score)
                    if si_snr_score is not None:
                        si_snr_scores.append(si_snr_score)

                    if len(audio_samples) < max_samples:
                        sample_dict = {
                            "clean": clean_audio,
                            "noisy": noisy_audio,
                            "enhanced": enhanced_audio,
                            "snr": snr[i].item() if isinstance(snr, torch.Tensor) else snr,
                            "pesq": pesq_score,
                            "stoi": stoi_score,
                            "si_snr": si_snr_score,
                        }

                        if is_complex or is_magnitude:
                            if is_complex:
                                sample_dict["clean_spec"] = clean_stft
                                sample_dict["noisy_spec"] = noisy_stft
                                sample_dict["enhanced_spec"] = enhanced_stft
                            else:
                                sample_dict["clean_spec"] = clean_mag_i
                                sample_dict["noisy_spec"] = noisy_mag_i
                                sample_dict["enhanced_spec"] = enhanced_mag_i

                        audio_samples.append(sample_dict)

    results = {
        "mse": total_mse / len(test_loader) if len(test_loader) > 0 else 0.0,
        "pesq_mean": np.mean(pesq_scores) if pesq_scores else 0.0,
        "pesq_std": np.std(pesq_scores) if pesq_scores else 0.0,
        "stoi_mean": np.mean(stoi_scores) if stoi_scores else 0.0,
        "stoi_std": np.std(stoi_scores) if stoi_scores else 0.0,
        "si_snr_mean": np.mean(si_snr_scores) if si_snr_scores else 0.0,
        "si_snr_std": np.std(si_snr_scores) if si_snr_scores else 0.0,
        "pesq_scores": pesq_scores,
        "stoi_scores": stoi_scores,
        "si_snr_scores": si_snr_scores,
        "audio_samples": audio_samples,
    }

    return results


# =============================================================================
# Additional utilities
# =============================================================================

def compute_stoi_extended(clean, enhanced, sr=16000, min_length=0.4):
    """
    Extended STOI - more robust to short segments.
    
    Extended STOI (ESTOI) is better for:
    - Short utterances
    - Low SNR conditions
    - Handles edge cases better
    """
    try:
        min_len = min(len(clean), len(enhanced))
        min_samples = int(min_length * sr)
        
        if min_len < min_samples:
            return None
        
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        if np.max(np.abs(clean)) < 1e-6 or np.max(np.abs(enhanced)) < 1e-6:
            return None
        
        # Normalize BOTH by the same factor to preserve relative amplitudes
        max_val = max(np.max(np.abs(clean)), np.max(np.abs(enhanced)))
        clean = clean / (max_val + 1e-8)
        enhanced = enhanced / (max_val + 1e-8)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use extended=True for better robustness
            score = stoi(clean, enhanced, sr, extended=True)
        
        if not (-1 <= score <= 1):  # Extended STOI can be negative
            return None
        
        return score
    
    except Exception:
        return None


def diagnose_audio_issues(clean_audio, enhanced_audio, sr=16000):
    """
    Diagnose why PESQ/STOI might be failing.
    Call this if you're getting lots of failures.
    """
    print(f"Clean audio:")
    print(f"  Length: {len(clean_audio)} samples ({len(clean_audio)/sr:.2f}s)")
    print(f"  Min: {clean_audio.min():.6f}, Max: {clean_audio.max():.6f}")
    print(f"  Mean: {clean_audio.mean():.6f}, Std: {clean_audio.std():.6f}")
    print(f"  Energy: {np.sum(clean_audio**2):.6f}")
    
    print(f"\nEnhanced audio:")
    print(f"  Length: {len(enhanced_audio)} samples ({len(enhanced_audio)/sr:.2f}s)")
    print(f"  Min: {enhanced_audio.min():.6f}, Max: {enhanced_audio.max():.6f}")
    print(f"  Mean: {enhanced_audio.mean():.6f}, Std: {enhanced_audio.std():.6f}")
    print(f"  Energy: {np.sum(enhanced_audio**2):.6f}")
    
    # Check for issues
    issues = []
    if len(clean_audio) < sr * 0.4:
        issues.append("Audio too short for STOI (< 0.4s)")
    if len(clean_audio) < sr * 0.5:
        issues.append("Audio too short for PESQ (< 0.5s)")
    if np.max(np.abs(clean_audio)) < 1e-6:
        issues.append("Clean audio is nearly silent")
    if np.max(np.abs(enhanced_audio)) < 1e-6:
        issues.append("Enhanced audio is nearly silent")
    
    if issues:
        print("\n⚠️  Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No obvious issues detected")


if __name__ == "__main__":
    # Test the functions
    
    # Load a test file
    audio, sr = librosa.load(librosa.ex('trumpet'), sr=16000, duration=4)
    
    # Test PESQ
    pesq_score = compute_pesq(audio, audio * 0.9, sr)
    print(f"PESQ (should work): {pesq_score}")
    
    # Test STOI
    stoi_score = compute_stoi_score(audio, audio * 0.9, sr)
    print(f"STOI (should work): {stoi_score}")

    # Test SI-SNR    
    si_snr_score = compute_si_snr_score(audio, audio * 0.9)
    print(f"SI-SNR (should work): {si_snr_score} dB")
    
    # Test with very short audio
    short_audio = audio[:1000]  # ~0.06 seconds
    pesq_score = compute_pesq(short_audio, short_audio, sr)
    stoi_score = compute_stoi_score(short_audio, short_audio, sr)
    si_snr_score = compute_si_snr_score(short_audio, short_audio)
    print(f"\nShort audio (should fail gracefully):")
    print(f"  PESQ: {pesq_score}")
    print(f"  STOI: {stoi_score}")

    print("\n✓ Unified metrics ready for Baseline CNN, DCCRN, and FullSubNet models")


