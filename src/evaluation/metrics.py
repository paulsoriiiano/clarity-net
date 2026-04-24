"""
Robust PESQ/STOI computation with proper error handling.

Replace your compute_pesq and compute_stoi functions with these versions.
"""

import numpy as np
from pesq import pesq
import torch
import torch.nn as nn
import warnings
from pystoi import stoi as compute_stoi_metric
from src.data.audio_utils import reconstruct_audio
from tqdm import tqdm

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
        
        # Normalize to prevent clipping issues
        clean = clean / (np.max(np.abs(clean)) + 1e-8)
        enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8)
        
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
        
        # Normalize
        clean = clean / (np.max(np.abs(clean)) + 1e-8)
        enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8)
        
        # Suppress STOI warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = compute_stoi_metric(clean, enhanced, sr, extended=False)
        
        # STOI should be between 0 and 1, but sometimes returns weird values
        if not (0 <= score <= 1):
            return None
        
        return score
    
    except Exception as e:
        # Return None on any error
        return None


def validate_with_metrics(model, loader, criterion, device, 
                          max_pesq_samples=50, hop_length=128, n_fft=512, sr=16000):
    """
    Validate with MSE, PESQ, and STOI metrics.
    Uses robust metric computation that handles edge cases.
    """
    import torch
    from tqdm import tqdm
    import librosa
    
    model.eval()
    total_loss = 0
    pesq_scores = []
    stoi_scores = []
    
    # Track failures for debugging
    pesq_failed = 0
    stoi_failed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            noisy_spec = batch['noisy'].to(device)
            clean_spec = batch['clean'].to(device)
            phase = batch['phase']  # Keep on CPU
            
            # Forward pass
            enhanced_spec = model(noisy_spec)
            
            # Compute MSE loss
            loss = criterion(enhanced_spec, clean_spec)
            total_loss += loss.item()
            
            # Compute PESQ/STOI on subset
            if len(pesq_scores) < max_pesq_samples:
                batch_size = noisy_spec.shape[0]
                
                for i in range(min(batch_size, max_pesq_samples - len(pesq_scores))):
                    # Get spectrograms
                    clean_mag = clean_spec[i, 0].cpu().numpy()
                    enhanced_mag = enhanced_spec[i, 0].cpu().numpy()
                    phase_np = phase[i].numpy()
                    
                    # Reconstruct audio
                    clean_stft = clean_mag * np.exp(1j * phase_np)
                    enhanced_stft = enhanced_mag * np.exp(1j * phase_np)
                    
                    clean_audio = librosa.istft(clean_stft, hop_length=hop_length, n_fft=n_fft)
                    enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, n_fft=n_fft)
                    
                    # Compute PESQ (returns None on failure)
                    pesq_score = compute_pesq(clean_audio, enhanced_audio, sr)
                    if pesq_score is not None:
                        pesq_scores.append(pesq_score)
                    else:
                        pesq_failed += 1
                    
                    # Compute STOI (returns None on failure)
                    stoi_score = compute_stoi_score(clean_audio, enhanced_audio, sr)
                    if stoi_score is not None:
                        stoi_scores.append(stoi_score)
                    else:
                        stoi_failed += 1
    
    # Compute averages (handle empty lists)
    avg_loss = total_loss / len(loader)
    avg_pesq = np.mean(pesq_scores) if len(pesq_scores) > 0 else 0.0
    avg_stoi = np.mean(stoi_scores) if len(stoi_scores) > 0 else 0.0
    
    # Print diagnostics if many failures
    total_attempted = len(pesq_scores) + pesq_failed
    if pesq_failed > 0 or stoi_failed > 0:
        print(f"    [Metrics] PESQ: {len(pesq_scores)}/{total_attempted} valid, "
              f"STOI: {len(stoi_scores)}/{total_attempted} valid")
    
    return {
        'loss': avg_loss,
        'pesq': avg_pesq,
        'stoi': avg_stoi,
        'pesq_count': len(pesq_scores),
        'stoi_count': len(stoi_scores)
    }


# =============================================================================
# Alternative: Use extended STOI (more robust)
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
        
        # Normalize
        clean = clean / (np.max(np.abs(clean)) + 1e-8)
        enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use extended=True for better robustness
            score = compute_stoi_metric(clean, enhanced, sr, extended=True)
        
        if not (-1 <= score <= 1):  # Extended STOI can be negative
            return None
        
        return score
    
    except Exception:
        return None


# =============================================================================
# Debugging helper
# =============================================================================

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


def evaluate_with_metrics(model, test_loader, device, desc="Evaluation"):
    """
    Evaluate model with MSE, PESQ, and STOI metrics.
    Also collects audio samples for visualization.
    """
    model.eval()
    
    total_mse = 0
    pesq_scores = []
    stoi_scores = []
    
    # Collect a few samples for visualization
    audio_samples = []
    max_samples = 5
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=desc)):
            noisy_spec = batch['noisy'].to(device)
            clean_spec = batch['clean'].to(device)
            noisy_phase = batch['phase'].numpy()
            snr = batch['snr']
            
            # Forward pass
            enhanced_spec = model(noisy_spec)
            
            # MSE loss
            mse = nn.functional.mse_loss(enhanced_spec, clean_spec)
            total_mse += mse.item()
            
            # Convert back to audio for PESQ/STOI
            batch_size = noisy_spec.shape[0]
            
            for i in range(batch_size):
                # Get spectrograms as numpy
                noisy_mag = noisy_spec[i, 0].cpu().numpy()
                clean_mag = clean_spec[i, 0].cpu().numpy()
                enhanced_mag = enhanced_spec[i, 0].cpu().numpy()
                phase = noisy_phase[i]
                
                # Reconstruct audio
                clean_audio = reconstruct_audio(clean_mag, phase)
                enhanced_audio = reconstruct_audio(enhanced_mag, phase)
                noisy_audio = reconstruct_audio(noisy_mag, phase)
                
                # Compute metrics
                pesq_score = compute_pesq(clean_audio, enhanced_audio)
                stoi_score = compute_stoi_score(clean_audio, enhanced_audio)
                
                pesq_scores.append(pesq_score)
                stoi_scores.append(stoi_score)
                
                # Save samples for visualization
                if len(audio_samples) < max_samples:
                    audio_samples.append({
                        'clean': clean_audio,
                        'noisy': noisy_audio,
                        'enhanced': enhanced_audio,
                        'clean_spec': clean_mag,
                        'noisy_spec': noisy_mag,
                        'enhanced_spec': enhanced_mag,
                        'snr': snr[i].item(),
                        'pesq': pesq_score,
                        'stoi': stoi_score
                    })
    
    results = {
        'mse': total_mse / len(test_loader),
        'pesq_mean': np.mean(pesq_scores),
        'pesq_std': np.std(pesq_scores),
        'stoi_mean': np.mean(stoi_scores),
        'stoi_std': np.std(stoi_scores),
        'pesq_scores': pesq_scores,
        'stoi_scores': stoi_scores,
        'audio_samples': audio_samples
    }
    
    return results

print("✓ Evaluation function defined")


if __name__ == "__main__":
    # Test the functions
    import librosa
    
    # Load a test file
    audio, sr = librosa.load(librosa.ex('trumpet'), sr=16000, duration=4)
    
    # Test PESQ
    pesq_score = compute_pesq(audio, audio * 0.9, sr)
    print(f"PESQ (should work): {pesq_score}")
    
    # Test STOI
    stoi_score = compute_stoi_score(audio, audio * 0.9, sr)
    print(f"STOI (should work): {stoi_score}")
    
    # Test with very short audio
    short_audio = audio[:1000]  # ~0.06 seconds
    pesq_score = compute_pesq(short_audio, short_audio, sr)
    stoi_score = compute_stoi_score(short_audio, short_audio, sr)
    print(f"\nShort audio (should fail gracefully):")
    print(f"  PESQ: {pesq_score}")
    print(f"  STOI: {stoi_score}")


