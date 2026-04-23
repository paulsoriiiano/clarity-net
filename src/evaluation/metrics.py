import numpy as np
from pesq import pesq
from pystoi import stoi

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.audio_utils import SAMPLE_RATE
from src.data.audio_utils import reconstruct_audio


def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int = SAMPLE_RATE, mode: str = 'wb') -> float:
    """
    Compute PESQ score.
    Returns: score between -0.5 and 4.5 (higher is better)
    """
    try:
        # PESQ requires same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        # PESQ mode: 'wb' for wideband (16kHz), 'nb' for narrowband (8kHz)
        score = pesq(sr, clean, enhanced, mode)
        return score
    except Exception as e:
        print(f"PESQ error: {e}")
        return 0.0

def compute_stoi_score(clean: np.ndarray, enhanced: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Compute STOI score.
    Returns: score between 0 and 1 (higher is better)
    """
    try:
        # STOI requires same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        score = stoi(clean, enhanced, sr, extended=False)
        return score
    except Exception as e:
        print(f"STOI error: {e}")
        return 0.0
    
def evaluate_with_metrics(model, test_loader, desc="Evaluation", device='cpu'):
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