""""
Utility functions for model training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.audio_utils import SAMPLE_RATE

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device, 
    desc="Training",
    use_mixed_precision=False,
    gradient_accumulation_steps=1
):
    """Train for one epoch with optional memory optimizations.

    Args:
        model: Training model
        train_loader: DataLoader for training
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        desc: Description for progress bar
        use_mixed_precision: Enable mixed precision (float16) training
        gradient_accumulation_steps: Number of steps to accumulate gradients

    Supports:
    - waveform models: batch["noisy"], batch["clean"] -> [B, T]
    - magnitude spectrogram models: batch["noisy"], batch["clean"] -> [B, 1, F, T]
    - complex spectrogram models: batch["noisy"], batch["clean"] -> [B, 2, F, T]
    - FullSubNet+ style batches:
        noisy_mag, noisy_real, noisy_imag,
        clean_mag, clean_real, clean_imag
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    scaler = GradScaler() if use_mixed_precision else None
    optimizer.zero_grad()  # Clear any leftover gradients

    pbar = tqdm(train_loader, desc=desc)
    for batch_idx, batch in enumerate(pbar):
        try:
            # Determine device type for autocast
            device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
            
            # Forward pass with optional mixed precision
            if use_mixed_precision:
                with autocast(device_type=device_type):
                    loss = _compute_loss(
                        model, batch, criterion, device,
                        is_fullsubnet_plus=all(
                            k in batch for k in [
                                "noisy_mag", "noisy_real", "noisy_imag",
                                "clean_mag", "clean_real", "clean_imag"
                            ]
                        )
                    )
                loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss = _compute_loss(
                    model, batch, criterion, device,
                    is_fullsubnet_plus=all(
                        k in batch for k in [
                            "noisy_mag", "noisy_real", "noisy_imag",
                            "clean_mag", "clean_real", "clean_imag"
                        ]
                    )
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()

            # Check for NaN
            if torch.isnan(loss):
                print(f"\n⚠️  NaN loss detected at batch {batch_idx}")
                raise RuntimeError(f"NaN loss at batch {batch_idx}")

            # Update weights after accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_mixed_precision:
                    # Unscale gradients before clipping to avoid overflow
                    scaler.unscale_(optimizer)
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            total_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.6f}"})

            # Clear cache every batch to prevent memory fragmentation
            if device_type == 'cuda':
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ OOM on batch {batch_idx}/{len(train_loader)}")
                print(f"   Suggestions:")
                print(f"   1. Reduce BATCH_SIZE (currently: 2)")
                print(f"   2. Increase GRADIENT_ACCUMULATION_STEPS")
                print(f"   3. Skip PESQ/STOI computation in validation")
                print(f"   4. Reduce augmentation_factor in dataset")
                if torch.cuda.is_available():
                    print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                raise
            else:
                raise

    return total_loss / max(total_batches, 1)


def _compute_loss(model, batch, criterion, device, is_fullsubnet_plus):
    """Compute loss for a batch (extracted to support autocast context)."""
    if is_fullsubnet_plus:
        noisy_mag = batch["noisy_mag"].to(device)     # [B, F, T]
        noisy_real = batch["noisy_real"].to(device)   # [B, F, T]
        noisy_imag = batch["noisy_imag"].to(device)   # [B, F, T]

        clean_mag = batch["clean_mag"].to(device)     # [B, F, T]
        clean_real = batch["clean_real"].to(device)   # [B, F, T]
        clean_imag = batch["clean_imag"].to(device)   # [B, F, T]

        # Wrapper returns pred_mask: [B, 2, F, T]
        pred_mask = model(noisy_mag, noisy_real, noisy_imag)

        # Convert mask -> enhanced complex STFT
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
    else:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        is_waveform = (noisy.dim() == 2)                          # [B, T]
        is_complex = (noisy.dim() == 4 and noisy.shape[1] == 2)   # [B, 2, F, T]
        is_magnitude = (noisy.dim() == 4 and noisy.shape[1] == 1) # [B, 1, F, T]

        enhanced = model(noisy)

        if is_waveform and enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)

        loss = criterion(enhanced, clean)
    
    return loss

def validate(model, val_loader, criterion, device, desc="Validation"):
    """Validate on a dataset."""
    model.eval()
    total_loss = 0
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc=desc, leave=False):
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            
            enhanced = model(noisy)
            loss = criterion(enhanced, clean)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_checkpoint(model, history, epoch, optimizer, scheduler, best_val_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history
    }

    if is_best:
        path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, path)
        print(f"  ✓ Saved best model to {path}")
    else:
        path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        print(f"  ✓ Saved checkpoint to {path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    history = checkpoint.get('history', {})
    
    return model, history, epoch
