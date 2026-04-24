""""
Utility functions for model training and evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.data.audio_utils import SAMPLE_RATE

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, train_loader, criterion, optimizer, device, desc="Training"):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=desc)
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        
        # Forward
        enhanced = model(noisy)
        loss = criterion(enhanced, clean)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)


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
