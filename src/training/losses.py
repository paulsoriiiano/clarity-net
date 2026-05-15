"""
Loss functions for training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mag_l1_loss(pred_mag, target_mag):
    """
    L1 loss on magnitude.

    Args:
        pred_mag:   [B, F, T]
        target_mag: [B, F, T]
    """
    if pred_mag.shape != target_mag.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mag.shape}, target {target_mag.shape}")
    if pred_mag.dim() != 3:
        raise ValueError(f"Expected [B, F, T], got {pred_mag.shape}")

    return F.l1_loss(pred_mag, target_mag)


def complex_mse_loss(pred, target):
    """
    MSE on real and imaginary parts.

    Args:
        pred:   [B, 2, F, T]
        target: [B, 2, F, T]
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    if pred.dim() != 4 or pred.size(1) != 2:
        raise ValueError(f"Expected [B, 2, F, T], got {pred.shape}")

    loss_real = F.mse_loss(pred[:, 0], target[:, 0])
    loss_imag = F.mse_loss(pred[:, 1], target[:, 1])
    return loss_real + loss_imag


def complex_l1_loss(pred, target):
    """
    L1 loss on real and imaginary parts.

    Args:
        pred:   [B, 2, F, T]
        target: [B, 2, F, T]
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    if pred.dim() != 4 or pred.size(1) != 2:
        raise ValueError(f"Expected [B, 2, F, T], got {pred.shape}")

    loss_real = F.l1_loss(pred[:, 0], target[:, 0])
    loss_imag = F.l1_loss(pred[:, 1], target[:, 1])
    return loss_real + loss_imag


def fullsubnet_plus_l1_loss(
    enh_real,
    enh_imag,
    clean_real,
    clean_imag,
    enhanced_mag=None,
    clean_mag=None,
    pred_mask=None,
):
    loss = (
        F.l1_loss(enh_real, clean_real) +
        F.l1_loss(enh_imag, clean_imag)
    )

    if enhanced_mag is not None and clean_mag is not None:
        loss = loss + 0.5 * F.l1_loss(enhanced_mag, clean_mag)

    return loss


def get_loss_fn(loss_name):
    if loss_name == "mag_l1":
        return mag_l1_loss
    elif loss_name == "complex_mse":
        return complex_mse_loss
    elif loss_name == "fullsubnet_plus_l1":
        return fullsubnet_plus_l1_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    

class MagnitudeL1Loss(nn.Module):
    """L1 loss on STFT magnitude spectrograms."""
    
    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def forward(self, enhanced, clean):
        """
        Args:
            enhanced: [B, 1, T] waveform
            clean: [B, 1, T] or [B, T] waveform
        """
        # Squeeze channel dimension
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)
        
        # Compute STFT
        window = torch.hann_window(self.win_length, device=enhanced.device)
        
        enhanced_stft = torch.stft(
            enhanced, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        clean_stft = torch.stft(
            clean, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        
        # Magnitude
        enhanced_mag = torch.abs(enhanced_stft)
        clean_mag = torch.abs(clean_stft)
        
        return F.l1_loss(enhanced_mag, clean_mag)


class MagnitudeMSELoss(nn.Module):
    """MSE loss on STFT magnitude spectrograms."""
    
    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def forward(self, enhanced, clean):
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)
        
        window = torch.hann_window(self.win_length, device=enhanced.device)
        
        enhanced_stft = torch.stft(
            enhanced, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        clean_stft = torch.stft(
            clean, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        
        enhanced_mag = torch.abs(enhanced_stft)
        clean_mag = torch.abs(clean_stft)
        
        return F.mse_loss(enhanced_mag, clean_mag)


class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio loss."""
    
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, enhanced, clean):
        """Returns negative SI-SNR (to minimize)."""
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)
        
        # Zero-mean
        enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)
        clean = clean - clean.mean(dim=-1, keepdim=True)
        
        # Projection
        alpha = (enhanced * clean).sum(dim=-1, keepdim=True) / \
                (clean * clean).sum(dim=-1, keepdim=True).clamp(min=self.epsilon)
        projection = alpha * clean
        
        # Noise
        noise = enhanced - projection
        
        # SI-SNR
        si_snr = 10 * torch.log10(
            (projection * projection).sum(dim=-1) / 
            (noise * noise).sum(dim=-1).clamp(min=self.epsilon)
        )
        
        return -si_snr.mean()


class MagnitudeL1PlusSISNRLoss(nn.Module):
    """Combined Magnitude L1 and SI-SNR loss."""
    
    def __init__(self, n_fft=512, hop_length=128, win_length=512, alpha=0.5):
        super().__init__()
        self.mag_l1 = MagnitudeL1Loss(n_fft, hop_length, win_length)
        self.si_snr = SISNRLoss()
        self.alpha = alpha  # Weight for magnitude loss
    
    def forward(self, enhanced, clean):
        mag_loss = self.mag_l1(enhanced, clean)
        snr_loss = self.si_snr(enhanced, clean)
        return self.alpha * mag_loss + (1 - self.alpha) * snr_loss

class WaveformL1Loss(nn.Module):
    """Plain L1 loss directly on waveform samples."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, enhanced, clean):
        """
        Args:
            enhanced: [B, 1, T] or [B, T]
            clean:    [B, 1, T] or [B, T]
        """
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)
        
        return F.l1_loss(enhanced, clean)

class ComplexMSELoss(nn.Module):
    """MSE loss on complex STFT (real + imaginary parts)."""
    
    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def forward(self, enhanced, clean):
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)
        
        window = torch.hann_window(self.win_length, device=enhanced.device)
        
        enhanced_stft = torch.stft(
            enhanced, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        clean_stft = torch.stft(
            clean, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        
        # Separate real and imaginary
        enhanced_real = enhanced_stft.real
        enhanced_imag = enhanced_stft.imag
        clean_real = clean_stft.real
        clean_imag = clean_stft.imag
        
        # MSE on both
        loss_real = F.mse_loss(enhanced_real, clean_real)
        loss_imag = F.mse_loss(enhanced_imag, clean_imag)
        
        return loss_real + loss_imag


def build_loss(cfg):
    """Build loss function from config."""
    loss_name = cfg['loss']['name'].lower()
    model_cfg = cfg['model']
    
    if loss_name == 'mag_l1':
        return MagnitudeL1Loss(
            n_fft=model_cfg['n_fft'],
            hop_length=model_cfg['hop_length'],
            win_length=model_cfg['win_length']
        )
    
    elif loss_name == 'mag_mse':
        return MagnitudeMSELoss(
            n_fft=model_cfg['n_fft'],
            hop_length=model_cfg['hop_length'],
            win_length=model_cfg['win_length']
        )
    
    elif loss_name == 'sisnr':
        return SISNRLoss()
    
    elif loss_name == 'waveform_l1':
        return WaveformL1Loss()
    
    elif loss_name == 'mag_l1_sisnr':
        alpha = cfg['loss'].get('alpha', 0.5)
        return MagnitudeL1PlusSISNRLoss(
            n_fft=model_cfg['n_fft'],
            hop_length=model_cfg['hop_length'],
            win_length=model_cfg['win_length'],
            alpha=alpha
        )
    
    elif loss_name == 'complex_mse':
        return ComplexMSELoss(
            n_fft=model_cfg['n_fft'],
            hop_length=model_cfg['hop_length'],
            win_length=model_cfg['win_length']
        )
    
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

print("✓ Loss functions defined")