"""
FullSubNet+ wrapper for training and evaluation.
"""

import sys
import importlib
from pathlib import Path

import torch
import torch.nn as nn

# Repo root that contains `speech_enhance/`
FULLSUBNET_PLUS_REPO_ROOT = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "fullsubnet_plus"
)

if str(FULLSUBNET_PLUS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(FULLSUBNET_PLUS_REPO_ROOT))

# ------------------------------------------------------------------
# Compatibility shims for inconsistent upstream imports
# ------------------------------------------------------------------
speech_enhance_audio_zen = importlib.import_module("speech_enhance.audio_zen")
speech_enhance_utils = importlib.import_module("speech_enhance.utils")

sys.modules["audio_zen"] = speech_enhance_audio_zen
sys.modules["utils"] = speech_enhance_utils

from speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus


class FullSubNetPlusWrapper(nn.Module):
    """
    Thin wrapper around the official FullSubNet+ implementation.

    Expected batch tensors:
        noisy_mag:  [B, F, T]
        noisy_real: [B, F, T]
        noisy_imag: [B, F, T]

    Returns:
        pred_mask: [B, 2, F, T]
    """

    def __init__(
        self,
        num_freqs,
        look_ahead=2,
        sequence_model="LSTM",
        fb_num_neighbors=0,
        sb_num_neighbors=15,
        fb_output_activate_function="ReLU",
        sb_output_activate_function=False,
        fb_model_hidden_size=512,
        sb_model_hidden_size=384,
        channel_attention_model="SE",
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=1,
        output_size=2,
        subband_num=1,
        kersize=(3, 5, 10),
        weight_init=True,
    ):
        super().__init__()

        self.model = FullSubNet_Plus(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function=fb_output_activate_function,
            sb_output_activate_function=sb_output_activate_function,
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            channel_attention_model=channel_attention_model,
            norm_type=norm_type,
            num_groups_in_drop_band=num_groups_in_drop_band,
            output_size=output_size,
            subband_num=subband_num,
            kersize=list(kersize),
            weight_init=weight_init,
        )

    def forward(self, noisy_mag, noisy_real, noisy_imag):
        # Convert [B, F, T] -> [B, 1, F, T]
        if noisy_mag.dim() == 3:
            noisy_mag = noisy_mag.unsqueeze(1)
        if noisy_real.dim() == 3:
            noisy_real = noisy_real.unsqueeze(1)
        if noisy_imag.dim() == 3:
            noisy_imag = noisy_imag.unsqueeze(1)

        pred_mask = self.model(noisy_mag, noisy_real, noisy_imag)  # [B, 2, F, T]
        return pred_mask

    @staticmethod
    def apply_mask(pred_mask, noisy_real, noisy_imag):
        """
        Apply predicted complex mask to noisy complex STFT.

        Inputs:
            pred_mask:  [B, 2, F, T]
            noisy_real: [B, F, T]
            noisy_imag: [B, F, T]

        Returns:
            enh_real, enh_imag: [B, F, T]
        """
        mask_real = pred_mask[:, 0]
        mask_imag = pred_mask[:, 1]

        enh_real = mask_real * noisy_real - mask_imag * noisy_imag
        enh_imag = mask_real * noisy_imag + mask_imag * noisy_real
        return enh_real, enh_imag
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick test
    B, F, T = 4, 257, 200
    model = FullSubNetPlusWrapper(num_freqs=F)
    noisy_mag = torch.randn(B, F, T)
    noisy_real = torch.randn(B, F, T)
    noisy_imag = torch.randn(B, F, T)

    pred_mask = model(noisy_mag, noisy_real, noisy_imag)
    print(f"Predicted mask shape: {pred_mask.shape}")
    enh_real, enh_imag = model.apply_mask(pred_mask, noisy_real, noisy_imag)
    print(f"Enhanced real shape: {enh_real.shape}, Enhanced imag shape: {enh_imag.shape}")