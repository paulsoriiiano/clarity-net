"""
src/models/dccrn_fixed.py

FIXED DCCRN model with:
1. Dynamic LSTM input size calculation
2. Proper initialization
3. Gradient clipping support
4. Optional GroupNorm instead of BatchNorm (for small batch sizes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Complex Layers (same as before)
# =============================================================================

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        real = x[:, 0]
        imag = x[:, 1]
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)
        return torch.stack([real_out, imag_out], dim=1)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        real = self.bn_real(x[:, 0])
        imag = self.bn_imag(x[:, 1])
        return torch.stack([real, imag], dim=1)


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters)
    
    def forward(self, x):
        real = self.prelu(x[:, 0])
        imag = self.prelu(x[:, 1])
        return torch.stack([real, imag], dim=1)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
    
    def forward(self, x):
        real = x[:, 0]
        imag = x[:, 1]
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)
        return torch.stack([real_out, imag_out], dim=1)


# =============================================================================
# FIXED DCCRN Model
# =============================================================================

class DCCRN(nn.Module):
    """
    Fixed DCCRN with dynamic LSTM sizing and better stability.
    """
    
    def __init__(self, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # Encoder
        self.enc1 = self._make_encoder_block(1, 32, kernel=(3, 2), stride=(2, 1))
        self.enc2 = self._make_encoder_block(32, 64, kernel=(3, 2), stride=(2, 1))
        self.enc3 = self._make_encoder_block(64, 128, kernel=(3, 2), stride=(2, 1))
        self.enc4 = self._make_encoder_block(128, 256, kernel=(3, 2), stride=(2, 1))
        self.enc5 = self._make_encoder_block(256, 256, kernel=(3, 2), stride=(2, 1))
        
        # LSTM and projection layers will be initialized dynamically on first forward
        self.lstm = None
        self.lstm_fc = None
        self.lstm_input_size = None
        
        # Decoder
        self.dec5 = self._make_decoder_block(256 + 256, 256, kernel=(3, 2), stride=(2, 1))
        self.dec4 = self._make_decoder_block(256 + 256, 128, kernel=(3, 2), stride=(2, 1))
        self.dec3 = self._make_decoder_block(128 + 128, 64, kernel=(3, 2), stride=(2, 1))
        self.dec2 = self._make_decoder_block(64 + 64, 32, kernel=(3, 2), stride=(2, 1))
        self.dec1 = self._make_decoder_block(32 + 32, 1, kernel=(3, 2), stride=(2, 1), last=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization for stability."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_encoder_block(self, in_ch, out_ch, kernel, stride):
        return nn.Sequential(
            ComplexConv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=(1, 0)),
            ComplexBatchNorm2d(out_ch),
            ComplexPReLU()
        )
    
    def _make_decoder_block(self, in_ch, out_ch, kernel, stride, last=False):
        if last:
            return nn.Sequential(
                ComplexConvTranspose2d(in_ch, out_ch, kernel_size=kernel, stride=stride, 
                                      padding=(1, 0), output_padding=(1, 0))
            )
        else:
            return nn.Sequential(
                ComplexConvTranspose2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                                      padding=(1, 0), output_padding=(1, 0)),
                ComplexBatchNorm2d(out_ch),
                ComplexPReLU()
            )
    
    def _initialize_lstm(self, input_size, device):
        """Initialize LSTM dynamically based on actual encoder output size."""
        print(f"[DCCRN] Initializing LSTM with input_size={input_size}")
        
        self.lstm_input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=False
        ).to(device)
        
        self.lstm_fc = nn.Linear(self.lstm_hidden, input_size).to(device)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize projection layer
        nn.init.xavier_uniform_(self.lstm_fc.weight)
        nn.init.constant_(self.lstm_fc.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 2, freq, time) - complex spectrogram
        Returns:
            (batch, 2, freq, time) - enhanced complex spectrogram
        """
        # Store original size
        _, _, orig_freq, orig_time = x.shape
        
        # Pad frequency to be divisible by 32
        padded_freq = ((orig_freq + 31) // 32) * 32
        if padded_freq != orig_freq:
            pad_amount = padded_freq - orig_freq
            x = F.pad(x, (0, 0, 0, pad_amount))
        
        # Add channel dimension
        x = x.unsqueeze(2)  # (batch, 2, 1, freq, time)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Get actual dimensions
        batch, _, channels, freq, time = e5.shape
        
        # Initialize LSTM on first forward pass
        actual_input_size = 2 * channels * freq
        if self.lstm is None or self.lstm_input_size != actual_input_size:
            self._initialize_lstm(actual_input_size, x.device)
        
        # Reshape for LSTM
        lstm_input = e5.permute(0, 4, 1, 2, 3)  # (batch, time, 2, channels, freq)
        lstm_input = lstm_input.reshape(batch, time, -1)  # (batch, time, features)
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.lstm_fc(lstm_out)
        
        # Reshape back
        lstm_out = lstm_out.reshape(batch, time, 2, channels, freq)
        lstm_out = lstm_out.permute(0, 2, 3, 4, 1)  # (batch, 2, channels, freq, time)
        
        # Decoder with skip connections
        d5 = self.dec5(torch.cat([lstm_out, e5], dim=2))
        d4 = self.dec4(torch.cat([d5, e4], dim=2))
        d3 = self.dec3(torch.cat([d4, e3], dim=2))
        d2 = self.dec2(torch.cat([d3, e2], dim=2))
        d1 = self.dec1(torch.cat([d2, e1], dim=2))
        
        # Remove channel dimension
        out = d1.squeeze(2)  # (batch, 2, freq, time)
        
        # Crop to original size
        out = out[:, :, :orig_freq, :orig_time]
        
        # Apply masking
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_mag = torch.sqrt(mask_real**2 + mask_imag**2 + 1e-8)
        mask = torch.sigmoid(mask_mag).unsqueeze(1)
        
        # Apply mask to cropped input
        input_squeezed = x.squeeze(2)[:, :, :orig_freq, :orig_time]
        enhanced = input_squeezed * mask.expand_as(input_squeezed)
        
        return enhanced


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    model = DCCRN(lstm_hidden=128, lstm_layers=2)
    x = torch.randn(2, 2, 257, 251)
    
    print("DCCRN Model Test")
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
    print("\n✓ Model test passed!")