"""
Size-preserving U-Net CNN for speech enhancement.
Guarantees output size exactly matches input size.
 
Replace the UNetCNN class in notebook 03 with this version.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class UNetCNN(nn.Module):
    """
    U-Net baseline with guaranteed size preservation.
    Uses interpolate for upsampling to ensure exact size matching.
    """
    
    def __init__(self, in_channels=1, base_channels=32):
        super(UNetCNN, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.dec4 = self._conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def _conv_block(self, in_c, out_c):
        """Convolutional block with padding to preserve size."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Store input size for upsampling
        input_size = x.shape[2:]  # (H, W)
        
        # Encoder with size tracking
        enc1 = self.enc1(x)  # Same size as input
        enc1_pooled = F.max_pool2d(enc1, kernel_size=2, stride=2)
        enc1_size = enc1.shape[2:]
        
        enc2 = self.enc2(enc1_pooled)
        enc2_pooled = F.max_pool2d(enc2, kernel_size=2, stride=2)
        enc2_size = enc2.shape[2:]
        
        enc3 = self.enc3(enc2_pooled)
        enc3_pooled = F.max_pool2d(enc3, kernel_size=2, stride=2)
        enc3_size = enc3.shape[2:]
        
        enc4 = self.enc4(enc3_pooled)
        enc4_pooled = F.max_pool2d(enc4, kernel_size=2, stride=2)
        enc4_size = enc4.shape[2:]
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4_pooled)
        
        # Decoder with exact size matching via interpolate
        dec4 = F.interpolate(bottleneck, size=enc4_size, mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = F.interpolate(dec4, size=enc3_size, mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = F.interpolate(dec3, size=enc2_size, mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = F.interpolate(dec2, size=enc1_size, mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final upsampling to exactly match input size
        dec1 = F.interpolate(dec1, size=input_size, mode='bilinear', align_corners=False)
        
        # Output mask
        mask = self.sigmoid(self.out_conv(dec1))
        
        # Apply mask to input
        enhanced = x * mask
        
        return enhanced
    

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    # Create model
    model = UNetCNN(in_channels=1, base_channels=32)
    print(f"Model created with {count_parameters(model):,} parameters")
        
    # Test multiple sizes to ensure it works
    test_sizes = [
        (2, 1, 257, 501),  # Actual STFT size from data
        (2, 1, 257, 251),  # Alternative size
        (2, 1, 256, 500),  # Even dimensions
]

    print("Model Architecture Test")
    print(f"Total parameters: {count_parameters(model):,}\n")
        
    for test_size in test_sizes:
        x = torch.randn(*test_size)
            
        with torch.no_grad():
            output = model(x).cpu()
            
        print(f"Input:  {tuple(x.shape)} -> Output: {tuple(output.shape)}")
        assert x.shape == output.shape, f"Size mismatch! {x.shape} != {output.shape}"
        
        print("\n✓ All tests passed! Output size matches input size exactly.")
