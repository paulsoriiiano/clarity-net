"""
Gradio UI for ClarityNet Speech Enhancement Model Demo

This script creates an interactive web interface for the FullSubNet speech enhancement model.
Users can either record live audio through a microphone or upload an audio file,
and the model will enhance it and display waveforms for comparison.
"""

import os
import sys
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fullsubnet import FullSubNet
from src.data.audio_utils import SAMPLE_RATE, N_FFT, HOP_LENGTH


class SpeechEnhancementDemo:
    """Wrapper class for speech enhancement inference."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the speech enhancement model.
        
        Args:
            model_path: Path to the model checkpoint (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
        """
        self.device = torch.device(device)
        self.model = FullSubNet().to(self.device)
        self.model.eval()
        
        # Load checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✓ Model loaded from: {model_path}")
    
    def preprocess_audio(self, audio: np.ndarray) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio for model inference.
        
        Args:
            audio: Raw audio waveform (mono, 16kHz)
        
        Returns:
            Tuple of (audio_tensor, original_length)
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        original_length = len(audio)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        audio_tensor = audio_tensor.unsqueeze(0)  # [1, T]
        
        return audio_tensor, original_length
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance audio using the speech enhancement model.
        
        Args:
            audio: Raw audio waveform (mono, 16kHz)
        
        Returns:
            Enhanced audio waveform
        """
        audio_tensor, original_length = self.preprocess_audio(audio)
        
        with torch.no_grad():
            enhanced = self.model(audio_tensor)  # [1, 1, T]
        
        # Convert back to numpy
        enhanced = enhanced.squeeze().cpu().numpy()
        
        # Trim to original length
        enhanced = enhanced[:original_length]
        
        # Normalize
        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced / max_val
        
        return enhanced


def create_waveform_plot(
    audio_noisy: np.ndarray,
    audio_enhanced: np.ndarray,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Create a visualization comparing input and output waveforms.
    
    Args:
        audio_noisy: Input (noisy) audio waveform
        audio_enhanced: Output (enhanced) audio waveform
        sr: Sample rate
    
    Returns:
        Plot as numpy array
    """
    import io
    from PIL import Image
    
    time_noisy = np.arange(len(audio_noisy)) / sr
    time_enhanced = np.arange(len(audio_enhanced)) / sr
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    # Input waveform
    axes[0].plot(time_noisy, audio_noisy, linewidth=0.5, color='#FF6B6B')
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Input (Noisy) Waveform', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, max(time_noisy[-1], time_enhanced[-1]))
    
    # Enhanced waveform
    axes[1].plot(time_enhanced, audio_enhanced, linewidth=0.5, color='#4ECDC4')
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_title('Output (Enhanced) Waveform', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, max(time_noisy[-1], time_enhanced[-1]))
    
    plt.tight_layout()
    
    # Convert plot to numpy array (matplotlib version compatible)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    plot_array = np.array(image)
    plt.close(fig)
    
    return plot_array


def enhance_speech(
    audio_input: Tuple[int, np.ndarray],
    model_demo: SpeechEnhancementDemo
) -> Tuple[Tuple[int, np.ndarray], Tuple[int, np.ndarray], np.ndarray]:
    """
    Main enhancement function called by Gradio interface.
    
    Args:
        audio_input: Tuple of (sample_rate, audio_data) from Gradio
        model_demo: SpeechEnhancementDemo instance
    
    Returns:
        Tuple of (enhanced_audio, original_audio, waveform_plot)
    """
    if audio_input is None:
        return None, None, None
    
    sr, audio = audio_input
    
    # Resample if necessary
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    else:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / (2 ** 15)  # If int16
    
    # Enhance audio
    enhanced = model_demo.enhance_audio(audio)
    
    # Create waveform plot
    plot = create_waveform_plot(audio, enhanced, sr=SAMPLE_RATE)
    
    return (SAMPLE_RATE, enhanced), (SAMPLE_RATE, audio), plot


def create_gradio_interface(model_path: str) -> gr.Blocks:
    """
    Create the Gradio interface for speech enhancement demo.
    
    Args:
        model_path: Path to the model checkpoint
    
    Returns:
        Gradio Blocks interface
    """
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_demo = SpeechEnhancementDemo(model_path, device=device)
    
    # Create interface
    with gr.Blocks(title="ClarityNet - Speech Enhancement Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎙️ ClarityNet: Speech Enhancement Demo
            
            Upload an audio file or record live audio through your microphone.
            The model will enhance the audio by reducing background noise.
            
            ### How it works:
            1. **Input**: Upload an audio file (.wav, .mp3, etc.) or record via microphone
            2. **Processing**: FullSubNet model enhances the audio
            3. **Output**: Listen to the enhanced audio and compare waveforms
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Audio")
                
                # Audio input options
                audio_input = gr.Audio(
                    label="Upload or Record Audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Processing")
                
                # Process button
                process_btn = gr.Button(
                    "🚀 Enhance Audio",
                    variant="primary",
                    size="lg"
                )
        
        # Output section
        gr.Markdown("### Results")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Input Audio")
                audio_noisy_output = gr.Audio(
                    label="Original (Noisy) Audio",
                    type="numpy",
                    interactive=False,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### Enhanced Audio")
                audio_enhanced_output = gr.Audio(
                    label="Enhanced Audio",
                    type="numpy",
                    interactive=False,
                )
        
        # Waveform comparison
        gr.Markdown("### Waveform Comparison")
        waveform_plot = gr.Image(
            label="Input vs Output Waveforms",
            type="numpy"
        )
        
        # Set up button click event
        process_btn.click(
            fn=lambda audio: enhance_speech(audio, model_demo),
            inputs=[audio_input],
            outputs=[audio_enhanced_output, audio_noisy_output, waveform_plot],
        )
        
        # Info section
        gr.Markdown(
            """
            ---
            **About ClarityNet:**
            - Model: FullSubNet (Full-band and Sub-band Processing Network)
            - Sample Rate: 16 kHz (mono audio)
            - Input: Raw audio waveform
            - Output: Noise-reduced speech
            
            **Tips:**
            - For best results, use audio samples with clear speech
            - The model works best with SNR (Signal-to-Noise Ratio) between 0-20 dB
            - Audio will be automatically resampled if needed
            """
        )
    
    return demo


def main():
    """Main entry point for the Gradio demo."""
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    
    # Default model path (can be overridden)
    # DEFAULT_MODEL = PROJECT_ROOT / "checkpoints" / "fullsubnet_final" / "best_model.pth"
    DEFAULT_MODEL = PROJECT_ROOT / "results" / "final_fullsubnet" / "final_fullsubnet_sisnr" / "checkpoints" / "best_unseen_pesq.pth"
    model_path = os.environ.get("CLARITY_NET_MODEL", str(DEFAULT_MODEL))
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print(f"Model path: {model_path}")
    
    # Create and launch interface
    demo = create_gradio_interface(model_path)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
