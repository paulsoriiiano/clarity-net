"""
Visualization utilities for training and evaluation results.
"""

import json

import json

from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
from src.data.audio_utils import SAMPLE_RATE

def plot_loss_curves(history, results_dir):
    """Plot training/validation loss, PESQ, STOI, SI-SNR, and generalization gaps."""
    # # Load history
    # with open(results_dir / 'training_history.json') as f:
    #     history = json.load(f)

    epochs_range = range(1, len(history['train_loss']) + 1)
    has_si_snr = all(
        key in history for key in ['val_seen_si_snr', 'val_unseen_si_snr']
    )

    if 'generalization_gap_si_snr' in history:
        si_snr_gap = history['generalization_gap_si_snr']
    elif has_si_snr:
        si_snr_gap = [
            seen - unseen
            for seen, unseen in zip(history['val_seen_si_snr'], history['val_unseen_si_snr'])
        ]
    else:
        si_snr_gap = None

    # Create subplot layout: 4 metric panels + one wide generalization-gap panel
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_pesq = fig.add_subplot(gs[0, 1])
    ax_stoi = fig.add_subplot(gs[1, 0])
    ax_si_snr = fig.add_subplot(gs[1, 1])
    ax_gap = fig.add_subplot(gs[2, :])

    # ============================================================================
    # Plot 1: Loss curves
    # ============================================================================
    ax_loss.plot(epochs_range, history['train_loss'],
                 label='Train', linewidth=2, color='blue')
    ax_loss.plot(epochs_range, history['val_seen_loss'],
                 label='Val (Seen)', linewidth=2, color='orange')
    ax_loss.plot(epochs_range, history['val_unseen_loss'],
                 label='Val (Unseen)', linewidth=2, color='green')
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss (MSE)', fontsize=12)
    ax_loss.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    # ============================================================================
    # Plot 2: PESQ scores
    # ============================================================================
    ax_pesq.plot(epochs_range, history['val_seen_pesq'],
                 label='Seen', linewidth=2, color='blue')
    ax_pesq.plot(epochs_range, history['val_unseen_pesq'],
                 label='Unseen', linewidth=2, color='red')
    ax_pesq.set_xlabel('Epoch', fontsize=12)
    ax_pesq.set_ylabel('PESQ Score', fontsize=12)
    ax_pesq.set_title('PESQ (Perceptual Quality)', fontsize=14, fontweight='bold')
    ax_pesq.legend(fontsize=10)
    ax_pesq.grid(True, alpha=0.3)
    ax_pesq.set_ylim([0, 4.5])  # PESQ range is -0.5 to 4.5

    # ============================================================================
    # Plot 3: STOI scores
    # ============================================================================
    ax_stoi.plot(epochs_range, history['val_seen_stoi'],
                 label='Seen', linewidth=2, color='blue')
    ax_stoi.plot(epochs_range, history['val_unseen_stoi'],
                 label='Unseen', linewidth=2, color='red')
    ax_stoi.set_xlabel('Epoch', fontsize=12)
    ax_stoi.set_ylabel('STOI Score', fontsize=12)
    ax_stoi.set_title('STOI (Intelligibility)', fontsize=14, fontweight='bold')
    ax_stoi.legend(fontsize=10)
    ax_stoi.grid(True, alpha=0.3)
    ax_stoi.set_ylim([0, 1])  # STOI range is 0 to 1

    # ============================================================================
    # Plot 4: SI-SNR scores
    # ============================================================================
    if has_si_snr:
        ax_si_snr.plot(epochs_range, history['val_seen_si_snr'],
                       label='Seen', linewidth=2, color='blue')
        ax_si_snr.plot(epochs_range, history['val_unseen_si_snr'],
                       label='Unseen', linewidth=2, color='red')
        ax_si_snr.set_xlabel('Epoch', fontsize=12)
        ax_si_snr.set_ylabel('SI-SNR (dB)', fontsize=12)
        ax_si_snr.set_title('SI-SNR (Scale-Invariant SNR)', fontsize=14, fontweight='bold')
        ax_si_snr.legend(fontsize=10)
        ax_si_snr.grid(True, alpha=0.3)
    else:
        ax_si_snr.axis('off')
        ax_si_snr.text(
            0.5, 0.5,
            'SI-SNR history not available\nExpected keys:\nval_seen_si_snr\nval_unseen_si_snr',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    # ============================================================================
    # Plot 5: Generalization gaps
    # ============================================================================
    ax_gap.plot(epochs_range, history['generalization_gap_pesq'],
                label='PESQ Gap', linewidth=2, color='purple')
    ax_gap.plot(epochs_range, history['generalization_gap_stoi'],
                label='STOI Gap', linewidth=2, color='orange')
    if si_snr_gap is not None:
        ax_gap.plot(epochs_range, si_snr_gap,
                    label='SI-SNR Gap', linewidth=2, color='teal')
    ax_gap.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax_gap.set_xlabel('Epoch', fontsize=12)
    ax_gap.set_ylabel('Gap (Seen - Unseen)', fontsize=12)
    ax_gap.set_title('Generalization Gap Over Training', fontsize=14, fontweight='bold')
    ax_gap.legend(fontsize=10)
    ax_gap.grid(True, alpha=0.3)

    # Add note about gap direction
    ax_gap.text(0.02, 0.98,
                'Positive gap = worse on unseen\nNegative gap = better on unseen',
                transform=ax_gap.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves_with_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Saved training curves to {results_dir / 'training_curves_with_metrics.png'}")

    # ============================================================================
    # Print summary statistics
    # ============================================================================
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)

    # Find best epoch
    best_epoch = history['val_seen_loss'].index(min(history['val_seen_loss'])) + 1
    print(f"\nBest epoch: {best_epoch}")
    print(f"  Loss (seen):   {history['val_seen_loss'][best_epoch-1]:.6f}")
    print(f"  Loss (unseen): {history['val_unseen_loss'][best_epoch-1]:.6f}")
    print(f"  PESQ (seen):   {history['val_seen_pesq'][best_epoch-1]:.3f}")
    print(f"  PESQ (unseen): {history['val_unseen_pesq'][best_epoch-1]:.3f}")
    print(f"  STOI (seen):   {history['val_seen_stoi'][best_epoch-1]:.3f}")
    print(f"  STOI (unseen): {history['val_unseen_stoi'][best_epoch-1]:.3f}")
    if has_si_snr:
        print(f"  SI-SNR (seen): {history['val_seen_si_snr'][best_epoch-1]:.3f} dB")
        print(f"  SI-SNR (unseen): {history['val_unseen_si_snr'][best_epoch-1]:.3f} dB")

    # Final epoch
    final_epoch = len(history['train_loss'])
    print(f"\nFinal epoch: {final_epoch}")
    print(f"  Loss (seen):   {history['val_seen_loss'][-1]:.6f}")
    print(f"  Loss (unseen): {history['val_unseen_loss'][-1]:.6f}")
    print(f"  PESQ (seen):   {history['val_seen_pesq'][-1]:.3f}")
    print(f"  PESQ (unseen): {history['val_unseen_pesq'][-1]:.3f}")
    print(f"  STOI (seen):   {history['val_seen_stoi'][-1]:.3f}")
    print(f"  STOI (unseen): {history['val_unseen_stoi'][-1]:.3f}")
    if has_si_snr:
        print(f"  SI-SNR (seen): {history['val_seen_si_snr'][-1]:.3f} dB")
        print(f"  SI-SNR (unseen): {history['val_unseen_si_snr'][-1]:.3f} dB")

    # Generalization gaps
    print(f"\nGeneralization Gaps (at best epoch):")
    print(f"  PESQ: {history['generalization_gap_pesq'][best_epoch-1]:+.3f}")
    print(f"  STOI: {history['generalization_gap_stoi'][best_epoch-1]:+.3f}")
    if si_snr_gap is not None:
        print(f"  SI-SNR: {si_snr_gap[best_epoch-1]:+.3f} dB")

    print(f"\nGeneralization Gaps (at final epoch):")
    print(f"  PESQ: {history['generalization_gap_pesq'][-1]:+.3f}")
    print(f"  STOI: {history['generalization_gap_stoi'][-1]:+.3f}")
    if si_snr_gap is not None:
        print(f"  SI-SNR: {si_snr_gap[-1]:+.3f} dB")

    print("="*70)

def visualize_audio_samples(results_seen, results_dir, sample_idx=0):
    """Visualize spectrograms of audio samples."""
    # Pick a good sample (high PESQ improvement)
    sample_idx = min(sample_idx, len(results_seen['audio_samples']) - 1)
    sample = results_seen['audio_samples'][sample_idx]

    print(f"Sample {sample_idx + 1} - SNR: {sample['snr']:.1f} dB")
    print(f"  PESQ: {sample['pesq']:.3f}")
    print(f"  STOI: {sample['stoi']:.3f}")
    print("\n" + "="*60)

    # Create time axis for waveforms
    time_axis = np.arange(len(sample['clean'])) / SAMPLE_RATE

    # Plot waveforms
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(time_axis, sample['clean'], linewidth=0.5, color='blue')
    axes[0].set_title('Clean Speech (Ground Truth)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-1, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, sample['noisy'], linewidth=0.5, color='red')
    axes[1].set_title(f'Noisy Speech (SNR = {sample["snr"]:.1f} dB)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-1, 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, sample['enhanced'], linewidth=0.5, color='green')
    axes[2].set_title(f'Enhanced Speech (PESQ: {sample["pesq"]:.2f}, STOI: {sample["stoi"]:.2f})', 
                    fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_ylim(-1, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / f'waveform_comparison_sample_{sample_idx}.png', dpi=150)
    plt.show()

    print(f"Waveform saved to {results_dir / f'waveform_comparison_sample_{sample_idx}.png'}")


def visualize_audio_overlay(results_seen, results_dir, sample_idx=0):
    """Visualize overlay of clean vs enhanced and noisy vs enhanced."""
    sample_idx = min(sample_idx, len(results_seen['audio_samples']) - 1)
    sample = results_seen['audio_samples'][sample_idx]

    # Create overlay plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create time axis for waveforms
    time_axis = np.arange(len(sample['clean'])) / SAMPLE_RATE

    # Clean vs Enhanced
    axes[0].plot(time_axis, sample['clean'], linewidth=1, color='blue', label='Clean')
    axes[0].plot(time_axis, sample['enhanced'], linewidth=1, color='green', alpha=0.7, label='Enhanced')
    axes[0].set_title('Clean vs Enhanced Speech', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-1, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Noisy vs Enhanced
    axes[1].plot(time_axis, sample['noisy'], linewidth=1, color='red', label='Noisy')
    axes[1].plot(time_axis, sample['enhanced'], linewidth=1, color='green', alpha=0.7, label='Enhanced')
    axes[1].set_title('Noisy vs Enhanced Speech', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-1, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'overlay_comparison.png', dpi=150)
    plt.show()

    print(f"Overlay plots saved to {results_dir / 'overlay_comparison.png'}")


def playback_audio(results_seen, sample_idx=0):
    """Utility to play back audio samples in Jupyter."""
    # Debug audio data before playback
    sample_idx = min(sample_idx, len(results_seen['audio_samples']) - 1)
    sample = results_seen['audio_samples'][sample_idx]

    print("Audio Data Inspection:")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Clean audio shape: {sample['clean'].shape}")
    print(f"Clean audio dtype: {sample['clean'].dtype}")
    print(f"Clean audio range: [{sample['clean'].min():.6f}, {sample['clean'].max():.6f}]")
    print(f"Clean audio RMS: {np.sqrt(np.mean(sample['clean']**2)):.6f}")
    print(f"Clean audio duration: {len(sample['clean']) / SAMPLE_RATE:.2f} seconds")

    print(f"\nNoisy audio shape: {sample['noisy'].shape}")
    print(f"Noisy audio range: [{sample['noisy'].min():.6f}, {sample['noisy'].max():.6f}]")
    print(f"Noisy audio RMS: {np.sqrt(np.mean(sample['noisy']**2)):.6f}")

    print(f"\nEnhanced audio shape: {sample['enhanced'].shape}")
    print(f"Enhanced audio range: [{sample['enhanced'].min():.6f}, {sample['enhanced'].max():.6f}]")
    print(f"Enhanced audio RMS: {np.sqrt(np.mean(sample['enhanced']**2)):.6f}")

    # Check if audio is essentially silent
    threshold = 1e-6
    if np.max(np.abs(sample['clean'])) < threshold:
        print("\n⚠️  WARNING: Clean audio appears to be silent!")
    if np.max(np.abs(sample['noisy'])) < threshold:
        print("\n⚠️  WARNING: Noisy audio appears to be silent!")
    if np.max(np.abs(sample['enhanced'])) < threshold:
        print("\n⚠️  WARNING: Enhanced audio appears to be silent!")

    print("\n" + "="*60)

    print("Clean Speech:")
    ipd.display(ipd.Audio(sample['clean'], rate=SAMPLE_RATE))

    print(f"\nNoisy Speech (SNR = {sample['snr']:.1f} dB):")
    ipd.display(ipd.Audio(sample['noisy'], rate=SAMPLE_RATE))

    print(f"\nEnhanced Speech (PESQ = {sample['pesq']:.2f}, STOI = {sample['stoi']:.2f}):")
    ipd.display(ipd.Audio(sample['enhanced'], rate=SAMPLE_RATE))


def plot_metric_distributions(results_seen, results_unseen, results_dir):
    """Plot distributions of PESQ, STOI, and SI-SNR scores."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # PESQ distribution
    axes[0].hist(results_seen['pesq_scores'], bins=20, alpha=0.7, label='Seen', color='blue')
    axes[0].hist(results_unseen['pesq_scores'], bins=20, alpha=0.7, label='Unseen', color='red')
    axes[0].axvline(results_seen['pesq_mean'], color='blue', linestyle='--', linewidth=2)
    axes[0].axvline(results_unseen['pesq_mean'], color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('PESQ Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PESQ Score Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # STOI distribution
    axes[1].hist(results_seen['stoi_scores'], bins=20, alpha=0.7, label='Seen', color='blue')
    axes[1].hist(results_unseen['stoi_scores'], bins=20, alpha=0.7, label='Unseen', color='red')
    axes[1].axvline(results_seen['stoi_mean'], color='blue', linestyle='--', linewidth=2)
    axes[1].axvline(results_unseen['stoi_mean'], color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('STOI Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('STOI Score Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SI-SNR distribution
    si_snr_seen = results_seen.get('si_snr_scores', [])
    si_snr_unseen = results_unseen.get('si_snr_scores', [])

    if si_snr_seen or si_snr_unseen:
        if si_snr_seen:
            axes[2].hist(si_snr_seen, bins=20, alpha=0.7, label='Seen', color='blue')
        if si_snr_unseen:
            axes[2].hist(si_snr_unseen, bins=20, alpha=0.7, label='Unseen', color='red')
        if 'si_snr_mean' in results_seen:
            axes[2].axvline(results_seen['si_snr_mean'], color='blue', linestyle='--', linewidth=2)
        if 'si_snr_mean' in results_unseen:
            axes[2].axvline(results_unseen['si_snr_mean'], color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('SI-SNR (dB)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('SI-SNR Distribution', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis('off')
        axes[2].text(
            0.5, 0.5,
            'SI-SNR scores not available\nExpected key: si_snr_scores',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    plt.tight_layout()
    plt.savefig(results_dir / 'metric_distributions.png', dpi=150)
    plt.show()

    print(f"Metric distributions saved to {results_dir / 'metric_distributions.png'}")


def plot_metric_distribution(results_seen, results_unseen, results_dir):
    """Backward-compatible alias for plot_metric_distributions()."""
    return plot_metric_distributions(results_seen, results_unseen, results_dir)
