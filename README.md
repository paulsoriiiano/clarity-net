# ClarityNet: Cross-Domain Generalization in Speech Enhancement

**Robust Noise Suppression Across Unseen Acoustic Environments**

A deep learning project investigating how well speech enhancement models generalize to noise types they've never encountered during training.

---

**Status:** 🚧 
- Generated dataset
- Working on preprocessing

**Next Steps:**
- Validate pipeline with a vanilla CNN model
- Perform initial training and evaluation for both models to compare baseline performance
- Perform ablation studies for selected model

---

## 📋 Overview

Most speech enhancement systems are trained and tested on the same noise distribution, leaving their real-world robustness unexplored. This project systematically measures the **generalization gap** between seen and unseen noise conditions and evaluates training strategies to close it.

**Research Question:** *How well does a speech enhancement model trained on specific noise types generalize to completely unseen noise conditions, and what training strategies can improve this generalization?*

### Key Contributions

- **Explicit generalization evaluation:** Noise types partitioned into seen (training) vs. unseen (testing) categories
- **Music interference as a hard test case:** Structured harmonic content vs. incoherent environmental noise
- **Systematic ablations:** Spectral augmentation, curriculum learning, and perceptual loss redesign
- **SOTA baseline comparisons:** DeepFilterNet3 and DCCRN benchmarked against simple CNN baseline

---

## 🎯 Problem Statement

Speech enhancement models encounter diverse acoustic environments in deployment—cafeteria noise, wind, music interference—that differ from training conditions. This project asks whether models learn transferable noise suppression strategies or merely memorize training noise patterns.

**Noise Type Partitioning:**
- **Seen (Training):** Background music (classical, jazz), factory noise, street traffic
- **Unseen (Testing):** Cafeteria/crowd noise, wind, music interference (rock, electronic, pop)

The gap between seen and unseen performance quantifies generalization capability.

---

## 📊 Dataset

**Source:** Deep Noise Suppression (DNS) Challenge dataset

**Generation Process:**
1. Clean speech from LibriSpeech-derived corpus (16 kHz)
2. Noise clips categorized by type
3. Synthetic mixing at controlled SNRs (0-20 dB)
4. Conversion to magnitude spectrograms via STFT (n_fft=512, hop_length=128)

**Dataset Size:**
- Training: ~2,000 paired (noisy, clean) spectrograms from seen noise types
- Testing: ~500 pairs from unseen noise types

---

## 🧠 Models

### State-of-the-Art Baselines

1. **DeepFilterNet3** — Lightweight, real-time model optimized for edge deployment
2. **DCCRN** — Complex-valued network processing magnitude and phase jointly

### Simple Baseline

3. **U-Net CNN** — Encoder-decoder architecture for pipeline validation

---

## 📈 Evaluation Metrics

- **PESQ** (Perceptual Evaluation of Speech Quality): 1.0–4.5 scale
- **STOI** (Short-Time Objective Intelligibility): 0–1 scale
- **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio): Signal fidelity metric

Results reported separately by noise type to isolate generalization success/failure.

---

## 🔬 Training Strategy Ablations

### 1. Spectral Augmentation with Noise Mixing
- **Strategy:** Blend multiple noise types + SpecAugment-style masking
- **Hypothesis:** Hybrid noise conditions → more robust, noise-invariant features
- **Comparison:** Pure noise training vs. augmented training

### 2. Curriculum Learning with SNR Scheduling
- **Strategy:** Start with high-SNR (easy) examples, progressively introduce low-SNR (hard)
- **Hypothesis:** Gradual difficulty increase → better robustness to extreme noise
- **Comparison:** Uniform random SNR vs. scheduled SNR progression

### 3. Perceptual Loss Redesign (Optional)
- **Strategy:** Replace/augment MSE with multi-scale spectral loss or differentiable PESQ
- **Hypothesis:** Optimizing for perceptual quality → better generalization
- **Comparison:** MSE-only vs. perceptual loss variant

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- (Optional) CUDA for GPU acceleration

### Installation

**Option 1: Using `uv` (recommended, faster)**
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/paulsoriiiano/clarity-net.git
cd clarity-net

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Option 2: Using `pip` (standard)**
```bash
# Clone the repository
git clone https://github.com/paulsoriiiano/clarity-net.git
cd clarity-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download DNS Challenge Dataset

```bash
bash scripts/download_dns.sh
```

Or manually download from [DNS Challenge](https://github.com/microsoft/DNS-Challenge) and organize into `data/raw/`.

---

## 💻 Usage

### 1. Generate Paired Dataset

```bash
python scripts/prepare_dataset.py --config configs/base_config.yaml
```

This creates paired (noisy, clean) spectrograms in `data/processed/`.

### 2. Train Models

**Baseline CNN:**
```bash
python scripts/train.py --config configs/baseline_cnn.yaml
```

**DeepFilterNet:**
```bash
python scripts/train.py --config configs/deepfilternet.yaml
```

**DCCRN:**
```bash
python scripts/train.py --config configs/dccrn.yaml
```

**Ablations:**
```bash
python scripts/train.py --config configs/ablation_augmentation.yaml
python scripts/train.py --config configs/ablation_curriculum.yaml
```

### 3. Evaluate Models

```bash
python scripts/evaluate_all.py --checkpoint checkpoints/deepfilternet/best_model.pth
```

Results saved to `results/metrics/`.

### 4. Run Real-Time Demo

```bash
python src/inference/realtime_demo.py --model checkpoints/deepfilternet/best_model.pth
```

Speak into your microphone with background noise—enhanced audio plays in real-time.

---

## 📂 Project Structure

```
speech-enhancement-generalization/
├── data/                    # Raw and processed datasets (gitignored)
├── src/                     # Source code
│   ├── data/                # Data processing utilities
│   ├── models/              # Model architectures
│   ├── training/            # Training loop and augmentation
│   ├── evaluation/          # Metrics and evaluation scripts
│   └── inference/           # Inference and demo
├── configs/                 # YAML configuration files
├── scripts/                 # Standalone scripts for training/eval
├── checkpoints/             # Saved model weights (gitignored)
├── results/                 # Metrics, audio samples, plots (gitignored)
├── notebooks/               # Jupyter notebooks for exploration
└── docs/                    # Proposal, report, presentation
```

See `project_structure.txt` for full details.

---

## 📊 Results (To Be Updated)

| Model | Seen PESQ | Unseen PESQ | Generalization Gap |
|-------|-----------|-------------|--------------------|
| Baseline CNN | TBD | TBD | TBD |
| DeepFilterNet3 | TBD | TBD | TBD |
| DCCRN | TBD | TBD | TBD |
| + Augmentation | TBD | TBD | TBD |
| + Curriculum | TBD | TBD | TBD |

Full results breakdown by noise type coming soon.

---

## 🎥 Demo

[Link to demonstration video will be added here]

---

## 📝 Citation

If you use this code or methodology in your research, please cite:

```
@misc{ClarityNet,
  author = {Paul Junver Soriano},
  title = {ClarityNet: Cross-Domain Generalization in Speech Enhancement},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/paulsoriiiano/clarity-net}
}
```

---

## 📄 License

MIT License — see `LICENSE` file for details.

---

## 🙏 Acknowledgments

- DNS Challenge dataset by Microsoft
- DeepFilterNet by Rikorose
- DCCRN implementation references
- Course: CMPE 258 Deep Learning (San Jose State University)

---

## 📧 Contact

**Your Name**  
Email: pauljunversoriano@gmail.com 
LinkedIn: [Paul Junver Soriano](https://linkedin.com/in/paul-junver-soriano)

