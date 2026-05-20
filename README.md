# ClarityNet: Cross-Domain Generalization in Speech Enhancement

**Robust Noise Suppression Across Unseen Acoustic Environments**

A deep learning project investigating how well speech enhancement models generalize to noise types they've never encountered during training.

**Status:** ✅ Core framework complete with trained FullSubNet model
- Dataset preparation pipeline
- Train/val/test splits with seen/unseen noise categorization
- FullSubNet model with systematic architecture and loss ablations
- Interactive web demos and CLI tools
- Comprehensive evaluation on both seen and unseen noise

---

## 📋 Overview

Most speech enhancement systems are trained and tested on the same noise distribution, leaving their real-world robustness unexplored. This project systematically measures the **generalization gap** between seen and unseen noise conditions and evaluates training strategies to close it.

**Research Question:** *How well does a speech enhancement model trained on specific noise types generalize to completely unseen noise conditions, and what training strategies can improve this generalization?*

### Key Contributions

- **Explicit generalization evaluation:** ESC-50 noise categories split into seen (training) vs. unseen (testing) partitions
- **Deterministic data reproducibility:** MANIFEST.json ensures identical splits across runs and machines
- **FullSubNet architecture:** Fullband + subband processing with learned frequency-wise masking
- **Systematic ablations:** 8 architecture variants + 6 loss functions evaluated on seen/unseen generalization
- **Comprehensive demos:** Web UI, CLI, and Python API for inference

---

## 🎯 Problem Statement

Speech enhancement models encounter diverse acoustic environments in deployment—crowd noise, wind, traffic—that differ from training conditions. This project systematically measures the **generalization gap** when a model trained on specific noise types is evaluated on completely unseen noise.

**Dataset Approach:**
- **Clean speech:** LibriSpeech development set (~330MB)
- **Noise source:** ESC-50 environmental sound database (50 categories)
- **Seen noise:** Subset of ESC-50 categories used during training
- **Unseen noise:** Held-out ESC-50 categories to evaluate generalization
- **Metric:** Separate PESQ, STOI, SI-SNR evaluation on seen vs. unseen test sets

---

## 📊 Dataset

**Sources:**
- **Clean speech:** LibriSpeech dev-clean (16 kHz mono)
- **Noise:** ESC-50 environmental sounds database (50 categories)

**Preparation:**
1. Download via `python scripts/download_small_dataset.py`
2. Generate train/val/test splits and seen/unseen noise categories via `python -m src.data.manifest`
3. Creates `data/MANIFEST.json` with all file paths and split assignments

**Dataset Statistics (from MANIFEST):**
- Total samples: 200 clean speech files
- Train/Val/Test: 160 / 20 / 20 splits
- Training noise: 15 ESC-50 categories (seen)
- Testing noise: 7 ESC-50 categories (unseen, held-out for generalization testing)

**Noise Categories:**
- Natural continuous (wind, rain, etc.)
- Domestic mechanical (appliances, etc.)
- Transport & urban (traffic, sirens, etc.)
- Impulsive events (gunshots, breaking glass, etc.)

---

## 🧠 Models

### Primary Model: FullSubNet
**FullSubNet** - Frequency-wise speech enhancement via fullband and subband processing
- Fullband branch: Processes all frequencies globally
- Subband branch: Frequency-specific refinement with LSTM/GRU
- Outputs complex ratio mask (magnitude + phase)
- SI-SNR loss (selected from ablation studies)

### Baseline Models (For Comparison)

1. **Baseline CNN** — Simple encoder-decoder for pipeline validation
2. **DCCRN** — Complex-valued convolutional recurrent network
3. **FullSubNet+** — Enhanced variant with additional refinement

### Training Configuration

The final model uses the configuration selected from ablation studies:
- **Sequence Model:** LSTM (vs GRU)
- **Normalization:** Cumulative Laplace Norm (vs offline Gaussian)
- **Dropout:** 0.2 (vs 0.0, 0.1)
- **Loss Function:** SI-SNR (vs MSE, L1, Complex MSE)

---

## 📈 Evaluation Metrics

- **PESQ** (Perceptual Evaluation of Speech Quality): 1.0–4.5 scale
- **STOI** (Short-Time Objective Intelligibility): 0–1 scale
- **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio): Signal fidelity metric

Results reported separately by noise type to isolate generalization success/failure.

---

## 🔬 Training Strategy Ablations

The project systematically evaluated different training strategies. Results are saved in `results/` with separate metrics for seen and unseen noise.

### 1. Architecture Ablations
- **Sequence Model Variants:** LSTM vs GRU vs bidirectional
- **Dropout Rates:** 0.0, 0.1, 0.2
- **Normalization Strategies:** Cumulative Laplace Norm vs Offline Gaussian Norm

Hypothesis: Different architectures trade off generalization vs seen-set performance.

### 2. Loss Function Ablations
- **SI-SNR** (selected) — Scale-invariant signal-to-noise ratio
- **Magnitude MSE** — L2 loss on spectrogram magnitude
- **Magnitude L1** — L1 loss on spectrogram magnitude
- **Waveform L1** — L1 loss directly on waveforms
- **Complex MSE** — MSE on complex-valued STFT

Hypothesis: Perceptual losses (SI-SNR) better capture human-perceivable improvement than magnitude-only losses.

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

### 1. Download and Prepare Dataset

The project uses the **ESC-50** environmental sounds dataset (for noise) and **LibriSpeech** (for clean speech). Run the download script to fetch and organize the data:

```bash
python scripts/download_small_dataset.py
```

This will:
- Download ~330MB of LibriSpeech dev-clean audio (200 files)
- Download ESC-50 sound effects database
- Organize into `data/raw/clean_speech/` and `data/raw/noise/`

### 2. Create Train/Val/Test Splits and MANIFEST

After downloading, create the deterministic data splits:

```bash
python -m src.data.manifest
```

This generates `data/MANIFEST.json` containing:
- **Train/Val/Test splits** for clean speech (80/10/10)
- **Seen/Unseen noise category splits** (ESC-50 categories)
- All file paths relative to project root for portability

The MANIFEST ensures reproducible dataset partitioning and makes the seen/unseen generalization gap measurable.

**Noise Category Splits:**
- **Seen (Training):** Categories from natural, mechanical, transport, and impulsive sound groups
- **Unseen (Testing):** Held-out categories from each sound group to ensure diversity

### 3. Train the FullSubNet Model

Train the final FullSubNet model (with SI-SNR loss) selected from ablations:

```bash
python scripts/train_final_fullsubnet.py \
    --project-root . \
    --config configs/final_fullsubnet.yaml \
    --loss-name sisnr
```

**Available Options:**
- `--device` — Specify device: `cuda`, `mps`, or `cpu`
- `--resume-checkpoint` — Resume from existing checkpoint
- `--disable-eval` — Skip evaluation during training

The training will:
- Create deterministic train/val/test datasets from MANIFEST
- Train with validation on **both seen and unseen noise** separately
- Track PESQ, STOI, SI-SNR metrics and generalization gaps
- Save checkpoints for best seen loss and best unseen PESQ
- Save results to `results/final_fullsubnet/`

### 4. Run Architecture and Loss Ablations

Ablation studies are already configured. To re-run:

**Architecture Ablations** (test different dropout rates, normalization, sequence models):
```bash
python scripts/train_final_fullsubnet.py \
    --project-root . \
    --config configs/architecture_ablations/fullsubnet_lstm.yaml \
    --loss-name sisnr
```

Available architecture ablations:
- `fullsubnet_lstm.yaml` — LSTM sequence model
- `fullsubnet_gru.yaml` — GRU sequence model
- `fullsubnet_dp_*.yaml` — Different dropout rates
- `fullsubnet_*_laplace.yaml` / `*_gaussian.yaml` — Different normalization strategies

**Loss Ablations** (test different loss functions):
```bash
python scripts/train_final_fullsubnet.py \
    --project-root . \
    --config configs/loss_ablations/fullsubnet_sisnr.yaml \
    --loss-name sisnr
```

Available loss ablations:
- `fullsubnet_sisnr.yaml` — SI-SNR loss (recommended)
- `fullsubnet_mag_mse.yaml` — Magnitude MSE
- `fullsubnet_mag_l1.yaml` — Magnitude L1
- `fullsubnet_wav_l1.yaml` — Waveform L1
- `fullsubnet_complex_mse.yaml` — Complex-valued MSE

Results saved to `results/fullsubnet_*_eval/`.

### 5. Run Interactive Demos

**Basic Web UI (recommended for most users):**
```bash
python gradio_demo.py
```
Access at `http://localhost:7860`. Upload audio files or record live microphone input for real-time enhancement.

**Advanced Web UI (with metrics and spectrograms):**
```bash
python gradio_demo_advanced.py
```
Includes PESQ, STOI, SI-SNR metrics and Mel-spectrogram visualizations.

**Command-Line Tool (for batch processing):**
```bash
# Single file
python enhance_audio_cli.py -i noisy.wav -o enhanced.wav -m checkpoints/fullsubnet_final/best_model.pth

# Batch processing
python enhance_audio_cli.py -b /path/to/audios -o /output -m checkpoints/fullsubnet_final/best_model.pth
```

**Python API (for custom workflows):**
```python
from gradio_demo import SpeechEnhancementDemo
import librosa

model = SpeechEnhancementDemo("checkpoints/fullsubnet_final/best_model.pth", device="cuda")
audio, sr = librosa.load("noisy.wav", sr=16000)
enhanced = model.enhance_audio(audio)
```

See `examples.py` for more usage patterns (batch enhancement, streaming/chunked processing, metric computation).

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
| DCCRN | TBD | TBD | TBD |
| FullSubNet (MSE loss) | TBD | TBD | TBD |
| FullSubNet (SI-SNR loss) | TBD | TBD | TBD |
| FullSubNet + Architecture Ablations | TBD | TBD | TBD |

Full results breakdown by ESC-50 noise category and architecture variant available in `results/`.

---

## 📊 Project Implementation Status

**✅ Completed:**
- Dataset download and organization (`scripts/download_small_dataset.py`)
- Deterministic data manifesting (`src/data/manifest.py`)
- Train/val/test splits with ESC-50 seen/unseen noise categories
- WaveformDataset with on-the-fly augmentation
- **Models:** FullSubNet, DCCRN, Baseline CNN, FullSubNet+
- **Architecture ablations:** 8 configurations (LSTM/GRU, dropout rates, normalization strategies)
- **Loss ablations:** 6 configurations (SI-SNR, MSE, L1, Complex MSE, etc.)
- Training framework with seen/unseen validation split tracking
- Interactive Gradio web UIs (basic & advanced)
- CLI enhancement tool and Python API examples

**Future Extensions:**
- Curriculum learning with SNR scheduling
- Spectral time masking augmentation
- Multi-band processing analysis
- Real-time model optimization (quantization, pruning)

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

