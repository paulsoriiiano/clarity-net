#!/usr/bin/env python3
"""Train the final FullSubNet model from a YAML config.

This script is designed to mirror the 04_fullsubnet_training notebook while also
supporting the config-driven workflow used in later ablation notebooks.

Features
--------
- Loads experiment settings from YAML (supports ``!!python/tuple`` via
  ``yaml.full_load``)
- Builds FullSubNet with the best config discovered in ablations
- Creates deterministic train/val/test datasets using ``FullSubNetDataset``
- Trains with validation on seen and unseen noise
- Tracks PESQ, STOI, SI-SNR, and generalization gaps per epoch
- Saves checkpoints for:
    * best seen validation loss
    * best unseen validation PESQ
    * final epoch
- Optionally evaluates the best checkpoint on the test set at the end

Example
-------
python scripts/train_final_fullsubnet.py \
    --project-root . \
    --config configs/final_fullsubnet.yaml \
    --loss-name sisnr
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.full_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict config in {path}, got {type(cfg)}")
    return cfg


def save_yaml_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

class WaveformL1Loss(nn.Module):
    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)
        return nn.functional.l1_loss(enhanced, clean)


class WaveformMSELoss(nn.Module):
    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)
        return nn.functional.mse_loss(enhanced, clean)


class SISNRLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)

        clean = clean - clean.mean(dim=-1, keepdim=True)
        enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)

        target = (torch.sum(enhanced * clean, dim=-1, keepdim=True) * clean) / (
            torch.sum(clean ** 2, dim=-1, keepdim=True) + self.eps
        )
        noise = enhanced - target

        ratio = (torch.sum(target ** 2, dim=-1) + self.eps) / (
            torch.sum(noise ** 2, dim=-1) + self.eps
        )
        si_snr = 10.0 * torch.log10(ratio + self.eps)
        return -si_snr.mean()


class MagnitudeL1Loss(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)

        window = self.window.to(enhanced.device)
        enhanced_stft = torch.stft(
            enhanced,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        clean_stft = torch.stft(
            clean,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        return nn.functional.l1_loss(torch.abs(enhanced_stft), torch.abs(clean_stft))


class MagnitudeMSELoss(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if enhanced.dim() == 3 and enhanced.size(1) == 1:
            enhanced = enhanced.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)

        window = self.window.to(enhanced.device)
        enhanced_stft = torch.stft(
            enhanced,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        clean_stft = torch.stft(
            clean,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        return nn.functional.mse_loss(torch.abs(enhanced_stft), torch.abs(clean_stft))


class CombinedMagL1SISNRLoss(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512,
                 mag_weight: float = 1.0, sisnr_weight: float = 1.0):
        super().__init__()
        self.mag = MagnitudeL1Loss(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.sisnr = SISNRLoss()
        self.mag_weight = mag_weight
        self.sisnr_weight = sisnr_weight

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return self.mag_weight * self.mag(enhanced, clean) + self.sisnr_weight * self.sisnr(enhanced, clean)


# -----------------------------------------------------------------------------
# Core training helpers
# -----------------------------------------------------------------------------

def compute_si_snr_metric(enhanced: np.ndarray, clean: np.ndarray, eps: float = 1e-8) -> float:
    enhanced = enhanced - np.mean(enhanced)
    clean = clean - np.mean(clean)
    target = np.dot(enhanced, clean) * clean / (np.sum(clean ** 2) + eps)
    noise = enhanced - target
    ratio = (np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return float(10.0 * np.log10(ratio + eps))


def build_loss(config: Dict[str, Any]) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = str(loss_cfg.get("name", "sisnr")).lower()

    model_cfg = config.get("model", {})
    n_fft = model_cfg.get("n_fft", 512)
    hop_length = model_cfg.get("hop_length", 128)
    win_length = model_cfg.get("win_length", 512)

    if name in {"wav_l1", "waveform_l1", "l1"}:
        return WaveformL1Loss()
    if name in {"wav_mse", "waveform_mse", "mse"}:
        return WaveformMSELoss()
    if name in {"sisnr", "si_snr"}:
        return SISNRLoss()
    if name == "mag_l1":
        return MagnitudeL1Loss(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    if name == "mag_mse":
        return MagnitudeMSELoss(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    if name in {"mag_l1_sisnr", "combined_mag_l1_sisnr"}:
        return CombinedMagL1SISNRLoss(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            mag_weight=float(loss_cfg.get("mag_weight", 1.0)),
            sisnr_weight=float(loss_cfg.get("sisnr_weight", 1.0)),
        )
    raise ValueError(f"Unsupported loss name: {name}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    history: Dict[str, List[Any]],
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "config": config,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def choose_model_kwargs(fullsubnet_cls: Any, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fullsubnet_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in model_cfg.items() if k in allowed}


def maybe_import_project(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def create_datasets(manifest: Dict[str, Any], config: Dict[str, Any], dataset_cls: Any) -> Dict[str, Any]:
    seed = int(config.get("seed", 42))
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    snr_range = tuple(data_cfg.get("train_snr_range", data_cfg.get("snr_range", (-5, 20))))
    augmentation_factor = int(data_cfg.get("augmentation_factor", train_cfg.get("augmentation_factor", 1)))
    eval_pairs = int(data_cfg.get("eval_pairs", train_cfg.get("eval_pairs", 1)))
    use_snr_curriculum = bool(data_cfg.get("use_snr_curriculum", False))
    snr_curriculum = data_cfg.get("snr_curriculum", None)

    train_dataset = dataset_cls(
        clean_files=manifest["train"],
        noise_files=manifest["noise_seen"],
        mode="train",
        snr_range=snr_range,
        augmentation_factor=augmentation_factor,
        base_seed=seed,
        epoch=0,
        use_snr_curriculum=use_snr_curriculum,
        snr_curriculum=snr_curriculum,
    )

    val_seen_dataset = dataset_cls(
        clean_files=manifest["val"],
        noise_files=manifest["noise_seen"],
        mode="val",
        snr_range=snr_range,
        eval_pairs_per_clean=eval_pairs,
        base_seed=seed,
    )
    val_unseen_dataset = dataset_cls(
        clean_files=manifest["val"],
        noise_files=manifest["noise_unseen"],
        mode="val",
        snr_range=snr_range,
        eval_pairs_per_clean=eval_pairs,
        base_seed=seed + 1,
    )
    test_seen_dataset = dataset_cls(
        clean_files=manifest["test"],
        noise_files=manifest["noise_seen"],
        mode="test",
        snr_range=snr_range,
        eval_pairs_per_clean=eval_pairs,
        base_seed=seed + 2,
    )
    test_unseen_dataset = dataset_cls(
        clean_files=manifest["test"],
        noise_files=manifest["noise_unseen"],
        mode="test",
        snr_range=snr_range,
        eval_pairs_per_clean=eval_pairs,
        base_seed=seed + 3,
    )

    return {
        "train": train_dataset,
        "val_seen": val_seen_dataset,
        "val_unseen": val_unseen_dataset,
        "test_seen": test_seen_dataset,
        "test_unseen": test_unseen_dataset,
    }


def create_dataloaders(datasets: Dict[str, Any], batch_size: int, seed: int, num_workers: int = 0) -> Dict[str, DataLoader]:
    train_gen = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=train_gen),
        "val_seen": DataLoader(datasets["val_seen"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "val_unseen": DataLoader(datasets["val_unseen"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_seen": DataLoader(datasets["test_seen"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_unseen": DataLoader(datasets["test_unseen"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def init_history() -> Dict[str, List[Any]]:
    return {
        "epoch": [],
        "train_loss": [],
        "val_seen_loss": [],
        "val_seen_pesq": [],
        "val_seen_stoi": [],
        "val_seen_si_snr": [],
        "val_unseen_loss": [],
        "val_unseen_pesq": [],
        "val_unseen_stoi": [],
        "val_unseen_si_snr": [],
        "generalization_gap_loss": [],
        "generalization_gap_pesq": [],
        "generalization_gap_stoi": [],
        "generalization_gap_si_snr": [],
        "learning_rate": [],
        "epoch_time_sec": [],
    }



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    desc = "Training"
    if epoch is not None and total_epochs is not None:
        desc = f"Training (Epoch {epoch}/{total_epochs})"

    batch_bar = tqdm(loader, desc=desc, leave=False, position=1)
    for batch in batch_bar:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        pred = model(noisy)
        if pred.dim() == 3 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        loss = criterion(pred, clean)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        count += 1
        batch_bar.set_postfix(loss=f"{total_loss / max(count, 1):.6f}")

    return total_loss / max(count, 1)


def validate_with_metrics_fallback(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_pesq_samples: int = 100,
    sample_rate: int = 16000,
) -> Dict[str, float]:
    # Try project-native implementation first.
    try:
        from src.evaluation.metrics import validate_with_metrics as project_validate  # type: ignore
        result = project_validate(model=model, loader=loader, criterion=criterion, device=device, max_pesq_samples=max_pesq_samples)
        if "si_snr" not in result:
            result["si_snr"] = float("nan")
        return result
    except Exception:
        pass

    # Fallback implementation if project helper is unavailable.
    try:
        from pesq import pesq as pesq_fn  # type: ignore
    except Exception:
        pesq_fn = None
    try:
        from pystoi import stoi as stoi_fn  # type: ignore
    except Exception:
        stoi_fn = None

    model.eval()
    total_loss = 0.0
    pesq_scores: List[float] = []
    stoi_scores: List[float] = []
    sisnr_scores: List[float] = []

    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred = model(noisy)
            if pred.dim() == 3 and pred.size(1) == 1:
                pred_for_loss = pred.squeeze(1)
            else:
                pred_for_loss = pred
            total_loss += float(criterion(pred_for_loss, clean).item())

            if len(pesq_scores) >= max_pesq_samples:
                continue

            pred_np = pred_for_loss.detach().cpu().numpy()
            clean_np = clean.detach().cpu().numpy()
            batch_take = min(len(pred_np), max_pesq_samples - len(pesq_scores))
            for i in range(batch_take):
                enh = pred_np[i]
                ref = clean_np[i]
                min_len = min(len(enh), len(ref))
                enh = enh[:min_len]
                ref = ref[:min_len]
                if pesq_fn is not None:
                    try:
                        pesq_scores.append(float(pesq_fn(sample_rate, ref, enh, "wb" if sample_rate == 16000 else "nb")))
                    except Exception:
                        pass
                if stoi_fn is not None:
                    try:
                        stoi_scores.append(float(stoi_fn(ref, enh, sample_rate, extended=False)))
                    except Exception:
                        pass
                try:
                    sisnr_scores.append(compute_si_snr_metric(enh, ref))
                except Exception:
                    pass

    return {
        "loss": total_loss / max(len(loader), 1),
        "pesq": float(np.mean(pesq_scores)) if pesq_scores else float("nan"),
        "stoi": float(np.mean(stoi_scores)) if stoi_scores else float("nan"),
        "si_snr": float(np.mean(sisnr_scores)) if sisnr_scores else float("nan"),
        "pesq_count": len(pesq_scores),
        "stoi_count": len(stoi_scores),
        "si_snr_count": len(sisnr_scores),
    }


def better(value: float, best: float, mode: str) -> bool:
    if math.isnan(value):
        return False
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError(mode)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final FullSubNet model from config")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root containing src/ and data/")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional MANIFEST.json path")
    parser.add_argument("--results-root", type=Path, default=None, help="Optional root directory for results")
    parser.add_argument("--loss-name", type=str, default=None, help="Override loss name (e.g. sisnr, wav_l1, mag_l1)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--max-metric-samples", type=int, default=100, help="Validation PESQ/STOI sample budget")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop-metric", type=str, default="seen_loss", choices=["seen_loss", "unseen_pesq"],
                        help="Metric used for early stopping patience")
    parser.add_argument("--no-final-eval", action="store_true", help="Skip final test-set evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    maybe_import_project(project_root)

    # Project imports after sys.path update.
    from src.data.manifest import load_manifest  # type: ignore
    from src.data.dataset import FullSubNetDataset  # type: ignore
    from src.models.fullsubnet import FullSubNet  # type: ignore

    config_path = args.config.resolve()
    config = load_yaml_config(config_path)

    # Inject/override loss config if needed.
    config.setdefault("loss", {})
    if args.loss_name is not None:
        config["loss"]["name"] = args.loss_name
    config["loss"].setdefault("name", "sisnr")

    seed = int(config.get("seed", config.get("training", {}).get("seed", 42)))
    set_seed(seed)

    device = get_device(args.device)

    exp_name = str(config.get("experiment_name", config_path.stem))
    results_root = args.results_root.resolve() if args.results_root else (project_root / "results" / "final_fullsubnet")
    results_dir = ensure_path(results_root / exp_name)
    checkpoint_dir = ensure_path(results_dir / "checkpoints")
    save_yaml_config(config, results_dir / "resolved_config.yaml")

    manifest_path = args.manifest.resolve() if args.manifest else (project_root / "data" / "MANIFEST.json")
    manifest = load_manifest(manifest_path)

    datasets = create_datasets(manifest, config, FullSubNetDataset)
    batch_size = int(config.get("training", {}).get("batch_size", 8))
    loaders = create_dataloaders(datasets, batch_size=batch_size, seed=seed, num_workers=args.num_workers)

    model_kwargs = choose_model_kwargs(FullSubNet, config.get("model", {}))
    model = FullSubNet(**model_kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = build_loss(config)
    training_cfg = config.get("training", {})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Model params: {total_params:,}")
    print(f"Loss: {config['loss']['name']}")
    print(f"Results dir: {results_dir}")
    print("=" * 80)

    history = init_history()
    num_epochs = int(training_cfg.get("epochs", 40))
    patience = int(training_cfg.get("patience", 10))
    grad_clip = training_cfg.get("gradient_clip", training_cfg.get("grad_clip", 5.0))
    grad_clip = None if grad_clip is None else float(grad_clip)

    best_seen_loss = float("inf")
    best_unseen_pesq = -float("inf")
    best_early_stop_score = float("inf") if args.early_stop_metric == "seen_loss" else -float("inf")
    epochs_without_improvement = 0

    start_time = time.time()

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs", position=0)
    for epoch in epoch_bar:
        epoch_start = time.time()
        datasets["train"].set_epoch(epoch)

        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            grad_clip=grad_clip,
            epoch=epoch,
            total_epochs=num_epochs,
        )
        val_seen = validate_with_metrics_fallback(model, loaders["val_seen"], criterion, device, max_pesq_samples=args.max_metric_samples)
        val_unseen = validate_with_metrics_fallback(model, loaders["val_unseen"], criterion, device, max_pesq_samples=args.max_metric_samples)

        scheduler.step(val_seen["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        gap_loss = val_unseen["loss"] - val_seen["loss"]
        gap_pesq = val_seen["pesq"] - val_unseen["pesq"]
        gap_stoi = val_seen["stoi"] - val_unseen["stoi"]
        gap_si_snr = val_seen["si_snr"] - val_unseen["si_snr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_seen_loss"].append(val_seen["loss"])
        history["val_seen_pesq"].append(val_seen["pesq"])
        history["val_seen_stoi"].append(val_seen["stoi"])
        history["val_seen_si_snr"].append(val_seen["si_snr"])
        history["val_unseen_loss"].append(val_unseen["loss"])
        history["val_unseen_pesq"].append(val_unseen["pesq"])
        history["val_unseen_stoi"].append(val_unseen["stoi"])
        history["val_unseen_si_snr"].append(val_unseen["si_snr"])
        history["generalization_gap_loss"].append(gap_loss)
        history["generalization_gap_pesq"].append(gap_pesq)
        history["generalization_gap_stoi"].append(gap_stoi)
        history["generalization_gap_si_snr"].append(gap_si_snr)
        history["learning_rate"].append(current_lr)
        history["epoch_time_sec"].append(time.time() - epoch_start)

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            unseen_pesq=f"{val_unseen['pesq']:.3f}",
            unseen_sisnr=f"{val_unseen['si_snr']:.3f}",
            gap=f"{gap_pesq:.3f}",
        )

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss:         {train_loss:.6f}")
        print(f"  Val (seen)   - Loss: {val_seen['loss']:.6f}  PESQ: {val_seen['pesq']:.3f}  STOI: {val_seen['stoi']:.3f}  SI-SNR: {val_seen['si_snr']:.3f}")
        print(f"  Val (unseen) - Loss: {val_unseen['loss']:.6f}  PESQ: {val_unseen['pesq']:.3f}  STOI: {val_unseen['stoi']:.3f}  SI-SNR: {val_unseen['si_snr']:.3f}")
        print(f"  Gap          - Loss: {gap_loss:+.6f}  PESQ: {gap_pesq:+.3f}  STOI: {gap_stoi:+.3f}  SI-SNR: {gap_si_snr:+.3f}")
        print(f"  Learning Rate:      {current_lr:.2e}")

        # Save best checkpoints using both criteria.
        if val_seen["loss"] < best_seen_loss:
            best_seen_loss = val_seen["loss"]
            save_checkpoint(
                checkpoint_dir / "best_seen_loss.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                config,
                extra={"best_seen_loss": best_seen_loss, "best_unseen_pesq": best_unseen_pesq},
            )
            print("  ✓ Saved best_seen_loss checkpoint")

        if not math.isnan(val_unseen["pesq"]) and val_unseen["pesq"] > best_unseen_pesq:
            best_unseen_pesq = val_unseen["pesq"]
            save_checkpoint(
                checkpoint_dir / "best_unseen_pesq.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                config,
                extra={"best_seen_loss": best_seen_loss, "best_unseen_pesq": best_unseen_pesq},
            )
            print("  ✓ Saved best_unseen_pesq checkpoint")

        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                config,
                extra={"best_seen_loss": best_seen_loss, "best_unseen_pesq": best_unseen_pesq},
            )

        # Early stopping
        if args.early_stop_metric == "seen_loss":
            current_score = val_seen["loss"]
            improved = better(current_score, best_early_stop_score, "min")
        else:
            current_score = val_unseen["pesq"]
            improved = better(current_score, best_early_stop_score, "max")

        if improved:
            best_early_stop_score = current_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  No early-stop improvement for {epochs_without_improvement} epoch(s)")

        with open(results_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            break

    total_minutes = (time.time() - start_time) / 60.0
    save_checkpoint(
        checkpoint_dir / "final_model.pth",
        model,
        optimizer,
        scheduler,
        history["epoch"][-1],
        history,
        config,
        extra={"best_seen_loss": best_seen_loss, "best_unseen_pesq": best_unseen_pesq, "total_training_minutes": total_minutes},
    )

    summary = {
        "experiment_name": exp_name,
        "device": str(device),
        "loss_name": config["loss"]["name"],
        "model_params": total_params,
        "epochs_completed": history["epoch"][-1] if history["epoch"] else 0,
        "best_seen_loss": best_seen_loss,
        "best_unseen_pesq": best_unseen_pesq,
        "total_training_minutes": total_minutes,
        "best_unseen_epoch": int(np.nanargmax(np.asarray(history["val_unseen_pesq"], dtype=float)) + 1) if history["val_unseen_pesq"] else None,
    }
    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete")
    print(json.dumps(summary, indent=2))
    print("=" * 80)

    if args.no_final_eval:
        return

    try:
        from src.evaluation.evaluate import evaluate_model  # type: ignore
    except Exception as e:
        print(f"Skipping final evaluation because evaluate_model could not be imported: {e}")
        return

    # Evaluate using best unseen-PESQ checkpoint if available, otherwise best seen-loss.
    eval_ckpt_path = checkpoint_dir / "best_unseen_pesq.pth"
    if not eval_ckpt_path.exists():
        eval_ckpt_path = checkpoint_dir / "best_seen_loss.pth"
    ckpt = torch.load(eval_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    evaluate_model(model, loaders["test_seen"], loaders["test_unseen"], results_dir, device=device)


if __name__ == "__main__":
    main()
