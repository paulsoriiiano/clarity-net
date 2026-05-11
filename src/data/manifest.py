"""
Manifest generation and loading utilities for the clarity-net project.

This module serves two purposes:

1. Generate a deterministic MANIFEST.json file containing:
   - train / val / test splits for clean speech
   - seen / unseen splits for noise files

2. Load MANIFEST.json and resolve its stored relative paths to absolute Paths.

Design goals
------------
- Safe to import from notebooks/scripts without side effects
- Runnable directly as a script
- Portable across machines by storing relative paths in JSON
- Reproducible data splitting via a fixed random seed
- More balanced seen/unseen noise split by holding out categories from
  every super-group instead of assigning whole super-groups to unseen

Recommended usage
-----------------
From the project root:

    python -m src.data.manifest

You can also run:

    python src/data/manifest.py

because this file includes a fallback import path patch.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

# ---------------------------------------------------------------------
# Robust import so this file works both as:
#   1) python -m src.data.manifest
#   2) python src/data/manifest.py
# ---------------------------------------------------------------------
try:
    from src.data.audio_utils import extract_category_from_filename
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.audio_utils import extract_category_from_filename


# ---------------------------------------------------------------------
# Noise category definitions
# ---------------------------------------------------------------------
NOISE_CATEGORIES: Dict[str, List[int]] = {
    "natural_continuous": [10, 11, 12, 15, 16, 17, 19],
    "domestic_mechanical": [35, 36],
    "transport_urban": [40, 42, 43, 44, 45, 47],
    "impulsive_events": [30, 34, 39, 46, 48, 49],
}

# Hold out a subset from EACH super-group to make unseen more diverse.
# This is more representative than assigning whole super-groups to unseen.
UNSEEN_CATS: List[int] = [
    17, 19,      # natural_continuous
    38,          # domestic_mechanical
    45, 47,      # transport_urban
    48, 49,      # impulsive_events
]

ALL_NOISE_CATS: List[int] = sorted(
    set(
        NOISE_CATEGORIES["natural_continuous"]
        + NOISE_CATEGORIES["domestic_mechanical"]
        + NOISE_CATEGORIES["transport_urban"]
        + NOISE_CATEGORIES["impulsive_events"]
    )
)

SEEN_CATS: List[int] = sorted(set(ALL_NOISE_CATS) - set(UNSEEN_CATS))


def get_project_root() -> Path:
    """
    Return the project root directory.

    This assumes the current file lives at:
        <project_root>/src/data/manifest.py

    So the project root is two parents above this file.
    """
    return Path(__file__).resolve().parents[2]


def _to_relative(path: Path, root: Path) -> str:
    """
    Convert an absolute/relative path to a path relative to project root.

    Storing relative paths in MANIFEST.json makes the manifest portable
    across machines and folders.
    """
    return str(path.resolve().relative_to(root))


def _print_category_counts(title: str, files: Iterable[Path]) -> None:
    """
    Print a summary of how many files belong to each category ID.

    This helps sanity-check that seen/unseen splits are balanced enough
    and that no single category dominates the evaluation.
    """
    counter = Counter()

    for f in files:
        try:
            cat = extract_category_from_filename(f.name)
            counter[cat] += 1
        except Exception:
            # If a file doesn't match the expected naming scheme,
            # just skip counting it. The file may still exist, but
            # it is not usable for category-based splitting.
            continue

    print(f"\n{title}")
    if not counter:
        print("  (no files counted)")
        return

    for cat in sorted(counter):
        print(f"  Category {cat}: {counter[cat]} files")


def create_manifest(
    clean_dir: Path | str,
    noise_dir: Path | str,
    save_path: Path | str,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> dict:
    """
    Create a deterministic MANIFEST.json with clean train/val/test splits
    and seen/unseen noise splits.

    Parameters
    ----------
    clean_dir:
        Directory containing clean speech files.
    noise_dir:
        Root directory containing noise subfolders.
    save_path:
        Where to write the manifest JSON.
    seed:
        Random seed for reproducible clean-file shuffling.
    train_ratio:
        Fraction of clean files assigned to training.
    val_ratio:
        Fraction of clean files assigned to validation.
        Test ratio is whatever remains.

    Returns
    -------
    dict
        The manifest dictionary that was also written to disk.
    """
    clean_dir = Path(clean_dir)
    noise_dir = Path(noise_dir)
    save_path = Path(save_path)

    root = get_project_root()

    # -----------------------------------------------------------------
    # Load clean files recursively
    # -----------------------------------------------------------------
    clean_files = sorted(clean_dir.rglob("*.flac")) + sorted(clean_dir.rglob("*.wav"))

    if len(clean_files) == 0:
        raise FileNotFoundError(
            f"No clean speech files found under: {clean_dir}"
        )

    # -----------------------------------------------------------------
    # Load all noise files recursively from the defined category folders
    # -----------------------------------------------------------------
    all_noise_files: List[Path] = []
    for group_name in NOISE_CATEGORIES.keys():
        group_dir = noise_dir / group_name
        if group_dir.exists():
            all_noise_files.extend(sorted(group_dir.rglob("*.wav")))

    if len(all_noise_files) == 0:
        raise FileNotFoundError(
            f"No noise files found under expected noise groups in: {noise_dir}"
        )

    # -----------------------------------------------------------------
    # Split noise into seen / unseen by category ID
    # -----------------------------------------------------------------
    seen_noise: List[Path] = []
    unseen_noise: List[Path] = []
    skipped_noise: List[Path] = []

    for noise_file in all_noise_files:
        try:
            cat = extract_category_from_filename(noise_file.name)
        except Exception:
            skipped_noise.append(noise_file)
            continue

        if cat in SEEN_CATS:
            seen_noise.append(noise_file)
        elif cat in UNSEEN_CATS:
            unseen_noise.append(noise_file)
        else:
            skipped_noise.append(noise_file)

    # -----------------------------------------------------------------
    # Shuffle clean files deterministically and create splits
    # -----------------------------------------------------------------
    rng = random.Random(seed)
    rng.shuffle(clean_files)

    n_total = len(clean_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_files = clean_files[:n_train]
    val_files = clean_files[n_train:n_train + n_val]
    test_files = clean_files[n_train + n_val:]

    # -----------------------------------------------------------------
    # Build manifest with RELATIVE paths
    # -----------------------------------------------------------------
    manifest = {
        "root_dir": str(root),
        "seed": seed,
        "created_at": datetime.now().isoformat(),
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "noise_split": {
            "seen_categories": SEEN_CATS,
            "unseen_categories": UNSEEN_CATS,
        },
        "splits": {
            "train": [_to_relative(f, root) for f in train_files],
            "val": [_to_relative(f, root) for f in val_files],
            "test": [_to_relative(f, root) for f in test_files],
        },
        "noise": {
            "seen": [_to_relative(f, root) for f in seen_noise],
            "unseen": [_to_relative(f, root) for f in unseen_noise],
        },
    }

    # -----------------------------------------------------------------
    # Save manifest
    # -----------------------------------------------------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print("✓ Created MANIFEST")
    print(f"  Train clean files:  {len(train_files)}")
    print(f"  Val clean files:    {len(val_files)}")
    print(f"  Test clean files:   {len(test_files)}")
    print(f"  Seen noise files:   {len(seen_noise)}")
    print(f"  Unseen noise files: {len(unseen_noise)}")
    print(f"  Skipped noise files:{len(skipped_noise)}")
    print(f"  Saved to:           {save_path}")

    print("\nSeen categories:", SEEN_CATS)
    print("Unseen categories:", UNSEEN_CATS)

    _print_category_counts("Seen category counts:", seen_noise)
    _print_category_counts("Unseen category counts:", unseen_noise)

    if skipped_noise:
        print("\nSkipped noise files (could not parse category or category not assigned):")
        for f in skipped_noise[:10]:
            print(f"  {f}")
        if len(skipped_noise) > 10:
            print(f"  ... and {len(skipped_noise) - 10} more")

    return manifest


def load_manifest(manifest_path: Path | str | None = None) -> dict:
    """
    Load MANIFEST.json and resolve relative paths to absolute Path objects.

    Parameters
    ----------
    manifest_path:
        Optional path to the manifest file. If None, uses:
            <project_root>/data/MANIFEST.json

    Returns
    -------
    dict
        A dictionary with absolute Paths for clean/noise splits:
        {
            "train": [...],
            "val": [...],
            "test": [...],
            "noise_seen": [...],
            "noise_unseen": [...],
        }
    """
    root = get_project_root()

    if manifest_path is None:
        manifest_path = root / "data" / "MANIFEST.json"
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    def resolve(paths: List[str]) -> List[Path]:
        return [(root / p).resolve() for p in paths]

    return {
        "train": resolve(manifest["splits"]["train"]),
        "val": resolve(manifest["splits"]["val"]),
        "test": resolve(manifest["splits"]["test"]),
        "noise_seen": resolve(manifest["noise"]["seen"]),
        "noise_unseen": resolve(manifest["noise"]["unseen"]),
    }


def main() -> None:
    """
    CLI entrypoint for manifest generation.
    """
    root = get_project_root()

    create_manifest(
        clean_dir=root / "data" / "raw" / "clean_speech",
        noise_dir=root / "data" / "raw" / "noise",
        save_path=root / "data" / "MANIFEST.json",
        seed=42,
        train_ratio=0.80,
        val_ratio=0.10,
    )


if __name__ == "__main__":
    main()