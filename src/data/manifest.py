import json
import random
from pathlib import Path
from datetime import datetime

from src.data.audio_utils import extract_category_from_filename


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def create_manifest(clean_dir, noise_dir, save_path):
    """Create fixed train/val/test splits."""

    clean_dir = Path(clean_dir)
    noise_dir = Path(noise_dir)
    save_path = Path(save_path)

    clean_files = list(clean_dir.rglob("*.flac")) + list(clean_dir.rglob("*.wav"))

    noise_categories = {
        "natural_continuous": [10, 11, 12, 15, 16, 17, 19],
        "domestic_mechanical": [35, 36],
        "transport_urban": [40, 42, 43, 44, 45, 47],
        "impulsive_events": [30, 34, 39, 46, 48, 49],
    }

    seen_cats = noise_categories["natural_continuous"] + noise_categories["domestic_mechanical"]
    unseen_cats = noise_categories["transport_urban"] + noise_categories["impulsive_events"]

    all_noise_files = []
    for category in noise_categories.keys():
        cat_path = noise_dir / category
        if cat_path.exists():
            all_noise_files.extend(list(cat_path.rglob("*.wav")))

    seen_noise = []
    unseen_noise = []

    for f in all_noise_files:
        cat = extract_category_from_filename(f.name)
        if cat in seen_cats:
            seen_noise.append(f)
        elif cat in unseen_cats:
            unseen_noise.append(f)

    random.seed(42)
    random.shuffle(clean_files)

    n_total = len(clean_files)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    root = get_project_root()

    def rel(p: Path) -> str:
        return str(p.resolve().relative_to(root))

    manifest = {
        "root_dir": str(root),
        "seed": 42,
        "created_at": datetime.now().isoformat(),
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_total - n_train - n_val,
        "splits": {
            "train": [rel(f) for f in clean_files[:n_train]],
            "val": [rel(f) for f in clean_files[n_train:n_train + n_val]],
            "test": [rel(f) for f in clean_files[n_train + n_val:]],
        },
        "noise": {
            "seen": [rel(f) for f in seen_noise],
            "unseen": [rel(f) for f in unseen_noise],
        },
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("✓ Created MANIFEST:")
    print(f"  Train: {len(manifest['splits']['train'])} files")
    print(f"  Val:   {len(manifest['splits']['val'])} files")
    print(f"  Test:  {len(manifest['splits']['test'])} files")
    print(f"  Seen noise: {len(manifest['noise']['seen'])} files")
    print(f"  Unseen noise: {len(manifest['noise']['unseen'])} files")
    print(f"  Saved to: {save_path}")

    return manifest


def load_manifest(manifest_path=None):
    """Load splits from MANIFEST file and resolve to absolute Paths."""
    root = get_project_root()

    if manifest_path is None:
        manifest_path = root / "data" / "MANIFEST.json"
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path

    with open(manifest_path) as f:
        manifest = json.load(f)

    def resolve(paths):
        return [(root / p).resolve() for p in paths]

    return {
        "train": resolve(manifest["splits"]["train"]),
        "val": resolve(manifest["splits"]["val"]),
        "test": resolve(manifest["splits"]["test"]),
        "noise_seen": resolve(manifest["noise"]["seen"]),
        "noise_unseen": resolve(manifest["noise"]["unseen"]),
    }


def main():
    root = get_project_root()
    create_manifest(
        clean_dir=root / "data" / "raw" / "clean_speech",
        noise_dir=root / "data" / "raw" / "noise",
        save_path=root / "data" / "MANIFEST.json",
    )


if __name__ == "__main__":
    main()