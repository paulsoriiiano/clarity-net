import json
import random
from pathlib import Path

from src.data.audio_utils import (
    load_audio,
    slice_audio,
    extract_category_from_filename,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SNR_MAX_DB,
    SNR_MIN_DB,
    HOP_LENGTH,
    N_FFT,
    SEGMENT_DURATION
)

DATA_DIR = Path("data/raw")
CLEAN_DIR = DATA_DIR / "clean_speech"
NOISE_DIR = DATA_DIR / "noise"

# Noise categorization
NOISE_GROUPS = {
    "natural_continuous": [10, 11, 12, 15, 16, 17, 19],
    "domestic_mechanical": [35, 36],
    "transport_urban": [40, 42, 43, 44, 45, 47],
    "impulsive_events": [30, 34, 39, 46, 48, 49],
}

SEEN_GROUPS = ["natural_continuous", "domestic_mechanical"]
UNSEEN_GROUPS = ["transport_urban", "impulsive_events"]

SEEN_CATEGORIES = [cat for group in SEEN_GROUPS for cat in NOISE_GROUPS[group]]
UNSEEN_CATEGORIES = [cat for group in UNSEEN_GROUPS for cat in NOISE_GROUPS[group]]


MANIFEST_PATH = Path("data/processed/speech_manifest.json")
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

def sorted_files(directory: Path, patterns=("*.wav", "*.flac")):
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(files, key=lambda p: p.as_posix())

def build_segment_index(files, project_root=Path(".")):
    """
    Returns one record per segment, not one record per file.
    Each record says: this segment lives in file X at segment_idx Y.
    """
    index = []

    for f in sorted(files, key=lambda p: p.as_posix()):
        audio = load_audio(str(f), sr=SAMPLE_RATE)
        segments = slice_audio(audio, SEGMENT_SAMPLES)
        rel_path = f.relative_to(project_root).as_posix()

        for seg_idx in range(len(segments)):
            index.append({
                "path": rel_path,
                "segment_idx": seg_idx,
            })

    return index

def make_pairs(clean_index, noise_index, split_name, copies_per_clean, snr_range, seed):
    rng = random.Random(seed)
    rows = []

    if len(noise_index) == 0:
        raise ValueError(f"No noise segments found for split '{split_name}'")

    for clean in clean_index:
        for _ in range(copies_per_clean):
            noise = noise_index[rng.randrange(len(noise_index))]
            snr_db = round(rng.uniform(*snr_range), 4)

            rows.append({
                "pair_id": f"{split_name}_{len(rows):06d}",
                "clean_path": clean["path"],
                "clean_segment_idx": clean["segment_idx"],
                "noise_path": noise["path"],
                "noise_segment_idx": noise["segment_idx"],
                "snr_db": snr_db,
            })

    return rows

def build_manifest(manifest_path: Path, split_seed=42):
    # ---------- Clean files ----------
    clean_files = sorted_files(CLEAN_DIR, ("*.flac", "*.wav"))

    # ---------- Noise files ----------
    seen_noise_files = []
    unseen_noise_files = []

    for group_name in sorted(NOISE_GROUPS.keys()):
        group_dir = NOISE_DIR / group_name
        if not group_dir.exists():
            continue

        for f in sorted_files(group_dir, ("*.wav",)):
            cat = extract_category_from_filename(f.name)
            if cat in SEEN_CATEGORIES:
                seen_noise_files.append(f)
            elif cat in UNSEEN_CATEGORIES:
                unseen_noise_files.append(f)

    # Optional extras
    bg_dir = NOISE_DIR / "background_music"
    if bg_dir.exists():
        seen_noise_files.extend(sorted_files(bg_dir, ("*.wav",)))

    mi_dir = NOISE_DIR / "music_interference"
    if mi_dir.exists():
        unseen_noise_files.extend(sorted_files(mi_dir, ("*.wav",)))

    # ---------- Deterministic clean split ----------
    split_rng = random.Random(split_seed)
    clean_files = clean_files[:]   # copy before shuffling
    split_rng.shuffle(clean_files)

    n_total = len(clean_files)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    clean_train = clean_files[:n_train]
    clean_val = clean_files[n_train:n_train + n_val]
    clean_test = clean_files[n_train + n_val:]   # use clean_test here

    # ---------- Segment indices ----------
    train_clean_index = build_segment_index(clean_train)
    val_clean_index = build_segment_index(clean_val)
    test_clean_index = build_segment_index(clean_test)

    seen_noise_index = build_segment_index(seen_noise_files)
    unseen_noise_index = build_segment_index(unseen_noise_files)

    # ---------- Fixed pair generation ----------
    manifest = {
        "version": 1,
        "audio_config": {
            "sample_rate": SAMPLE_RATE,
            "segment_samples": SEGMENT_SAMPLES,
            "segment_duration_sec": SEGMENT_DURATION,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "snr_min_db": SNR_MIN_DB,
            "snr_max_db": SNR_MAX_DB,
        },
        "split_seed": split_seed,
        "pair_seeds": {
            "train": 1001,
            "val_seen": 2001,
            "val_unseen": 2002,
            "test_seen": 3001,
            "test_unseen": 3002,
        },
        "splits": {
            "train": make_pairs(
                train_clean_index,
                seen_noise_index,
                split_name="train",
                copies_per_clean=5,   # same as your augmentation_factor=5
                snr_range=(SNR_MIN_DB, SNR_MAX_DB),
                seed=1001,
            ),
            "val_seen": make_pairs(
                val_clean_index,
                seen_noise_index,
                split_name="val_seen",
                copies_per_clean=1,
                snr_range=(SNR_MIN_DB, SNR_MAX_DB),
                seed=2001,
            ),
            "val_unseen": make_pairs(
                val_clean_index,
                unseen_noise_index,
                split_name="val_unseen",
                copies_per_clean=1,
                snr_range=(SNR_MIN_DB, SNR_MAX_DB),
                seed=2002,
            ),
            "test_seen": make_pairs(
                test_clean_index,
                seen_noise_index,
                split_name="test_seen",
                copies_per_clean=1,
                snr_range=(SNR_MIN_DB, SNR_MAX_DB),
                seed=3001,
            ),
            "test_unseen": make_pairs(
                test_clean_index,
                unseen_noise_index,
                split_name="test_unseen",
                copies_per_clean=1,
                snr_range=(SNR_MIN_DB, SNR_MAX_DB),
                seed=3002,
            ),
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to: {manifest_path}")
    for split_name, rows in manifest["splits"].items():
        print(f"{split_name:12s}: {len(rows)} pairs")

build_manifest(MANIFEST_PATH)