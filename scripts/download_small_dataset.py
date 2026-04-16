"""
scripts/download_small_dataset.py

Python script to download a minimal dataset for ClarityNet.
Alternative to the bash script for users without ffmpeg/sox or on Windows.

Usage:
    python scripts/download_small_dataset.py
"""

import os
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm


# Configuration
DATA_DIR = Path("data/raw")
CLEAN_DIR = DATA_DIR / "clean_speech"
NOISE_DIR = DATA_DIR / "noise"
TMP_DIR = DATA_DIR / "tmp"

# URLs
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def download_file(url: str, dest: Path):
    """Download a file with progress bar."""
    print(f"Downloading from {url}...")
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = min(count * block_size / total_size, 1.0)
            bar_len = 50
            filled = int(bar_len * percent)
            bar = '=' * filled + '-' * (bar_len - filled)
            print(f'\r[{bar}] {percent*100:.1f}%', end='', flush=True)
    
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # New line after progress bar


def setup_directories():
    """Create necessary directory structure."""
    print("[1/5] Creating directory structure...")
    
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    (NOISE_DIR / "background_music").mkdir(parents=True, exist_ok=True)
    (NOISE_DIR / "domestic_mechanical").mkdir(parents=True, exist_ok=True)
    (NOISE_DIR / "natural_continuous").mkdir(parents=True, exist_ok=True)
    (NOISE_DIR / "transport_urban").mkdir(parents=True, exist_ok=True)
    (NOISE_DIR / "impulsive_events").mkdir(parents=True, exist_ok=True)
    (NOISE_DIR / "music_interference").mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created\n")


def download_librispeech():
    """Download and extract LibriSpeech dev-clean."""
    print("[2/5] Downloading clean speech (LibriSpeech dev-clean, ~330MB)...")
    
    archive_path = TMP_DIR / "dev-clean.tar.gz"
    
    if not archive_path.exists():
        download_file(LIBRISPEECH_URL, archive_path)
        print("✓ Download complete\n")
    else:
        print("✓ Already downloaded, skipping\n")
    
    return archive_path


def extract_and_convert_librispeech(archive_path: Path):
    """Extract LibriSpeech and copy FLAC files (conversion note below)."""
    print("[3/5] Extracting clean speech files...")
    
    # Extract
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(TMP_DIR)
    
    print("✓ Extracted")
    
    # Find all FLAC files
    librispeech_dir = TMP_DIR / "LibriSpeech" / "dev-clean"
    flac_files = list(librispeech_dir.rglob("*.flac"))
    
    print(f"Found {len(flac_files)} FLAC files")
    
    # Copy a subset (200 files) - NOTE: these are still FLAC
    # librosa.load() can handle FLAC directly, so no conversion needed
    max_files = 200
    
    print(f"Copying {max_files} files to clean_speech/...")
    for i, flac_file in enumerate(flac_files[:max_files]):
        if i % 20 == 0:
            print(f"  Copied {i}/{max_files} files...")
        
        dest = CLEAN_DIR / flac_file.name
        shutil.copy(flac_file, dest)
    
    print(f"✓ Copied {min(max_files, len(flac_files))} clean speech files")
    print("  Note: Files are in FLAC format. librosa.load() will handle conversion automatically.\n")


def download_esc50():
    """Download ESC-50 noise dataset."""
    print("[4/5] Downloading noise samples (ESC-50, ~600MB)...")
    
    archive_path = TMP_DIR / "ESC-50-master.zip"
    
    if not archive_path.exists():
        download_file(ESC50_URL, archive_path)
        print("✓ Download complete\n")
    else:
        print("✓ Already downloaded, skipping\n")
    
    return archive_path


def organize_noise_files(archive_path: Path):
    """Extract and organize ESC-50 files into noise categories."""
    print("[5/5] Organizing noise files by category...")
    
    # Extract
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(TMP_DIR)
    
    esc50_audio = TMP_DIR / "ESC-50-master" / "audio"
    
    if not esc50_audio.exists():
        print(f"ERROR: Expected audio directory not found at {esc50_audio}")
        return
    
    all_audio = list(esc50_audio.glob("*.wav"))
    print(f"Found {len(all_audio)} audio files in ESC-50")
    
    # ESC-50 filename format: fold-clip_id-take-category.wav
    # Category numbers: 0-49 (5 per class, 50 classes total)
    # We'll map based on category number in filename
    
    # Category mappings (ESC-50 category numbers)
    # Format: our_category -> list of ESC-50 category numbers
    category_map = {
    # SEEN (Training) - Continuous, stationary patterns
    "natural_continuous": [10, 11, 12, 15, 16, 17, 19],  # rain, sea waves, fire, water drops, wind, pouring water, thunderstorm
    "domestic_mechanical": [35, 36, 38],                     # washing machine, vacuum cleaner, clock tick
    
    # UNSEEN (Testing) - Different acoustic characteristics
    "transport_urban": [40, 42, 43, 44, 45, 47],         # helicopter, siren, car horn, engine, train, airplane
    "impulsive_events": [30, 34, 39, 46, 48, 49],        # door knock, can opening, glass breaking, church bells, fireworks, hand saw
    }
    
    # Organize files
    for category, esc_categories in category_map.items():
        dest_dir = NOISE_DIR / category
        count = 0
        
        for audio_file in all_audio:
            # Extract category from filename: fold-clip_id-take-category.wav
            parts = audio_file.stem.split('-')
            if len(parts) >= 4:
                try:
                    file_category = int(parts[3])
                    if file_category in esc_categories:
                        shutil.copy(audio_file, dest_dir / audio_file.name)
                        count += 1
                        if count >= 15:  # Limit to 15 files per category
                            break
                except ValueError:
                    continue
        
        print(f"  {category}: {count} files")
    
    print("\n✓ Noise files organized by category\n")


def print_music_instructions():
    """Print instructions for adding music files."""
    print("=" * 50)
    print("IMPORTANT: Music Interference")
    print("=" * 50)
    print()
    print("ESC-50 has limited music samples. For 'background_music' and")
    print("'music_interference' categories, you'll need to add your own files:")
    print()
    print("Option 1: Use royalty-free music from:")
    print("  - https://freemusicarchive.org")
    print("  - https://incompetech.com")
    print("  - YouTube Audio Library")
    print()
    print("Option 2: Record clips from Spotify/Apple Music (personal use only)")
    print()
    print("Save files to:")
    print("  - data/raw/noise/background_music/  (classical, jazz)")
    print("  - data/raw/noise/music_interference/ (rock, electronic, pop)")
    print()
    print("Recommended: 15-20 files per category, 10-30 seconds each")
    print()


def cleanup():
    """Remove temporary files."""
    print("[Cleanup] Removing temporary files...")
    shutil.rmtree(TMP_DIR)
    print("✓ Cleanup complete\n")


def print_summary():
    """Print dataset summary."""
    print("=" * 50)
    print("Dataset Download Complete!")
    print("=" * 50)
    print()
    print("Summary:")
    print(f"  Clean speech:                     {len(list(CLEAN_DIR.glob('*')))} files in {CLEAN_DIR}")
    print(f"  Natural Continuous noise:         {len(list((NOISE_DIR / 'natural_continuous').glob('*.wav')))} files")
    print(f"  Domenstic Mechanical:             {len(list((NOISE_DIR / 'domestic_mechanical').glob('*.wav')))} files")
    print(f"  Transport Urban:                  {len(list((NOISE_DIR / 'transport_urban').glob('*.wav')))} files")
    print(f"  Impulsive Events:                 {len(list((NOISE_DIR / 'impulsive_events').glob('*.wav')))} files")
    print()
    print("Next steps:")
    print("  1. Add music files to background_music/ and music_interference/")
    print("  2. Run: python scripts/prepare_dataset.py")
    print("  3. Start training!")
    print()


def main():
    """Main download and setup process."""
    print("=" * 50)
    print("ClarityNet - Small Dataset Setup")
    print("=" * 50)
    print()
    
    try:
        # Step 1: Setup directories
        setup_directories()
        
        # Step 2-3: Download and process LibriSpeech
        librispeech_archive = download_librispeech()
        extract_and_convert_librispeech(librispeech_archive)
        
        # Step 4-5: Download and organize ESC-50
        esc50_archive = download_esc50()
        organize_noise_files(esc50_archive)
        
        # Instructions and cleanup
        print_music_instructions()
        cleanup()
        print_summary()
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("You can re-run this script to resume.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
