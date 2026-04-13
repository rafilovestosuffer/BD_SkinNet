import os
import zipfile
import shutil
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASETS = {
    "SkinDiseaseBD": {
        "doi": "https://data.mendeley.com/datasets/9ggd3shdr7/2",
        "zip_url": "https://data.mendeley.com/public-files/datasets/9ggd3shdr7/files/download",
        "folder_map": {
            "Dermatitis":  "contact_dermatitis",
            "Eczema":      "eczema",
            "Scabies":     "scabies",
            "Tinea":       "tinea",
            "Vitiligo":    "vitiligo",
        },
    },
    "SkinDisNet": {
        "doi": "https://data.mendeley.com/datasets/yj3md44hxg/2",
        "zip_url": "https://data.mendeley.com/public-files/datasets/yj3md44hxg/files/download",
        "folder_map": {
            "Atopic Dermatitis":     "atopic_dermatitis",
            "Contact Dermatitis":    "contact_dermatitis",
            "Eczema":                "eczema",
            "Scabies":               "scabies",
            "Seborrheic Dermatitis": "seborrheic_dermatitis",
            "Tinea Corporis":        "tinea",
        },
    },
}

CLASS_NAMES = [
    "atopic_dermatitis",
    "contact_dermatitis",
    "eczema",
    "scabies",
    "seborrheic_dermatitis",
    "tinea",
    "vitiligo",
]

RAW_DIR  = Path("data/raw")
DATA_DIR = Path("data")
SPLITS   = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED     = 42


def download_zip(url, dest_path):
    print(f"  Downloading from {url} ...")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def merge_raw():
    merged = RAW_DIR / "merged"
    for cls in CLASS_NAMES:
        (merged / cls).mkdir(parents=True, exist_ok=True)

    for name, cfg in DATASETS.items():
        extract_dir = RAW_DIR / name
        if not extract_dir.exists():
            print(f"\n[!] {name} not found at {extract_dir}")
            print(f"    Please download it manually from: {cfg['doi']}")
            print(f"    Extract it to: {extract_dir}/")
            continue

        print(f"\nMerging {name} ...")
        for src_folder, cls_name in cfg["folder_map"].items():
            src = extract_dir / src_folder
            dst = merged / cls_name
            if not src.exists():
                print(f"  [!] Folder not found: {src}")
                continue
            images = list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + list(src.glob("*.png"))
            for img in images:
                shutil.copy2(img, dst / img.name)
            print(f"  {src_folder} -> {cls_name} ({len(images)} images)")

    return merged


def split_data(merged_dir):
    print("\nSplitting into train/val/test ...")
    for split in SPLITS:
        for cls in CLASS_NAMES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASS_NAMES:
        images = list((merged_dir / cls).glob("*.*"))
        if not images:
            print(f"  [!] No images found for class: {cls}")
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=SEED)
        val_imgs, test_imgs   = train_test_split(temp_imgs, test_size=0.50, random_state=SEED)

        for imgs, split in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
            for img in imgs:
                shutil.copy2(img, DATA_DIR / split / cls / img.name)

        print(f"  {cls}: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")


def main():
    print("=" * 60)
    print("  BD-SkinNet — Dataset Setup")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in DATASETS.items():
        dest = RAW_DIR / f"{name}.zip"
        extract_dir = RAW_DIR / name
        if extract_dir.exists():
            print(f"\n[✓] {name} already extracted, skipping download.")
            continue
        print(f"\n[{name}]")
        ok = download_zip(cfg["zip_url"], dest)
        if ok:
            print(f"  Extracting ...")
            with zipfile.ZipFile(dest, "r") as z:
                z.extractall(extract_dir)
            dest.unlink()
        else:
            print(f"\n  Mendeley may require a browser login to download.")
            print(f"  Manual steps:")
            print(f"    1. Visit: {cfg['doi']}")
            print(f"    2. Download the zip file")
            print(f"    3. Extract it to: {RAW_DIR / name}/")
            print(f"    4. Re-run this script")

    merged_dir = merge_raw()
    split_data(merged_dir)

    print("\n" + "=" * 60)
    print("  Done! Dataset organized at data/train, data/val, data/test")
    print("=" * 60)


if __name__ == "__main__":
    main()
