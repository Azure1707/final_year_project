import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

SPLITS_DIR = Path("./stage2_nodule_crops/Meta/splits")  # where train.csv/val.csv/test.csv are
OUT_ROOT = Path("./stage2_nodule_crops/DataSplit")      # output directory tree
COPY_LOG = OUT_ROOT / "copy_log.csv"

# If duplicates exist, skip if already copied
SKIP_EXISTING = True

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def norm_label(x: str) -> str:
    x = str(x).strip().lower()
    if x not in {"benign", "malignant"}:
        raise ValueError(f"Unexpected label: {x}")
    return x

def process_split(split_name: str):
    csv_path = SPLITS_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"file_path", "label"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain file_path and label columns")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        src = Path(r["file_path"])
        if not src.exists():
            rows.append([split_name, norm_label(r["label"]), str(src), "", "missing_src"])
            continue

        label = norm_label(r["label"])
        dst_dir = OUT_ROOT / split_name / label
        ensure_dir(dst_dir)

        # keep original filename; if collision, prefix with patient_id if available
        dst = dst_dir / src.name
        if dst.exists() and not SKIP_EXISTING:
            # make a unique name if needed
            pid = str(r["patient_id"]) if "patient_id" in df.columns else "unknownpid"
            dst = dst_dir / f"{pid}_{src.name}"

        if dst.exists() and SKIP_EXISTING:
            rows.append([split_name, label, str(src), str(dst), "skipped_exists"])
            continue

        shutil.copy2(src, dst)
        rows.append([split_name, label, str(src), str(dst), "copied"])

    return rows

def main():
    ensure_dir(OUT_ROOT)

    all_rows = []
    for split in ["train", "val", "test"]:
        all_rows.extend(process_split(split))

    log = pd.DataFrame(all_rows, columns=["split", "label", "src_path", "dst_path", "status"])
    log.to_csv(COPY_LOG, index=False)

    print("Saved copy log:", COPY_LOG)
    print("\nCounts (copied only):")
    print(log[log["status"] == "copied"].groupby(["split", "label"]).size())

if __name__ == "__main__":
    main()
