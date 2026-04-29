import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import pylidc as pl
from imageio.v2 import imwrite
from scipy.ndimage import label as cc_label  

DICOM_DIR = "./LIDC-IDRI"

OUT_ROOT = Path("./data_png_wholeslice_unlabelled")
OUT_ALL = OUT_ROOT / "All"
META_DIR = OUT_ROOT / "Meta"
OUT_META_CSV = META_DIR / "meta_info.csv"
OUT_SKIP_CSV = META_DIR / "skip_log.csv"

# lung HU band
LUNG_MIN_HU = -950
LUNG_MAX_HU = -500

MIN_LUNG_CC_FRACTION = 0.05 

#trim a few slices on both ends
TRIM_TOP = 5
TRIM_BOTTOM = 5

# Require at least this many slices in the run, otherwise skip the scan
MIN_GOOD_RUN_LEN = 50



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def hu_to_uint16_raw(img_hu: np.ndarray) -> np.ndarray:
     x = img_hu.astype(np.int32) + 1024
    x = np.clip(x, 0, 4095).astype(np.uint16)
    return (x.astype(np.uint32) * (65535 // 4095)).astype(np.uint16)


def save_png(path: Path, img_hu_2d: np.ndarray):
    arr = hu_to_uint16_raw(img_hu_2d)
    imwrite(str(path), arr)


def largest_true_run(mask: np.ndarray):
    
    if mask.size == 0 or not mask.any():
        return None

    best_start = best_end = -1
    best_len = 0
    start = None

    for i, v in enumerate(mask):
        if v and start is None:
            start = i

        if (not v or i == len(mask) - 1) and start is not None:
            end = i if (v and i == len(mask) - 1) else i - 1
            run_len = end - start + 1
            if run_len > best_len:
                best_len = run_len
                best_start, best_end = start, end
            start = None

    return best_start, best_end


def lung_cc_fraction(img_hu: np.ndarray) -> float:
    
    lung_mask = (img_hu > LUNG_MIN_HU) & (img_hu < LUNG_MAX_HU)

    labeled, n = cc_label(lung_mask)
    if n == 0:
        return 0.0

    sizes = np.bincount(labeled.ravel())
   
    largest = int(sizes[1:].max()) if sizes.size > 1 else 0
    return float(largest) / float(img_hu.size)


if __name__ == "__main__":
    for d in [OUT_ALL, META_DIR]:
        ensure_dir(d)

    patients = sorted([p for p in os.listdir(DICOM_DIR) if p.startswith("LIDC")])

    meta_rows, skip_rows = [], []

    for pid in tqdm(patients, desc="Whole-slice export (unlabelled, lung CC range only)"):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            skip_rows.append([pid, "missing_scan"])
            continue

        series_uid = getattr(scan, "series_instance_uid", "") or ""
        pid_short = pid[-4:]

        try:
            vol = scan.to_volume().astype(np.int16)  # (H, W, Z)
        except Exception as e:
            skip_rows.append([pid, f"volume_load_failed: {e}"])
            continue

        H, W, Z = vol.shape

        # Mark slices that contain enough contiguous aerated lung
        good = np.zeros(Z, dtype=bool)
        cc_fracs = np.zeros(Z, dtype=np.float32)

        for z in range(Z):
            f = lung_cc_fraction(vol[:, :, z])
            cc_fracs[z] = f
            good[z] = f >= MIN_LUNG_CC_FRACTION

        run = largest_true_run(good)
        if run is None:
            skip_rows.append([pid, f"no_lung_run_found (min_cc_frac={MIN_LUNG_CC_FRACTION})"])
            continue

        z0, z1 = run

        # Trim ends slightly
        z0 = min(max(z0 + TRIM_TOP, 0), Z - 1)
        z1 = max(min(z1 - TRIM_BOTTOM, Z - 1), 0)

        if z1 < z0:
            skip_rows.append([pid, "lung_run_collapsed_after_trim"])
            continue

        run_len = z1 - z0 + 1
        if run_len < MIN_GOOD_RUN_LEN:
            skip_rows.append([pid, f"lung_run_too_short ({run_len} slices)"])
            continue

        out_dir = OUT_ALL / pid
        ensure_dir(out_dir)

        saved = 0
        for z in range(z0, z1 + 1):
            fname = f"{pid_short}_slice{z:03d}.png"
            out_path = out_dir / fname
            save_png(out_path, vol[:, :, z])

            meta_rows.append([
                pid, series_uid, pid_short,
                "whole_slice",
                int(z),
                float(cc_fracs[z]),
                str(out_path)
            ])
            saved += 1

        if saved == 0:
            skip_rows.append([pid, "no_output_written"])
        else:
            skip_rows.append([pid, f"exported_lung_cc_range: {z0}-{z1} (saved {saved})"])

    meta = pd.DataFrame(meta_rows, columns=[
        "patient_id",
        "series_uid",
        "patient_short",
        "sample_type",
        "z_index",
        "lung_cc_frac",
        "file_path"
    ])
    meta.to_csv(OUT_META_CSV, index=False)

    skip_df = pd.DataFrame(skip_rows, columns=["patient_id", "reason"])
    skip_df.to_csv(OUT_SKIP_CSV, index=False)

    print(f"\nSaved meta → {OUT_META_CSV}")
    print(f"Saved skip log → {OUT_SKIP_CSV}")
    print("Total PNGs saved:", len(meta))
