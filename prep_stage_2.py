import os
from pathlib import Path
from statistics import median_high

import numpy as np
import pandas as pd
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
from imageio.v2 import imwrite

DICOM_DIR = "./LIDC-IDRI"

OUT_ROOT = Path("./stage2_nodule_crops")
OUT_BENIGN = OUT_ROOT / "Benign"
OUT_MALIGN = OUT_ROOT / "Malignant"
META_DIR = OUT_ROOT / "Meta"
OUT_META_CSV = META_DIR / "meta_stage2.csv"
OUT_SKIP_CSV = META_DIR / "skip_log_stage2.csv"

DROP_AMBIGUOUS = True

# Use median_high across radiologists (as in your reference code)
USE_MEDIAN_HIGH = True

# consensus() parameters
CONFIDENCE_LEVEL = 0.0  # start permissive; tighten later
PADDING_PIXELS = 12
PADDING = [(PADDING_PIXELS, PADDING_PIXELS),
           (PADDING_PIXELS, PADDING_PIXELS),
           (0, 0)]

# slice-level presence threshold
MASK_THRESHOLD = 1

# crop settings
FIXED_CROP_SIZE = 128
SLICE_PADDING = 8

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cluster_malignancy_label(cluster):
    """
    Returns (score, label_str) where label_str in {'benign','malignant','ambiguous','missing'}.
    Uses median_high (recommended) or median depending on USE_MEDIAN_HIGH.
    """
    scores = [a.malignancy for a in cluster if a.malignancy is not None]
    if not scores:
        return None, "missing"

    score = float(median_high(scores)) if USE_MEDIAN_HIGH else float(np.median(scores))

    if score > 3:
        return score, "malignant"
    if score < 3:
        return score, "benign"
    return score, "ambiguous"


def hu_to_uint16_raw(img_hu: np.ndarray) -> np.ndarray:
    x = img_hu.astype(np.int32) + 1024
    x = np.clip(x, 0, 4095).astype(np.uint16)
    return (x.astype(np.uint32) * (65535 // 4095)).astype(np.uint16)


def save_png(path: Path, img_hu_2d: np.ndarray):
    imwrite(str(path), hu_to_uint16_raw(img_hu_2d))


def safe_crop_2d(img2d: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    h, w = img2d.shape
    half = size // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bot = max(0, y1 - h)
    pad_right = max(0, x1 - w)

    if pad_top or pad_left or pad_bot or pad_right:
        img2d = np.pad(img2d, ((pad_top, pad_bot), (pad_left, pad_right)), mode="edge")
        y0 += pad_top; y1 += pad_top
        x0 += pad_left; x1 += pad_left

    return img2d[y0:y1, x0:x1]


def center_from_mask(mask2d: np.ndarray, pad: int = 0):
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + pad
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + pad
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    return cy, cx, y0, y1, x0, x1


if __name__ == "__main__":
    for d in [OUT_BENIGN, OUT_MALIGN, META_DIR]:
        ensure_dir(d)

    patients = sorted([p for p in os.listdir(DICOM_DIR) if p.startswith("LIDC")])

    meta_rows, skip_rows = [], []

    for pid in tqdm(patients, desc="Stage 2: GT nodule crops (benign/malignant; drop 3)"):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            skip_rows.append([pid, "missing_scan"])
            continue

        series_uid = getattr(scan, "series_instance_uid", "") or ""
        pid_short = pid[-4:]

        try:
            vol = scan.to_volume().astype(np.int16)  # (H,W,Z)
        except Exception as e:
            skip_rows.append([pid, f"volume_load_failed: {e}"])
            continue

        H, W, Z = vol.shape
        clusters = scan.cluster_annotations() or []

        wrote_any_for_patient = False

        for c_idx, cluster in enumerate(clusters):
            if not cluster:
                continue

            score, lab = cluster_malignancy_label(cluster)

            if lab == "missing":
                skip_rows.append([pid, f"cluster_{c_idx}: missing_malignancy_scores"])
                continue

            if lab == "ambiguous" and DROP_AMBIGUOUS:
                skip_rows.append([pid, f"cluster_{c_idx}: malignancy==3 (ambiguous) dropped"])
                continue

            if lab not in {"benign", "malignant"}:
                skip_rows.append([pid, f"cluster_{c_idx}: unexpected_label={lab}"])
                continue

            out_base = OUT_BENIGN if lab == "benign" else OUT_MALIGN
            out_dir = out_base / pid
            ensure_dir(out_dir)

            try:
                mask3d, cbbox, _ = consensus(cluster, CONFIDENCE_LEVEL, PADDING)
            except Exception as e:
                skip_rows.append([pid, f"cluster_{c_idx}: consensus_failed {e}"])
                continue

            # cbbox = (yslice, xslice, zslice)   <-- each is a slice object
            yslc, xslc, zslc = cbbox
            y_start = int(yslc.start or 0)
            x_start = int(xslc.start or 0)
            z_start = int(zslc.start or 0)

            saved_this_cluster = 0

            for local_z in range(mask3d.shape[2]):
                mask2d = mask3d[:, :, local_z].astype(bool)
                if int(mask2d.sum()) < MASK_THRESHOLD:
                    continue

                global_z = int(np.clip(z_start + local_z, 0, Z - 1))
                img_hu = vol[:, :, global_z]

                info = center_from_mask(mask2d, pad=SLICE_PADDING)
                if info is None:
                    continue
                cy_roi, cx_roi, y0r, y1r, x0r, x1r = info

                # Map ROI coords -> full image coords using cbbox starts
                cy = int(np.clip(y_start + cy_roi, 0, H - 1))
                cx = int(np.clip(x_start + cx_roi, 0, W - 1))

                patch = safe_crop_2d(img_hu, cy, cx, FIXED_CROP_SIZE)

                fname = f"{pid_short}_N{c_idx:03d}_slice{global_z:03d}.png"
                out_path = out_dir / fname
                save_png(out_path, patch)

                meta_rows.append([
                    pid, series_uid, pid_short,
                    int(c_idx),
                    float(score),
                    lab,
                    int(global_z),
                    int(local_z),
                    int(cy), int(cx),
                    int(patch.shape[0]), int(patch.shape[1]),
                    str(out_path)
                ])
                saved_this_cluster += 1
                wrote_any_for_patient = True

            if saved_this_cluster == 0:
                skip_rows.append([pid, f"cluster_{c_idx}: no_crops_saved (mask too small / thresholds)"])

        if not wrote_any_for_patient:
            skip_rows.append([pid, "no_output_written (no benign/malignant nodules saved)"])

    meta = pd.DataFrame(meta_rows, columns=[
        "patient_id",
        "series_uid",
        "patient_short",
        "cluster_index",
        "malignancy_score_median_high",
        "label",          # benign | malignant
        "global_z",
        "local_z",
        "center_y",
        "center_x",
        "height",
        "width",
        "file_path"
    ])
    meta.to_csv(OUT_META_CSV, index=False)

    skip_df = pd.DataFrame(skip_rows, columns=["patient_id", "reason"])
    skip_df.to_csv(OUT_SKIP_CSV, index=False)

    print(f"\nSaved meta → {OUT_META_CSV}")
    print(f"Saved skip log → {OUT_SKIP_CSV}")
    print("Total crops saved:", len(meta))
