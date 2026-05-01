import numpy as np
import pandas as pd
from pathlib import Path
from statistics import mean
from tqdm import tqdm

import pylidc as pl
from pylidc.utils import consensus

STAGE1_ROOT = Path("./data_png_wholeslice_unlabelled")
IN_META_CSV = STAGE1_ROOT / "Meta" / "meta_info.csv"

# Output: same rows, but with labels added (does NOT create images)
OUT_META_CSV = STAGE1_ROOT / "Meta" / "meta_info_labeled.csv"
OUT_SKIP_CSV = STAGE1_ROOT / "Meta" / "skip_stage1_labeling.csv"

# Slice label rule:
# default = normal
# if any nodule overlaps slice -> benign/malignant (malignant overrides benign)

MALIGNANT_THR = 4.0
BENIGN_THR = 2.0

# Keep ONLY 3 labels by mapping 3-ish to one side
MAP_INDETERMINATE = True
INDETERMINATE_SPLIT = 3.0  # <=3 benign, >3 malignant

# consensus() parameters
CONFIDENCE_LEVEL = 0.0
PADDING_PIXELS = 12
PADDING = [(PADDING_PIXELS, PADDING_PIXELS),
           (PADDING_PIXELS, PADDING_PIXELS),
           (0, 0)]

# Nodule-present on a slice if mask pixels >= this
MASK_THRESHOLD = 1


def cluster_mean_malignancy(cluster) -> float | None:
    scores = [a.malignancy for a in cluster if a.malignancy is not None]
    return None if not scores else float(mean(scores))


def label_from_mean(m: float) -> str:
    if m >= MALIGNANT_THR:
        return "malignant"
    if m <= BENIGN_THR:
        return "benign"
    if MAP_INDETERMINATE:
        return "malignant" if m > INDETERMINATE_SPLIT else "benign"
    return "indeterminate"


def main():
    if not IN_META_CSV.exists():
        raise FileNotFoundError(f"Missing input meta CSV: {IN_META_CSV}")

    df = pd.read_csv(IN_META_CSV)

    # Ensure types
    df["z_index"] = df["z_index"].astype(int)

    # We'll compute labels only for slices that exist in df (already extracted)
    labels = ["unknown"] * len(df)
    max_malig = [np.nan] * len(df)
    notes = [""] * len(df)

    skip_rows = []

    # Group by patient for speed
    for pid, g in tqdm(df.groupby("patient_id"), desc="Labelling already-extracted slices"):
        idxs = g.index.to_numpy()
        exported_z = set(int(z) for z in g["z_index"].tolist())

        # default all to normal
        z_to_label = {z: "normal" for z in exported_z}
        z_to_m = {z: np.nan for z in exported_z}

        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            for i in idxs:
                labels[i] = "unknown"
                notes[i] = "missing_scan"
            continue

        clusters = scan.cluster_annotations() or []

        for c_idx, cluster in enumerate(clusters):
            if not cluster:
                continue

            m = cluster_mean_malignancy(cluster)
            if m is None:
                skip_rows.append([pid, f"cluster_{c_idx}: missing_malignancy"])
                continue

            lab = label_from_mean(m)
            if lab == "indeterminate":
                # only if MAP_INDETERMINATE=False
                skip_rows.append([pid, f"cluster_{c_idx}: indeterminate_skipped"])
                continue

            try:
                mask3d, cbbox, _ = consensus(cluster, CONFIDENCE_LEVEL, PADDING)
            except Exception as e:
                skip_rows.append([pid, f"cluster_{c_idx}: consensus_failed {e}"])
                continue

            # cbbox is (yslice, xslice, zslice)
            _, _, zslc = cbbox
            z_start = int(zslc.start or 0)

            for local_z in range(mask3d.shape[2]):
                if int(mask3d[:, :, local_z].sum()) < MASK_THRESHOLD:
                    continue
                global_z = z_start + local_z
                if global_z not in exported_z:
                    continue

                # priority: malignant overrides benign
                if lab == "malignant":
                    z_to_label[global_z] = "malignant"
                    z_to_m[global_z] = float(m)
                elif lab == "benign" and z_to_label[global_z] == "normal":
                    z_to_label[global_z] = "benign"
                    z_to_m[global_z] = float(m)

        # write results back into the dataframe-row order
        for i, z in zip(idxs, g["z_index"].tolist()):
            labels[i] = z_to_label.get(int(z), "unknown")
            max_malig[i] = z_to_m.get(int(z), np.nan)

    out = df.copy()
    out["label"] = labels                 # normal | benign | malignant | unknown
    out["matched_mean_malignancy"] = max_malig
    out["notes"] = notes

    out.to_csv(OUT_META_CSV, index=False)

    skip_df = pd.DataFrame(skip_rows, columns=["patient_id", "reason"])
    skip_df.to_csv(OUT_SKIP_CSV, index=False)

    print(f"\nSaved labelled meta → {OUT_META_CSV}")
    print(f"Saved skip log → {OUT_SKIP_CSV}")
    print("\nLabel distribution:\n", out["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
