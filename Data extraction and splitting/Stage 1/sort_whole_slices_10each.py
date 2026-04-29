import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ROOT = Path("./data_png_wholeslice_unlabelled")
IN_CSV = ROOT / "Meta" / "meta_info_labeled.csv"

OUT_ROOT = ROOT / "Centre10"   # you can rename to "Nodule10" if you want
OUT_NODULE = OUT_ROOT / "nodule"
OUT_NO = OUT_ROOT / "no_nodule"
OUT_META = OUT_ROOT / "centre10_manifest.csv"
OUT_SKIP = OUT_ROOT / "centre10_skip.csv"

K = 10  # number of slices per patient

# For nodule patients with fewer than K nodule-labelled slices:
# - "skip" (recommended clean dataset)
# - "pad"  (fill remaining slots with nearest normal slices around nodule slices)
NUDGE_POLICY_IF_TOO_FEW_NODULE_SLICES = "skip"  # "skip" or "pad"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def patient_binary_from_slice_labels(labels: pd.Series):
    s = set(labels.dropna().astype(str))
    if ("benign" in s) or ("malignant" in s):
        return "nodule"
    if s <= {"normal"}:
        return "no_nodule"
    return None  # unknown/mixed -> skip

def pick_centre_k(sorted_rows: pd.DataFrame, k: int) -> pd.DataFrame | None:
    n = len(sorted_rows)
    if n < k:
        return None
    start = (n - k) // 2
    return sorted_rows.iloc[start:start + k]

def pick_nodule_k(g_sorted: pd.DataFrame, k: int) -> pd.DataFrame | None:
    """
    g_sorted: full patient rows sorted by z_index
    Returns k rows for nodule patients prioritizing benign/malignant slices.
    """
    nod = g_sorted[g_sorted["label"].isin(["benign", "malignant"])].copy()

    # If enough nodule slices, take centre k of the nodule subset
    if len(nod) >= k:
        nod = nod.sort_values("z_index").reset_index(drop=True)
        return pick_centre_k(nod, k)

    # Not enough nodule slices
    if NUDGE_POLICY_IF_TOO_FEW_NODULE_SLICES == "skip":
        return None

    # Pad with nearest normal slices around the nodule z's (keeps context)
    # Strategy:
    # 1) Start with all nodule slices
    # 2) Add closest non-nodule slices by distance to nearest nodule z until we reach k
    chosen = nod.copy()
    if len(chosen) == 0:
        return None

    need = k - len(chosen)
    others = g_sorted[~g_sorted.index.isin(chosen.index)].copy()

    nodule_z = chosen["z_index"].to_numpy()

    # distance of each candidate slice to nearest nodule slice
    cand_z = others["z_index"].to_numpy()
    # compute min absolute distance to any nodule z
    dists = [int(min(abs(z - nz) for nz in nodule_z)) for z in cand_z]
    others["dist_to_nodule"] = dists

    # prefer closest context; tie-breaker by z
    others = others.sort_values(["dist_to_nodule", "z_index"]).head(need)

    out = pd.concat([chosen, others], axis=0).sort_values("z_index").reset_index(drop=True)
    if len(out) < k:
        return None
    return out


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing: {IN_CSV}")

    ensure_dir(OUT_NODULE)
    ensure_dir(OUT_NO)

    df = pd.read_csv(IN_CSV)
    df["z_index"] = df["z_index"].astype(int)

    manifest_rows = []
    skip_rows = []

    for pid, g in tqdm(df.groupby("patient_id"), desc="Copy K slices per patient"):
        patient_cls = patient_binary_from_slice_labels(g["label"])
        if patient_cls is None:
            skip_rows.append([pid, "skipped_patient: unknown_or_mixed_labels"])
            continue

        g2 = g.sort_values("z_index").reset_index(drop=True)

        if patient_cls == "nodule":
            chosen = pick_nodule_k(g2, K)
            if chosen is None:
                # explain why
                n_nod = int((g2["label"].isin(["benign", "malignant"])).sum())
                skip_rows.append([pid, f"skipped_nodule_patient: only {n_nod} nodule slices (<{K})"])
                continue
        else:
            # no_nodule patients: centre K of whole run
            chosen = pick_centre_k(g2, K)
            if chosen is None:
                skip_rows.append([pid, f"skipped_no_nodule_patient: fewer_than_{K}_slices ({len(g2)})"])
                continue

        out_base = (OUT_NODULE if patient_cls == "nodule" else OUT_NO) / pid
        ensure_dir(out_base)

        copied = 0
        for _, r in chosen.iterrows():
            src = Path(r["file_path"])
            if not src.exists():
                skip_rows.append([pid, f"missing_png: {src}"])
                continue

            dst = out_base / src.name
            shutil.copy2(src, dst)
            copied += 1

            manifest_rows.append([
                pid,
                patient_cls,
                int(r["z_index"]),
                str(src),
                str(dst),
                str(r.get("label", "")),
                r.get("matched_mean_malignancy", "")
            ])

        if copied < K:
            skip_rows.append([pid, f"partial_copy: copied {copied}/{K}"])

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "patient_id",
            "patient_class",
            "z_index",
            "src_path",
            "dst_path",
            "slice_label",
            "matched_mean_malignancy"
        ],
    )
    manifest.to_csv(OUT_META, index=False)

    skip_df = pd.DataFrame(skip_rows, columns=["patient_id", "reason"])
    skip_df.to_csv(OUT_SKIP, index=False)

    print("\nSaved:", OUT_META)
    print("Saved:", OUT_SKIP)
    if len(manifest):
        print("\nPatient-class distribution (from copied slices):")
        print(manifest.groupby("patient_class")["patient_id"].nunique())

if __name__ == "__main__":
    main()