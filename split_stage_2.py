import pandas as pd
import numpy as np
from pathlib import Path

# ===== CONFIG =====
META_CSV = Path("./stage2_nodule_crops/Meta/meta_stage2.csv")
OUT_DIR = Path("./stage2_nodule_crops/Meta/splits")

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

SEED = 42

# If True: roughly balance splits by "patient has any malignant crop"
# (works even when patients contain both labels)
STRATIFY_BY_PATIENT_MALIGNANT_PRESENT = True
# ==================


def split_patients(patient_ids, rng, train_frac, val_frac):
    """Return (train_ids, val_ids, test_ids) from a list/array of patient_ids."""
    patient_ids = np.array(sorted(patient_ids))
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    # ensure sums don't exceed n
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    assert len(test_ids) == n_test
    return train_ids, val_ids, test_ids


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(META_CSV)

    # Basic checks
    required_cols = {"patient_id", "label", "file_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"meta CSV missing columns: {missing}")

    # Create a patient-level stratification label:
    # 1 if patient has ANY malignant crop, else 0.
    if STRATIFY_BY_PATIENT_MALIGNANT_PRESENT:
        patient_flag = (
            df.assign(is_malignant=(df["label"].str.lower() == "malignant").astype(int))
              .groupby("patient_id")["is_malignant"]
              .max()
        )
        malignant_present_ids = patient_flag[patient_flag == 1].index.values
        benign_only_ids = patient_flag[patient_flag == 0].index.values

        rng = np.random.default_rng(SEED)
        tr_m, va_m, te_m = split_patients(malignant_present_ids, rng, TRAIN_FRAC, VAL_FRAC)

        rng = np.random.default_rng(SEED + 1)
        tr_b, va_b, te_b = split_patients(benign_only_ids, rng, TRAIN_FRAC, VAL_FRAC)

        train_ids = np.concatenate([tr_m, tr_b])
        val_ids = np.concatenate([va_m, va_b])
        test_ids = np.concatenate([te_m, te_b])

        # shuffle within each set for neatness
        rng = np.random.default_rng(SEED + 2)
        rng.shuffle(train_ids); rng.shuffle(val_ids); rng.shuffle(test_ids)

    else:
        rng = np.random.default_rng(SEED)
        unique_patients = df["patient_id"].unique()
        train_ids, val_ids, test_ids = split_patients(unique_patients, rng, TRAIN_FRAC, VAL_FRAC)

    # Assign split to each row by patient_id
    split_map = {pid: "train" for pid in train_ids}
    split_map.update({pid: "val" for pid in val_ids})
    split_map.update({pid: "test" for pid in test_ids})

    df["split"] = df["patient_id"].map(split_map)

    # Safety check: no missing splits
    if df["split"].isna().any():
        missing_p = df.loc[df["split"].isna(), "patient_id"].unique()
        raise RuntimeError(f"Some patients were not assigned a split: {missing_p[:10]}")

    # Save split CSVs
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    # Optional: save file lists (useful for dataloaders)
    (OUT_DIR / "train_files.txt").write_text("\n".join(train_df["file_path"].tolist()) + "\n")
    (OUT_DIR / "val_files.txt").write_text("\n".join(val_df["file_path"].tolist()) + "\n")
    (OUT_DIR / "test_files.txt").write_text("\n".join(test_df["file_path"].tolist()) + "\n")

    # Print summary
    def summarize(name, d):
        n_pat = d["patient_id"].nunique()
        counts = d["label"].value_counts()
        print(f"\n{name}:")
        print(f"  patients: {n_pat}")
        print(f"  images:   {len(d)}")
        print(f"  label counts:\n{counts.to_string()}")

    # leakage check
    tr_pat = set(train_df["patient_id"].unique())
    va_pat = set(val_df["patient_id"].unique())
    te_pat = set(test_df["patient_id"].unique())
    assert tr_pat.isdisjoint(va_pat) and tr_pat.isdisjoint(te_pat) and va_pat.isdisjoint(te_pat)

    summarize("TRAIN", train_df)
    summarize("VAL", val_df)
    summarize("TEST", test_df)

    print(f"\nSaved splits to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()