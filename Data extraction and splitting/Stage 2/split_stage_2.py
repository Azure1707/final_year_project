import pandas as pd
import numpy as np
from pathlib import Path

meta_csv = Path("./stage2_nodule_crops/Meta/meta_stage2.csv")
out_dir = Path("./stage2_nodule_crops/Meta/splits")

train = 0.8
val = 0.1

np.random.seed(42)


def split(ids):
    ids = list(ids)
    np.random.shuffle(ids)

    n = len(ids)
    n_train = int(n * train)
    n_val = int(n * val)

    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]

    return train, val, test


df = pd.read_csv(meta_csv)

# group patients by whether they have malignant or not
df["is_malignant"] = (df["label"].str.lower() == "malignant").astype(int)
patient_flag = df.groupby("patient_id")["is_malignant"].max()

malignant_ids = patient_flag[patient_flag == 1].index
benign_ids = patient_flag[patient_flag == 0].index

# split separately
tr_m, va_m, te_m = split(malignant_ids)
tr_b, va_b, te_b = split(benign_ids)

train_ids = list(tr_m) + list(tr_b)
val_ids = list(va_m) + list(va_b)
test_ids = list(te_m) + list(te_b)

# assign split
split_map = {}

for p in train_ids:
    split_map[p] = "train"
for p in val_ids:
    split_map[p] = "val"
for p in test_ids:
    split_map[p] = "test"

df["split"] = df["patient_id"].map(split_map)

# save
out_dir.mkdir(parents=True, exist_ok=True)

df[df["split"] == "train"].to_csv(out_dir / "train.csv", index=False)
df[df["split"] == "val"].to_csv(out_dir / "val.csv", index=False)
df[df["split"] == "test"].to_csv(out_dir / "test.csv", index=False)

print("done")