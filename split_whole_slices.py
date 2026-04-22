import random
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path("./data_png_wholeslice_unlabelled/Centre10")
OUT = Path("./data_png_wholeslice_unlabelled/Centre10_split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def split_patients(patients):
    random.shuffle(patients)

    n = len(patients)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = patients[:n_train]
    val = patients[n_train:n_train+n_val]
    test = patients[n_train+n_val:]

    return train, val, test

def copy_group(patients, label, split_name):
    for pid in patients:
        src = ROOT / label / pid
        dst = OUT / split_name / label / pid
        ensure_dir(dst.parent)
        shutil.copytree(src, dst)

def main():

    nodule_patients = [p.name for p in (ROOT/"nodule").iterdir() if p.is_dir()]
    normal_patients = [p.name for p in (ROOT/"no_nodule").iterdir() if p.is_dir()]

    print("Patients:")
    print("nodule:", len(nodule_patients))
    print("no_nodule:", len(normal_patients))

    # split each class separately (stratified)
    n_train, n_val, n_test = split_patients(nodule_patients)
    nn_train, nn_val, nn_test = split_patients(normal_patients)

    copy_group(n_train, "nodule", "train")
    copy_group(n_val, "nodule", "val")
    copy_group(n_test, "nodule", "test")

    copy_group(nn_train, "no_nodule", "train")
    copy_group(nn_val, "no_nodule", "val")
    copy_group(nn_test, "no_nodule", "test")

    print("\nSplit complete\n")

    print("Train:")
    print("  nodule:", len(n_train))
    print("  no_nodule:", len(nn_train))

    print("Val:")
    print("  nodule:", len(n_val))
    print("  no_nodule:", len(nn_val))

    print("Test:")
    print("  nodule:", len(n_test))
    print("  no_nodule:", len(nn_test))

if __name__ == "__main__":
    main()
