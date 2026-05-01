import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from preprocess_roi import preprocess_roi_from_uint16png

#Dataset class for loading data to model
class CTImageDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.image_paths = []
        valid_exts = (".png",)

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(root, file))

        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in: {root_dir}")

        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # load as uint16
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # apply preprocessing
        img = preprocess_roi_from_uint16png(img, size=self.image_size)

        # convert to tensor
        img = torch.from_numpy(img).unsqueeze(0)  # shape: (1, H, W)

        return img
