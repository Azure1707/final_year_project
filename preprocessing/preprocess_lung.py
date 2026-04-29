import cv2
import numpy as np
import matplotlib.pyplot as plt

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter, binary_fill_holes
from skimage import measure, morphology
from sklearn.cluster import KMeans


def png_to_hu(png_u16):
    x4095 = (png_u16.astype(np.float32) / 65535.0) * 4095.0
    return x4095 - 1024.0


def window(hu, center=-600, width=1500):
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(hu, lo, hi)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def resize_to_256(img, size=256):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def apply_lung_mask(img01, mask):
    out = img01.copy()
    out[mask == 0] = 0.0
    return out


def keep_largest_two_components(mask):
    labels = measure.label(mask)
    props = sorted(measure.regionprops(labels), key=lambda r: r.area, reverse=True)

    out = np.zeros_like(mask, dtype=np.uint8)
    for region in props[:2]:
        out[labels == region.label] = 1
    return out


def crop_to_mask_square(img, mask, margin=12):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return img, mask

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(img.shape[0], y1 + margin)
    x1 = min(img.shape[1], x1 + margin)

    h = y1 - y0
    w = x1 - x0
    s = max(h, w)

    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    half = s // 2

    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(img.shape[0], y0 + s)
    x1 = min(img.shape[1], x0 + s)

    y0 = max(0, y1 - s)
    x0 = max(0, x1 - s)

    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def segment_lung_mask(img01):
    #Strict lung-only mask.
    
    img = img01.copy()

    mean = np.mean(img)
    std = np.std(img) + 1e-8
    img = (img - mean) / std

    h, w = img.shape
    y0, y1 = int(0.2 * h), int(0.8 * h)
    x0, x1 = int(0.2 * w), int(0.8 * w)
    middle = img[y0:y1, x0:x1]

    middle_mean = np.mean(middle)
    max_val = np.max(img)
    min_val = np.min(img)

    img[img == max_val] = middle_mean
    img[img == min_val] = middle_mean

    img = median_filter(img, size=3)
    img = anisotropic_diffusion(img)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(middle.reshape(-1, 1))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)

    thresh_img = (img < threshold).astype(np.uint8)

    # mild cleanup
    mask = morphology.erosion(thresh_img, morphology.footprint_rectangle((2, 2)))
    mask = morphology.dilation(mask, morphology.footprint_rectangle((6, 6)))

    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    good_labels = []
    for prop in regions:
        minr, minc, maxr, maxc = prop.bbox
        height = maxr - minr
        width = maxc - minc

        if (
            height < 0.95 * h and
            width < 0.95 * w and
            minr > 0.05 * h and
            maxr < 0.98 * h and
            prop.area > 500
        ):
            good_labels.append(prop.label)

    mask = np.zeros(img.shape, dtype=np.uint8)
    for label_id in good_labels:
        mask[labels == label_id] = 1

    # keep only left + right lungs
    mask = keep_largest_two_components(mask)

    # remove bright non-lung structures inside mask
    mask[img01 > 0.80] = 0

    # keep only left + right lungs again after thresholding
    mask = keep_largest_two_components(mask)

    # fill small holes, but do NOT do heavy closing
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask = morphology.binary_opening(mask, morphology.disk(1)).astype(np.uint8)

    return mask

def remove_tiny_bright_speckles(x01, size_thresh  = 50, thr = 0.98):
    x = x01.copy()
    mask = (x > thr).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < size_thresh:
            x[labels == i] = thr
    return x



def preprocess_slice(png_u16, size=256):
    if png_u16.ndim == 3:
        png_u16 = cv2.cvtColor(png_u16, cv2.COLOR_BGR2GRAY)

    hu = png_to_hu(png_u16)
    x01 = window(hu, center=-600, width=1500)

    lung_mask = segment_lung_mask(x01)
    lung_only = apply_lung_mask(x01, lung_mask)

    lung_crop, _ = crop_to_mask_square(lung_only, lung_mask, margin=12)
    
    lung_crop = remove_tiny_bright_speckles(lung_crop, size_thresh=50, thr=0.98)
    
    lung_crop = resize_to_256(lung_crop, size=size).astype(np.float32)

    return lung_crop
