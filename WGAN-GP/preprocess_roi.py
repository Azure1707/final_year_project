import numpy as np
import cv2

def png_uint16_to_hu(png_u16):
    x4095 = (png_u16.astype(np.float32) / 65535.0) * 4095.0
    return x4095 - 1024.0  
    
def apply_window_hu(img_hu, center = -600, width = 1500):
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)

def preprocess_roi_from_uint16png(img, size = 128):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize only if needed
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    hu = png_uint16_to_hu(img)
    x = apply_window_hu(hu, center=-600, width=1500) 
    x = x * 2.0 - 1.0   # for tanh GAN
    return x 