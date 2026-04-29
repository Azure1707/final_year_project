import numpy as np
from PIL import Image
import pydicom

#Scale image to 0-255 for display
def normalise_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = img - img.min()

    if img.max() > 0:
        img = img / img.max()

    return (img * 255).astype(np.uint8)

#convert HU to 16-bit range
def hu_to_uint16(img_hu: np.ndarray) -> np.ndarray:
    img = img_hu.astype(np.int32) + 1024
    img = np.clip(img, 0, 4095).astype(np.uint16)
    return (img.astype(np.uint32) * (65535 // 4095)).astype(np.uint16)

#load uploaded image
def load_uploaded_ct_image(uploaded_file):
    file_name = uploaded_file.name.lower()
    #dicom file handling
    if file_name.endswith((".dcm", ".dicom")):
        ds = pydicom.dcmread(uploaded_file)

        img = ds.pixel_array.astype(np.int16)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))

        img_hu = img.astype(np.float32) * slope + intercept
        return hu_to_uint16(img_hu)

    pil_img = Image.open(uploaded_file)
    return np.array(pil_img)

#crop input image using box from cropper
def crop_raw_from_box(raw_img: np.ndarray, box: dict) -> np.ndarray:
    left = int(box["left"])
    top = int(box["top"])
    width = int(box["width"])
    height = int(box["height"])

    return raw_img[top:top + height, left:left + width]

#show crop in interface
def crop_display_from_box(display_img: Image.Image, box: dict) -> Image.Image:
    left = int(box["left"])
    top = int(box["top"])
    width = int(box["width"])
    height = int(box["height"])

    return display_img.crop((left, top, left + width, top + height))

