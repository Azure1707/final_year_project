import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import numpy as np
import matplotlib.pyplot as plt
from preprocess_lung import preprocess_lung_only

import torch
import torch.nn as nn

from torchvision import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGE1_MODEL_PATH = "Sample/resnet101_ct_best.pth"

STAGE1_CLASSES = ["No suspicious nodule", "Suspicious nodule detected"]

if "uploaded_image_display" not in st.session_state:
    st.session_state.uploaded_image_display = None

if "uploaded_image_raw" not in st.session_state:
    st.session_state.uploaded_image_raw = None

if "stage1_result" not in st.session_state:
    st.session_state.stage1_result = None


st.set_page_config(
    page_title="Lung Tumour Decision-Support Dashboard",
    layout="wide"
)

st.title("Lung Tumour Clinician Decision-Support Dashboard")
st.caption(
    "Two-stage CT image analysis system for suspicious nodule detection "
    "and benign/malignant ROI classification."
)

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "Stage 1"

if "enable_crop" not in st.session_state:
    st.session_state.enable_crop = False

if "cropped_img" not in st.session_state:
    st.session_state.cropped_img = None

if "stage1_result" not in st.session_state:
    st.session_state.stage1_result = None

if "stage2_result" not in st.session_state:
    st.session_state.stage2_result = None


# =========================================================
# NAVIGATION HELPERS
# =========================================================
def go_stage1():
    st.session_state.page = "Stage 1"


def go_stage2():
    st.session_state.page = "Stage 2"


# =========================================================
# PREPROCESSING FUNCTIONS
# Replace with your real preprocessing if needed
# =========================================================
def normalise_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img


def preprocess_stage1(img: np.ndarray) -> torch.Tensor:
    x = preprocess_lung_only(img, size=256, normalize_to_minus1_1=False)
    x = np.stack([x, x, x], axis=0)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    x = (x - mean) / std

    x = np.expand_dims(x, axis=0)
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)



def preprocess_stage2(roi: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    """
    Placeholder preprocessing for stage 2 model.
    Replace this with your real preprocessing pipeline.
    """
    roi_img = Image.fromarray(roi).resize(target_size)
    x = np.array(roi_img).astype(np.float32) / 255.0
    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    return x


# =========================================================
# MODEL PLACEHOLDERS
# Replace with your actual model inference functions
# =========================================================

@st.cache_resource
def load_stage1_model():
    model = models.resnet101(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, len(STAGE1_CLASSES))
    )

    state_dict = torch.load(
        STAGE1_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def run_stage1_model(img: np.ndarray):
    model = load_stage1_model()
    x = preprocess_stage1(img)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    label = STAGE1_CLASSES[pred_idx]
    confidence = float(probs[pred_idx].item())
    return label, confidence



def run_stage2_model(roi: np.ndarray):
    """
    Replace this with your real stage 2 malignancy classifier.
    Expected output example:
    - label
    - confidence
    """
    tensor = preprocess_stage2(roi)
    _ = tensor
    label = "Malignant"
    confidence = 0.73
    return label, confidence


# =========================================================
# GRAD-CAM PLACEHOLDER
# Replace with your actual Grad-CAM
# =========================================================
def dummy_gradcam(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    cx, cy = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            heatmap[y, x] = np.exp(-dist / (min(h, w) / 4))

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def show_overlay(image: np.ndarray):
    heatmap = dummy_gradcam(image)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
    fig.patch.set_alpha(0)

    ax.imshow(image, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=0.35)

    ax.axis("off")
    plt.tight_layout(pad=0)

    st.pyplot(fig, transparent=True)

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload CT slice image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    raw_array = np.array(pil_img)

    display_array = normalise_for_display(raw_array)

    st.session_state.uploaded_image_raw = raw_array
    st.session_state.uploaded_image_display = Image.fromarray(display_array)

    st.session_state.stage1_result = None
    st.session_state.stage2_result = None
    st.session_state.cropped_img = None

# =========================================================
# MAIN UI
# =========================================================
if st.session_state.uploaded_image_display is None:
    st.info("Upload a CT slice image to begin.")
    st.stop()

image = st.session_state.uploaded_image_display
raw_image = st.session_state.uploaded_image_raw

# Left control panel + main content
left_col, right_col = st.columns([1, 4])

with left_col:
    st.markdown("### Navigation")
    selected_page = st.radio(
        label="",
        options=["Stage 1", "Stage 2"],
        index=0 if st.session_state.page == "Stage 1" else 1,
        key="nav_radio"
    )
    st.session_state.page = selected_page

    if st.session_state.page == "Stage 1":
        st.markdown("### Controls")
        st.toggle("Enable ROI Cropping", key="enable_crop")

    st.markdown("---")
    st.markdown("### Notes")
    st.write("This system is a prototype decision-support tool.")
    st.write("Final diagnosis should remain under clinician supervision.")

with right_col:
    # =====================================================
    # STAGE 1
    # =====================================================
    if st.session_state.page == "Stage 1":
        st.header("Stage 1: Suspicious Nodule Detection")

        if st.session_state.enable_crop:
            img_col, roi_col = st.columns([3, 2])

            with img_col:
                st.subheader("Original Image")
                cropped_img = st_cropper(
                    image,
                    realtime_update=True,
                    box_color="#FF0000",
                    aspect_ratio=(1, 1),
                    return_type="image"
                )

                if st.button("Run Stage 1 Detection"):
                    stage1_label, stage1_conf = run_stage1_model(raw_image)
                    st.session_state.stage1_result = (stage1_label, stage1_conf)

                if st.session_state.stage1_result is not None:
                    label, conf = st.session_state.stage1_result
                    st.success(f"Stage 1 Result: {label}")
                    st.write(f"Confidence: **{conf:.2%}**")
                    show_overlay(np.array(image))

            with roi_col:
                st.subheader("ROI Preview")
                st.image(cropped_img, caption="Selected ROI", width=280)

                st.write("ROI size:", cropped_img.size)

                if st.button("Crop ROI and go to Stage 2"):
                    st.session_state.cropped_img = cropped_img
                    st.session_state.stage2_result = None
                    go_stage2()
                    st.rerun()

        else:
            col_a, col_b = st.columns([3, 2])

            with col_a:
                st.subheader("Original Image")
                st.image(image, width=500)

                if st.button("Run Stage 1 Detection"):
                    stage1_label, stage1_conf = run_stage1_model(raw_image)
                    st.session_state.stage1_result = (stage1_label, stage1_conf)

                if st.session_state.stage1_result is not None:
                    label, conf = st.session_state.stage1_result
                    st.success(f"Stage 1 Result: {label}")
                    st.write(f"Confidence: **{conf:.2%}**")

            with col_b:
                st.subheader("Grad-CAM Output")

                if st.session_state.stage1_result is not None:
                    show_overlay(np.array(image))
                else:
                    st.info("Run Stage 1 Detection to view Grad-CAM.")
    # =====================================================
    # STAGE 2
    # =====================================================
    elif st.session_state.page == "Stage 2":
        st.header("Stage 2: Benign / Malignant ROI Classification")

        if st.session_state.cropped_img is None:
            st.warning("No ROI has been cropped yet.")
            if st.button("Back to Stage 1"):
                go_stage1()
                st.rerun()
        else:
            roi_col, result_col = st.columns([2, 2])

            with roi_col:
                st.subheader("Cropped ROI")
                st.image(st.session_state.cropped_img, caption="Selected ROI", width=300)
                st.write("ROI size:", st.session_state.cropped_img.size)

                if st.button("Back"):
                    go_stage1()
                    st.rerun()

            with result_col:
                st.subheader("Classification Output")

                if st.button("Run Stage 2 Classification"):
                    stage2_label, stage2_conf = run_stage2_model(
                        np.array(st.session_state.cropped_img)
                    )
                    st.session_state.stage2_result = (stage2_label, stage2_conf)

                if st.session_state.stage2_result is not None:
                    label, conf = st.session_state.stage2_result
                    st.success(f"Stage 2 Result: {label}")
                    st.write(f"Confidence: **{conf:.2%}**")
                    show_overlay(
                        np.array(st.session_state.cropped_img),
                        title="Stage 2 Grad-CAM"
                    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Prototype dashboard for image-based lung tumour classification and clinician decision support."
)