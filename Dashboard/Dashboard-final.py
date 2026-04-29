import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models

from preprocess_lung import preprocess_slice
from preprocess_roi import preprocess

from utils import (
    normalise_for_display,
    load_uploaded_ct_image,
    crop_raw_from_box,
    crop_display_from_box
)

from ensemble import WeightedSoftVotingEnsemble


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#classes
STAGE1_CLASSES = ["No suspicious nodule", "Suspicious nodule detected"]
STAGE2_CLASSES = ["Benign", "Malignant"]

STAGE1_WEIGHTS = [0.54, 0.43, 0.03] #weights found after tuning
num_classes = 2

STAGE2_MODEL_PATH = "saved_models/resnet50diff_best_auc.pth"

st.set_page_config(
    page_title="Lung Tumour Decision-Support Dashboard",
    layout="wide"
)

# Store values Streamlit needs to remember between button clicks
def init_state():
    defaults = {
        "page": "Stage 1",
        "enable_crop": False,
        "uploaded_image_display": None,
        "uploaded_image_raw": None,
        "uploaded_file_name": None,
        "cropped_img_display": None,
        "cropped_img_raw": None,
        "stage1_result": None,
        "stage2_result": None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# Clear old predictions when a new image is uploaded
def reset_results():
    st.session_state.stage1_result = None
    st.session_state.stage2_result = None
    st.session_state.cropped_img_display = None
    st.session_state.cropped_img_raw = None

#simple helper to switch between stage 1 and atge 2
def go_to(page):
    st.session_state.page = page

#load models for ensemble
resnet50 = models.resnet50(weights=None)
resnet50.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(resnet50.fc.in_features, num_classes))
resnet50.load_state_dict(torch.load("saved_models/resnet50_best_sen.pth", map_location=device))
resnet50.to(device)
resnet50.eval()

resnet101 = models.resnet101(weights=None)
resnet101.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(resnet101.fc.in_features, num_classes))
resnet101.load_state_dict(torch.load("saved_models/resnet101_best_sen.pth", map_location=device))
resnet101.to(device)
resnet101.eval()

densenet121 = models.densenet121(weights=None)
densenet121.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(densenet121.classifier.in_features, num_classes))
densenet121.load_state_dict(torch.load("saved_models/densenet121_best_sen.pth",map_location=device))
densenet121.to(device)
densenet121.eval()

#make ensemble model for stage 1
ensemble_model = WeightedSoftVotingEnsemble(
        models=[resnet50, densenet121, resnet101],
        weights=STAGE1_WEIGHTS).to(device)
ensemble_model.eval()


#load model for stage 2
resnet50_stage2 = models.resnet50(weights=None)
resnet50_stage2.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(resnet50_stage2.fc.in_features, num_classes))
resnet50_stage2.load_state_dict(torch.load("saved_models/resnet50diff_best_auc.pth", map_location=device))
resnet50_stage2.to(device)
resnet50_stage2.eval()


# ImageNet mean and std, used because the models are ResNet/DenseNet based
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#preprocess images for the model's format
def preprocess_stage1(image):    
    image = preprocess_slice(image, size=256)
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
    image = (image - mean)/ std
    image = image.unsqueeze(0)
    return image.to(device)

def preprocess_stage2(roi):
    roi = preprocess(roi, size=128)
    roi = torch.from_numpy(roi.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
    roi = (roi - mean)/ std
    roi = roi.unsqueeze(0)
    return roi.to(device)

#use model to get prediction
def predict(model, input_tensor, classes, use_softmax=False):
    with torch.no_grad():
        output = model(input_tensor)

        if use_softmax:
            output = torch.softmax(output, dim=1)

        probs = output[0]

    index = int(torch.argmax(probs).item())
    label = classes[index]
    confidence = float(probs[index].item())

    prob_dict = {
        classes[i]: float(probs[i].item())
        for i in range(len(classes))
    }

    return label, confidence, prob_dict

#run classifications
def run_stage1(image):
    return predict(
        ensemble_model,
        preprocess_stage1(image),
        STAGE1_CLASSES
    )


def run_stage2(roi):
    return predict(
        resnet50_stage2,
        preprocess_stage2(roi),
        STAGE2_CLASSES,
        use_softmax=True
    )

#GradCAM implementation

def generate_gradcam(model, input_tensor, target_layer, output_size):
    gradients, activations = [], []

    def save_activation(module, input, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)

    try:
        model.eval()
        model.zero_grad()

        with torch.enable_grad():
            input_tensor = input_tensor.requires_grad_(True)
            logits = model(input_tensor)

            class_idx = int(torch.argmax(logits, dim=1).item())
            score = logits[:, class_idx]

            score.backward()

        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations[0], dim=1)
        cam = torch.relu(cam).squeeze().detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        cam = Image.fromarray(np.uint8(cam * 255))
        cam = cam.resize(output_size, resample=Image.BILINEAR)

        return np.array(cam).astype(np.float32) / 255.0

    finally:
        forward_handle.remove()
        backward_handle.remove()


def show_gradcam(image, stage):
    if stage == "stage1":
        model = resnet50
        input_tensor = preprocess_stage1(image)
        output_size = (256, 256)
        title = "Stage 1 Grad-CAM Output"
    else:
        model = resnet50_stage2
        input_tensor = preprocess_stage2(image)
        output_size = (128, 128)
        title = "Stage 2 Grad-CAM Output"

    heatmap = generate_gradcam(model, input_tensor, model.layer4[-1], output_size)
    base_image = normalise_for_display(image)

    heatmap = Image.fromarray(np.uint8(heatmap * 255))
    heatmap = heatmap.resize(
        (base_image.shape[1], base_image.shape[0]),
        resample=Image.BILINEAR
    )

    heatmap = np.array(heatmap).astype(np.float32) / 255.0
    heatmap[heatmap < 0.30] = 0

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
    fig.patch.set_alpha(0)

    ax.imshow(base_image, cmap="gray")
    ax.imshow(heatmap, cmap="turbo", alpha=0.30)
    ax.set_title(title, fontsize=13, fontweight="bold", color="white")
    ax.axis("off")

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, transparent=True)


#prediction label and confidence outputs
def show_result(label, confidence, probs):
    col1, col2 = st.columns(2)

    col1.metric("Prediction", label)
    col2.metric("Confidence", f"{confidence:.2%}")

    st.markdown("Class Probabilities")

    for class_name, prob in probs.items():
        st.progress(prob, text=f"{class_name}: {prob:.2%}")

# initialise dashboard
init_state()

st.title("Lung Tumour Decision-Support Dashboard")

st.caption(
    "This dashboard analyses CT slice images in two stages: nodule detection, "
    "then benign or malignant ROI classification."
)

st.warning(
    "Prototype decision-support tool only. Results must be reviewed by a qualified clinician."
)


with st.sidebar:
    st.header("Dashboard Controls")

    uploaded_file = st.file_uploader(
        "Upload a CT slice image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "dicom"]
    )

    st.markdown("---")

    st.session_state.page = st.radio(
        "Navigation",
        ["Stage 1", "Stage 2"],
        index=0 if st.session_state.page == "Stage 1" else 1
    )

    if st.session_state.page == "Stage 1":
        st.toggle("Enable ROI cropping", key="enable_crop")

    st.markdown("---")

    st.info(
        "Stage 1 checks the full CT image. "
        "Stage 2 analyses a selected ROI."
    )

#reload display image to show newly uploaded one
if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        raw_image = load_uploaded_ct_image(uploaded_file)
        display_image = normalise_for_display(raw_image)

        st.session_state.uploaded_image_raw = raw_image
        st.session_state.uploaded_image_display = Image.fromarray(display_image)
        st.session_state.uploaded_file_name = uploaded_file.name

        reset_results()

#Start up message for user
if st.session_state.uploaded_image_display is None:
    st.info("Upload a CT slice image from the sidebar to begin.")
    st.stop()

#image states
raw_image = st.session_state.uploaded_image_raw
display_image = st.session_state.uploaded_image_display


if st.session_state.page == "Stage 1":
    #Stage 1 page
    st.header("Stage 1: Suspicious Nodule Detection")

    if st.session_state.enable_crop:
        image_col, roi_col = st.columns([3, 2])

        with image_col:
            with st.container(border=True):
                st.subheader("Original CT Image")
                #cropper to define ROI for stage 2
                crop_box = st_cropper(
                    display_image,
                    realtime_update=True,
                    box_color="#FF0000",
                    aspect_ratio=(1, 1),
                    return_type="box"
                )

        with roi_col:
            with st.container(border=True):
                st.subheader("ROI Preview")

                cropped_display = crop_display_from_box(display_image, crop_box)
                cropped_raw = crop_raw_from_box(raw_image, crop_box)

                st.image(cropped_display, caption="Selected ROI", width=280)
                st.write("ROI size:", cropped_display.size)

                if st.button("Select ROI and Continue to Stage 2", use_container_width=True):
                    st.session_state.cropped_img_display = cropped_display
                    st.session_state.cropped_img_raw = cropped_raw
                    st.session_state.stage2_result = None
                    
                    go_to("Stage 2")
                    st.rerun()

        st.markdown("---")

        #run nodule detection 
        if st.button("Run Nodule Detection", use_container_width=True):
            st.session_state.stage1_result = run_stage1(raw_image)

        if st.session_state.stage1_result:
            label, confidence, probs = st.session_state.stage1_result

            with st.container(border=True):
                st.subheader("Stage 1 Result")
                show_result(label, confidence, probs)
                show_gradcam(raw_image, "stage1")

    else:
        image_col, cam_col, result_col = st.columns([3, 3, 2])

        with image_col:
            with st.container(border=True):
                st.subheader("Original CT Image")
                st.image(display_image, use_container_width=True)

        with cam_col:
            with st.container(border=True):
                st.subheader("Grad-CAM Output")

                if st.session_state.stage1_result:
                    show_gradcam(raw_image, "stage1")
                else:
                    st.info("Run Stage 1 detection to view Grad-CAM.")

        with result_col:
            with st.container(border=True):
                st.subheader("Detection Output")

                if st.button("Run Nodule Detection", use_container_width=True):
                    st.session_state.stage1_result = run_stage1(raw_image)
                    st.rerun()

                if st.session_state.stage1_result:
                    show_result(*st.session_state.stage1_result)
                else:
                    st.write("No prediction has been made yet.")


else:
    #Stage 2 page
    st.header("Stage 2: ROI Benign / Malignant Classification")

    if st.session_state.cropped_img_raw is None:
        #Need to crop image in stage 1 for stage 2 input 
        st.warning("No ROI has been selected yet.")

        if st.button("Return to Stage 1", use_container_width=True):
            go_to("Stage 1")
            st.rerun()

    else:
        roi_col, cam_col, result_col = st.columns([3, 3, 2])

        with roi_col:
            with st.container(border=True):
                st.subheader("Selected ROI")

                st.image(
                    st.session_state.cropped_img_display,
                    caption="Selected ROI",
                    use_container_width=True
                )

                st.write("ROI size:", st.session_state.cropped_img_display.size)

                if st.button("Return to Stage 1", use_container_width=True):
                    go_to("Stage 1")
                    st.rerun()

        with cam_col:
            with st.container(border=True):
                st.subheader("Stage 2 Grad-CAM")

                if st.session_state.stage2_result:
                    show_gradcam(st.session_state.cropped_img_raw, "stage2")
                else:
                    st.info("Run Stage 2 classification to view Grad-CAM.")

        with result_col:
            with st.container(border=True):
                st.subheader("Classification Output")
                #run malignancy classification
                if st.button("Run ROI Classification", use_container_width=True):
                    st.session_state.stage2_result = run_stage2(
                        st.session_state.cropped_img_raw
                    )
                    st.rerun()

                if st.session_state.stage2_result:
                    label, confidence, probs = st.session_state.stage2_result

                    if label == "Malignant":
                        st.error("The selected ROI was classified as malignant.")
                    else:
                        st.success("The selected ROI was classified as benign.")

                    show_result(label, confidence, probs)
                else:
                    st.write("No Stage 2 prediction has been made yet.")


st.markdown("---")

st.caption(
    "Prototype dashboard for CT-based lung tumour analysis and clinical decision support."
)