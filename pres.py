# app.py
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from torchvision.datasets import VOCSegmentation
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot
from fastapi import FastAPI
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests



app = FastAPI()


# Set Streamlit page configuration for full-width layout
st.set_page_config(
    page_title="PASCAL-VOC 2012 EDA & Image Sampler",
    layout="wide",  # Enables full-width layout
    initial_sidebar_state="expanded"  # Sidebar starts expanded
)

# —————————————————————————————————————————————————————————————————————————————
# CONFIG
# —————————————————————————————————————————————————————————————————————————————
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

@st.cache(allow_output_mutation=True)
def load_dataset():
    ds = VOCSegmentation(
        root='data',
        year='2012',
        image_set='trainval',
        download=True
    )
    mapping = {cls: [] for cls in VOC_CLASSES}
    for idx, (_, mask) in enumerate(ds):
        arr = np.array(mask)
        uniq = np.unique(arr)
        valid = uniq[(uniq != 0) & (uniq != 255) & (uniq < len(VOC_CLASSES))]
        for ci in valid:
            mapping[VOC_CLASSES[ci]].append(idx)
    return ds, mapping

# —————————————————————————————————————————————————————————————————————————————
# SIDEBAR
# —————————————————————————————————————————————————————————————————————————————
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**Select class(es) to sample:**")
chosen = st.sidebar.multiselect(
    "VOC class",
    options=VOC_CLASSES[1:],  # omit background
    default=["person"],
    max_selections=4
)

# —————————————————————————————————————————————————————————————————————————————
# MAIN
# —————————————————————————————————————————————————————————————————————————————
st.title("📊 PASCAL-VOC 2012 EDA & Image Sampler")

# Top-level folder tabs, including a new “Introduction” section
tab_intro, tab_data, tab_preprocessing, tab_models, tab_results, tab_demo = st.tabs([
         "Introduction", "data", "data preprocessing", "models", "results", "demo"
])

# —————————————————————————————————————————————————————————————————————————————
# INTRODUCTION TAB
# —————————————————————————————————————————————————————————————————————————————
with tab_intro:
    st.markdown(
        "<h1 style='font-size:72px; margin-bottom:0.2em;'>🎉 Team Introduction</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='font-size:35px; margin-top:0.5em;'>Team Name: <em>Tassay aka.MARS</em></h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='font-size:40px; margin-top:1em;'><strong>Team Members:</strong></p>
        <ul style='font-size:18px; margin-left:1em;'>
          <li>Makhmud — Captain</li>
          <li>Aruay</li>
          <li>Ruana</li>
          <li>Sanzhar</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Team & Captain Photos")

    # Directory containing your intro images
    intro_dir = "introduction"

    # List all image files in the folder
    all_images = [
        f for f in os.listdir(intro_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # 1) Select the team photo
    team_choices = [img for img in all_images if "team" in img.lower()]
    team_photo = st.selectbox(
        "Select Team Photo:",
        options=team_choices or all_images,
        index=0 if team_choices else None
    )
    st.image(
        os.path.join(intro_dir, team_photo),
        caption="Team Photo",
        use_column_width=True
    )

    # 2) Select the captain photo
    captain_choices = [img for img in all_images if "captain" in img.lower() or "makhmud" in img.lower()]
    captain_photo = st.selectbox(
        "Our amazing captain!!!:",
        options=captain_choices or all_images,
        index=0 if captain_choices else None,
        key="captain_select"
    )
    st.image(
        os.path.join(intro_dir, captain_photo),
        caption="Captain: Makhmud",
        width=300
    )

    st.write("_P.S. Ruana is our photographer!_")
# —————————————————————————————————————————————————————————————————————————————
# DATA TAB
# —————————————————————————————————————————————————————————————————————————————
def list_dir(path):
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return None

with tab_data:
    # Define dataset information
    train_size = 2171
    val_size = 2147
    num_classes = 20
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # Info1: Dataset details
    info1_html = f"""
    <p style="font-size:20px; margin-bottom:0.5em;">
    <strong>Train dataset size:</strong> {train_size}<br>
    <strong>Validation dataset size:</strong> {val_size}<br>
    <strong>Number of classes:</strong> {num_classes}<br>
    <strong>Classes:</strong> {', '.join(classes)}
    </p>
    """

    # Info2: Dataset statistics
    info2_html = """
    <p style="font-size:20px; margin-bottom:0.5em;">
    <strong>Image Dimensions:</strong> 224×224 (resized)<br>
    <strong>Total Images:</strong> 4318<br>
    <strong>Annotations Format:</strong> Segmentation Masks<br>
    <strong>Dataset Year:</strong> 2012
    </p>
    """

    # Info3: Additional details
    info3_html = """
    <p style="font-size:20px; margin-bottom:0.5em;">
    <strong>Dataset Source:</strong> PASCAL VOC 2012<br>
    <strong>Download Link:</strong> <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/">VOC Dataset</a><br>
    <strong>License:</strong> Creative Commons Attribution 2.5<br>
    <strong>Preprocessing:</strong> Cropping, Resizing, Normalization
    </p>
    """
    # Display the information in three columns
    col1, col2, col3 = st.columns(3)
    col1.markdown(info1_html, unsafe_allow_html=True)
    col2.markdown(info2_html, unsafe_allow_html=True)
    col3.markdown(info3_html, unsafe_allow_html=True)


    # Dataset Analysis
    dataset, cls_to_idx = load_dataset()

    # 1) Class-presence histogram
    st.subheader("1. Class-presence Histogram")
    counts = [len(cls_to_idx[cls]) for cls in VOC_CLASSES]
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(VOC_CLASSES, counts)
    ax1.set_xticklabels(VOC_CLASSES, rotation=90)
    ax1.set_ylabel("Number of images")
    plt.tight_layout()
    st.pyplot(fig1)

    # 2) Multi-label count histogram
    st.subheader("2. Multi-label Count per Image")
    num_labels = []
    for _, mask in dataset:
        arr = np.array(mask)
        uniq = np.unique(arr)
        valid = uniq[(uniq != 0) & (uniq != 255) & (uniq < len(VOC_CLASSES))]
        num_labels.append(len(valid))

    num_arr = np.array(num_labels)
    bins = np.arange(0, num_arr.max() + 2) - 0.5

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(num_arr, bins=bins, edgecolor='k')
    ax2.set_xlabel('Number of object classes in image')
    ax2.set_ylabel('Number of images')
    ax2.set_title('Multi-label count per image')
    ax2.set_xticks(np.arange(0, num_arr.max() + 1))
    plt.tight_layout()
    st.pyplot(fig2)

    # 3) Co-occurrence heatmap
    st.subheader("3. Class Co-occurrence Heatmap")
    n = len(VOC_CLASSES)
    cooc = np.zeros((n, n), dtype=int)
    for _, mask in dataset:
        arr = np.array(mask)
        uniq = np.unique(arr)
        valid = uniq[(uniq != 0) & (uniq != 255) & (uniq < n)]
        for i, j in combinations(valid, 2):
            cooc[i, j] += 1
            cooc[j, i] += 1

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cooc,
        xticklabels=VOC_CLASSES,
        yticklabels=VOC_CLASSES,
        cmap="Reds",
        square=True,
        cbar_kws={'label': 'Co-occurrence count'},
        ax=ax3
    )
    ax3.set_title("Class Co-occurrence (trainval 2012)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig3)

    # 4) Sample images containing ALL selected class(es)
    st.subheader("4. Sample Images Containing All Selected Class(es)")
    if not chosen:
        st.info("Please select at least one class in the sidebar.")
    else:
        sets_of_idxs = [set(cls_to_idx[cls]) for cls in chosen]
        common_idxs = list(set.intersection(*sets_of_idxs)) if sets_of_idxs else []

        if not common_idxs:
            st.warning(f"No images contain all of: {', '.join(chosen)}")
        else:
            sample_idxs = common_idxs[:3]
            cols = st.columns(len(sample_idxs))
            for col, img_idx in zip(cols, sample_idxs):
                img, _ = dataset[img_idx]
                col.image(img, use_column_width=True)
                col.caption(f"Image #{img_idx}")

# —————————————————————————————————————————————————————————————————————————————
# DATA PREPROCESSING TAB
# —————————————————————————————————————————————————————————————————————————————
with tab_preprocessing:
    st.header("Data Preprocessing")
    st.markdown("### Preprocessing Visualizations")

    # Define dataset information
    train_size = 3688
    val_size = 1639
    test_size = 1311
    num_classes = 20
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # Info1: Dataset details
    # Info1: Dataset details (larger text)
    info1_html = f"""
    <p style="font-size:20px; margin-bottom:0.5em;">
    <strong>Train dataset size:</strong> {train_size}<br>
    <strong>Validation dataset size:</strong> {val_size}<br>
    <strong>Test dataset size:</strong> {test_size}<br>
    <strong>Number of classes:</strong> {num_classes}<br>
    <strong>Classes:</strong> {', '.join(classes)}
    </p>
    """

    # Info2: Dataset statistics
    info2_html = """
    <p style="font-size:18px; margin-bottom:0.5em;">
    <strong>Image Dimensions:</strong> 224×224 (resized)<br>
    <strong>Total Images:</strong> 6638<br>
    <strong>Annotations Format:</strong> Segmentation Masks<br>
    <strong>Dataset Year:</strong> 2012
    </p>
    """

    # Info3: Additional details
    info3_html = """
    <p style="font-size:18px; margin-bottom:0.5em;">
    <strong>Dataset Source:</strong> PASCAL VOC 2012<br>
    <strong>Download Link:</strong>
        <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/" target="_blank">
        VOC Dataset
        </a><br>
    <strong>License:</strong> Creative Commons Attribution 2.5<br>
    <strong>Preprocessing:</strong> Cropping, Resizing, Normalization
    </p>
    """
    # Display the information in three columns
    col1, col2, col3 = st.columns(3)
    col1.markdown(info1_html, unsafe_allow_html=True)
    col2.markdown(info2_html, unsafe_allow_html=True)
    col3.markdown(info3_html, unsafe_allow_html=True)

    # Define the folder and images to load
    image_folder = "images"
    image_files = [
        "train_before_crop.jpg", "val_before_crop.jpg",  # New images
        "before_crop.jpg", "after_crop.jpg",            # Existing images
        "crop_distribution.jpg"
    ]

    # Display the first two images (train_before_crop and val_before_crop) in one row
    st.markdown("### Before Crop Stage")
    col1, col2 = st.columns(2)
    image_path_1 = os.path.join(image_folder, image_files[0])
    image_path_2 = os.path.join(image_folder, image_files[1])

    if os.path.exists(image_path_1):
        col1.image(image_path_1, caption="Train Before Crop", use_column_width=True)
    else:
        col1.warning(f"Image `{image_files[0]}` not found in `{image_folder}`.")

    if os.path.exists(image_path_2):
        col2.image(image_path_2, caption="Validation Before Crop", use_column_width=True)
    else:
        col2.warning(f"Image `{image_files[1]}` not found in `{image_folder}`.")

    # Display the next two images (before_crop and after_crop) in one row
    st.markdown("### After Crop Stage")
    col3, col4 = st.columns(2)
    image_path_3 = os.path.join(image_folder, image_files[2])
    image_path_4 = os.path.join(image_folder, image_files[3])

    if os.path.exists(image_path_3):
        col3.image(image_path_3, caption="Before Crop", use_column_width=True)
    else:
        col3.warning(f"Image `{image_files[2]}` not found in `{image_folder}`.")

    if os.path.exists(image_path_4):
        col4.image(image_path_4, caption="After Crop", use_column_width=True)
    else:
        col4.warning(f"Image `{image_files[3]}` not found in `{image_folder}`.")

    # Display the crop distribution image below the rows
    st.markdown("### Crop Distribution")
    image_path_5 = os.path.join(image_folder, image_files[4])
    if os.path.exists(image_path_5):
        st.image(image_path_5, caption="Crop Distribution", use_column_width=True)
    else:
        st.warning(f"Image `{image_files[4]}` not found in `{image_folder}`.")
# —————————————————————————————————————————————————————————————————————————————
# MODELS TAB
# —————————————————————————————————————————————————————————————————————————————
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),  # GELU activation for better performance

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # Downsample
            nn.Dropout2d(0.1),   # Spatial dropout for regularization

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Fourth block
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # Downsample
            nn.Dropout2d(0.15),

            # Fifth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Sixth block
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # Downsample
            nn.Dropout2d(0.2),

            # Final feature block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling for flexible input sizes
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

        # Initialize weights

from torchvision import models

with tab_models:
    st.header("Models Folder")

    # Define a dictionary of models
    model_dict = {
        "SimpleCNN": SimpleCNN(num_classes=len(VOC_CLASSES) - 1),
        "ResNet50": models.resnet50(pretrained=False),
        "MobileNetV3": models.mobilenet_v3_large(pretrained=False),
        "ShuffleNetV2": models.shufflenet_v2_x1_0(pretrained=False),
        "EfficientNetB0": models.efficientnet_b0(pretrained=False),
        "SqueezeNet1_1": models.squeezenet1_1(pretrained=False),
        "WideResNet50_2": models.wide_resnet50_2(pretrained=False),
        "ConvNeXtTiny": models.convnext_tiny(pretrained=False)
    }

    # Iterate through each model and display its architecture
    for model_name, model_instance in model_dict.items():
        st.subheader(f"{model_name} Architecture")
        # st.code(
        #     f"""# {model_name} Architecture {model_instance}""",
        #     language="python"
        # )

        # Visualize model architecture interactively
        st.subheader(f"Interactive Visualization for {model_name}")
        # st.text("Model Summary:")
        summary_text = []
        model_instance.apply(lambda module: summary_text.append(str(module)))
        # st.text("\n".join(summary_text))

        # Generate a dummy input for visualization
        dummy_input = torch.randn(1, 3, 224, 224)  # Assuming input size is 224x224
        model_graph = make_dot(model_instance(dummy_input), params=dict(model_instance.named_parameters()))
        st.graphviz_chart(model_graph.source)

# —————————————————————————————————————————————————————————————————————————————
# RESULTS TAB
# —————————————————————————————————————————————————————————————————————————————
import pandas as pd

import pandas as pd

with tab_results:
    st.header("🔧 Custom CNN Preprocessing & HPO")

    # Not cropped
    # Before and After Cropping
    st.subheader("Before and After Cropping")
    col1, col2 = st.columns(2)
    col1.image(
        "custom_CNN/not_cropped.png",
        caption="Before Cropping",
        use_column_width=True
    )
    col2.image(
        "custom_CNN/cropped.png",
        caption="After Cropping",
        use_column_width=True
    )

    # Crop Confusion Matrix
    st.subheader("Crop Confusion Matrix")
    col3, col4 = st.columns(2)
    col3.image(
        "custom_CNN/cm_cropped.png",
        caption="Confusion Matrix (Cropped)",
        use_column_width=True
    )
    col4.image(
        "custom_CNN/cm_balanced.png",
        caption="Confusion Matrix (Balanced)",
        use_column_width=True
    )


    # Balanced
    st.subheader("Balanced")
    b1, b2 = st.columns(2)
    b1.image("custom_CNN/dist_orig.png",      caption="Original Distribution", use_column_width=True)
    b2.image("custom_CNN/dist_balanced.png",  caption="Balanced Distribution", use_column_width=True)
    st.image("custom_CNN/balanced.png",        caption="Balanced Samples", use_column_width=True)

    # Segmented
    st.subheader("Segmented")
    s1, s2, s3 = st.columns(3)
    s1.image("custom_CNN/segmented_samples.png", caption="Segmented Samples", use_column_width=True)
    s2.image("custom_CNN/segmented.png",          caption="Segmented Images",  use_column_width=True)
    s3.image("custom_CNN/cm_segmented.png",       caption="CM (Segmented)",   use_column_width=True)

    # Hyperparameter Search
    st.subheader("Hyperparameter Search Results")
    hp_path = os.path.join("custom_CNN", "hyperparameter_search.txt")
    if os.path.exists(hp_path):
        # 1) show raw table
        try:
            hp_df = pd.read_csv(hp_path, delim_whitespace=True)
            st.markdown("**HPO Log (as table)**")
            st.dataframe(hp_df, height=300)
        except Exception:
            st.warning("Could not parse hyperparameter_search.txt as whitespace-delimited table.")
            hp_text = open(hp_path, encoding="utf-8").read()
            st.code(hp_text, language="text")

        # 2) plot accuracy & loss side by side
        if "epoch" in hp_df.columns and {"train_acc","val_acc"}.issubset(hp_df.columns):
            fig, (ax_a, ax_l) = plt.subplots(1,2, figsize=(12,4))
            # Accuracy
            ax_a.plot(hp_df["epoch"], hp_df["train_acc"], label="Train Acc")
            ax_a.plot(hp_df["epoch"], hp_df["val_acc"],   label="Val Acc")
            ax_a.set_title("HPO Accuracy")
            ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Accuracy")
            ax_a.legend()
            # Loss
            ax_l.plot(hp_df["epoch"], hp_df["train_loss"], label="Train Loss")
            ax_l.plot(hp_df["epoch"], hp_df["val_loss"],   label="Val Loss")
            ax_l.set_title("HPO Loss")
            ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
            ax_l.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("HPO log missing expected columns (`epoch`,`train_acc`,`val_acc`,`train_loss`,`val_loss`).")
    else:
        st.warning("`custom_CNN/hyperparameter_search.txt` not found.")

    # SGD optimizer results
    st.subheader("SGD Optimizer Experiments")
    g1, g2 = st.columns(2)
    g1.image("custom_CNN/sgd.png",         caption="SGD Learning Curves",   use_column_width=True)
    g2.image("custom_CNN/sgd_results.png", caption="SGD Final Performance", use_column_width=True)


    # 3) Parse results.txt into DataFrames
    st.subheader("Per-Class Metrics Comparison")
    results_path = os.path.join("results", "results.txt")
    if not os.path.exists(results_path):
        st.warning("`results/results.txt` not found; cannot build tables.")
    else:
        text = open(results_path, encoding="utf-8").read()
        sections = text.split("Model: ")[1:]
        reports = {}
        for sec in sections:
            lines = [l for l in sec.splitlines() if l.strip()]
            model_name = lines[0].strip()
            hdr_idx = next(i for i,l in enumerate(lines) if "precision" in l and "recall" in l)
            class_lines = lines[hdr_idx+1 : hdr_idx+1 + (len(VOC_CLASSES)-1)]
            rows = []
            for ln in class_lines:
                parts = ln.split()
                cls    = parts[0]
                recall = float(parts[2])
                f1     = float(parts[3])
                rows.append((cls, recall, f1))
            df = pd.DataFrame(rows, columns=["class","accuracy","f1"]).set_index("class")
            reports[model_name] = df

        # wide tables
        acc_df = pd.DataFrame({m: df["accuracy"] for m, df in reports.items()})
        f1_df  = pd.DataFrame({m: df["f1"]       for m, df in reports.items()})

        # styling helper
        def highlight_max_bg(s, color='#ffff7f'):
            is_max = s == s.max()
            return [
                f'background-color: {color}; font-weight: bold; font-size:16px;'
                if m else 'font-size:14px;'
                for m in is_max
            ]

        # 4) Styled interactive tables
        st.markdown("**Table: Per-Class Accuracy")
        styled_acc = acc_df.style.format("{:.2f}").apply(highlight_max_bg, axis=1)
        st.dataframe(styled_acc, height=400)

        st.markdown("**Table: Per-Class F1 Score**")
        styled_f1 = f1_df.style.format("{:.2f}").apply(highlight_max_bg, axis=1)
        st.dataframe(styled_f1, height=400)

        # 5) Heatmaps side-by-side
        st.subheader("Metric Heatmaps")
        col1, col2 = st.columns(2)
        with col1:
            fig_acc, ax_acc = plt.subplots(figsize=(6,6))
            sns.heatmap(acc_df, annot=True, fmt=".2f",
                        cmap="YlGnBu", cbar_kws={"label":"Accuracy"},
                        annot_kws={"size":10}, ax=ax_acc)
            ax_acc.set_title("Accuracy Heatmap")
            ax_acc.set_ylabel("")
            plt.xticks(rotation=90)
            st.pyplot(fig_acc)
        with col2:
            fig_f1, ax_f1 = plt.subplots(figsize=(6,6))
            sns.heatmap(f1_df, annot=True, fmt=".2f",
                        cmap="YlOrRd", cbar_kws={"label":"F1 Score"},
                        annot_kws={"size":10}, ax=ax_f1)
            ax_f1.set_title("F1 Score Heatmap")
            ax_f1.set_ylabel("")
            plt.xticks(rotation=90)
            st.pyplot(fig_f1)

    # 6) Parse head_only_results.txt
    st.subheader("Fine-tuning classifier only for last layer.")
    head_path = os.path.join("results", "head_only_results.txt")
    if not os.path.exists(head_path):
        st.warning("`results/head_only_results.txt` not found; cannot build head-only tables.")
    else:
        text = open(head_path, encoding="utf-8").read()
        sections = text.split("Model: ")[1:]
        head_reports = {}
        for sec in sections:
            lines = [l for l in sec.splitlines() if l.strip()]
            model_name = lines[0].strip()
            hdr_idx = next(i for i,l in enumerate(lines) if "precision" in l and "recall" in l)
            class_lines = lines[hdr_idx+1 : hdr_idx+1 + (len(VOC_CLASSES)-1)]
            rows = []
            for ln in class_lines:
                parts = ln.split()
                cls    = parts[0]
                recall = float(parts[2])
                f1     = float(parts[3])
                rows.append((cls, recall, f1))
            df = pd.DataFrame(rows, columns=["class","accuracy","f1"]).set_index("class")
            head_reports[model_name] = df

        head_acc_df = pd.DataFrame({m: df["accuracy"] for m, df in head_reports.items()})
        head_f1_df  = pd.DataFrame({m: df["f1"]       for m, df in head_reports.items()})

        st.markdown("**Table: Fine-tuning classifier Per-Class Accuracy")
        styled_hacc = head_acc_df.style.format("{:.2f}").apply(highlight_max_bg, axis=1)
        st.dataframe(styled_hacc, height=400)

        st.markdown("**Table: Fine-tuning classifier Per-Class F1 Score**")
        styled_hf1 = head_f1_df.style.format("{:.2f}").apply(highlight_max_bg, axis=1)
        st.dataframe(styled_hf1, height=400)

        st.subheader("Fine-tuning classifier Heatmaps")
        c1, c2 = st.columns(2)
        with c1:
            fig_hacc, ax_hacc = plt.subplots(figsize=(6,6))
            sns.heatmap(head_acc_df, annot=True, fmt=".2f",
                        cmap="YlGnBu", cbar_kws={"label":"Accuracy"},
                        annot_kws={"size":10}, ax=ax_hacc)
            ax_hacc.set_title("Fine-tuning classifier Heatmap")
            ax_hacc.set_ylabel("")
            plt.xticks(rotation=90)
            st.pyplot(fig_hacc)
        with c2:
            fig_hf1, ax_hf1 = plt.subplots(figsize=(6,6))
            sns.heatmap(head_f1_df, annot=True, fmt=".2f",
                        cmap="YlOrRd", cbar_kws={"label":"F1 Score"},
                        annot_kws={"size":10}, ax=ax_hf1)
            ax_hf1.set_title("Fine-tuning classifier F1 Score Heatmap")
            ax_hf1.set_ylabel("")
            plt.xticks(rotation=90)
            st.pyplot(fig_hf1)
    # 7) Parse head_only_loss.txt and plot head-only training curves
    st.subheader("Fine-tuning classifier Training Curves")
    head_loss_path = os.path.join("results", "head_only_loss.txt")
    if not os.path.exists(head_loss_path):
        st.warning("`results/head_only_loss.txt` not found; cannot plot head-only curves.")
    else:
        text = open(head_loss_path, encoding="utf-8").read()
        blocks = text.split("Model: ")[1:]
        head_logs = {}
        for blk in blocks:
            lines = [l.strip() for l in blk.splitlines() if l.strip()]
            model_name = lines[0].strip()
            # collect all lines starting with "Epoch"
            epoch_lines = [l for l in lines if l.startswith("Epoch")]
            epochs, tr_loss, tr_acc, vl_loss, vl_acc = [], [], [], [], []
            for l in epoch_lines:
                # format: Epoch X/Y | Train Loss: ... Acc: ... | Val Loss: ... Acc: ...
                parts = [p.strip() for p in l.split("|")]
                ep = int(parts[0].split()[1].split("/")[0])
                epochs.append(ep)
                # Train metrics
                t_loss = float(parts[1].split("Train Loss:")[1].split()[0])
                t_acc  = float(parts[1].split("Acc:")[1].split()[0])
                tr_loss.append(t_loss); tr_acc.append(t_acc)
                # Val metrics
                v_loss = float(parts[2].split("Val Loss:")[1].split()[0])
                v_acc  = float(parts[2].split("Acc:")[1].split()[0])
                vl_loss.append(v_loss); vl_acc.append(v_acc)
            head_logs[model_name] = {
                "epochs": epochs,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": vl_loss,
                "val_acc": vl_acc
            }

        # 1×N grid of accuracy
        n = len(head_logs)
        fig_acc, axes_acc = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1: axes_acc = [axes_acc]
        for ax, (name, d) in zip(axes_acc, head_logs.items()):
            ax.plot(d["epochs"], d["train_acc"], label="Train Acc")
            ax.plot(d["epochs"], d["val_acc"],   label="Val Acc")
            ax.set_title(name)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
            ax.legend()
        plt.tight_layout()
        st.markdown("#### Fine-tuning classifier Accuracy per Epoch")
        st.pyplot(fig_acc)

        # 1×N grid of loss
        fig_loss, axes_loss = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1: axes_loss = [axes_loss]
        for ax, (name, d) in zip(axes_loss, head_logs.items()):
            ax.plot(d["epochs"], d["train_loss"], label="Train Loss")
            ax.plot(d["epochs"], d["val_loss"],   label="Val Loss")
            ax.set_title(name)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.legend()
        plt.tight_layout()
        st.markdown("#### Fine-tuning classifier Loss per Epoch")
        st.pyplot(fig_loss)


# —————————————————————————————————————————————————————————————————————————————
# DEMO TAB
# —————————————————————————————————————————————————————————————————————————————
with tab_demo:
    st.header("Demo: FastAPI Model Predictions")
    st.markdown("### Upload and Predict via FastAPI")

    # 1) Endpoint input
    api_endpoint = st.text_input(
        "API endpoint",
        value="http://localhost:8000/predict",
        help="The FastAPI `/predict` URL"
    )

    # 2) Image selector
    image_folder = "model/img"
    available_images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    selected = st.multiselect(
        "Select up to 5 images:",
        available_images,
        max_selections=5
    )

    if selected:
        st.markdown("#### Selected Images")
        cols = st.columns(len(selected))
        for col, img_name in zip(cols, selected):
            col.image(os.path.join(image_folder, img_name), caption=img_name, use_column_width=True)

        # 3) Show curl commands
        st.markdown("#### CURL Commands")
        curl_lines = []
        for img_name in selected:
            path = os.path.join(image_folder, img_name)
            curl_lines.append(f'curl -F "file=@{path}" {api_endpoint}')
        st.code("\n".join(curl_lines), language="bash")

        # 4) Send button
        if st.button("Send to API"):
            st.markdown("#### API Responses")
            for img_name in selected:
                path = os.path.join(image_folder, img_name)
                try:
                    with open(path, "rb") as f:
                        files = {"file": (img_name, f, "image/jpeg")}
                        resp = requests.post(api_endpoint, files=files, timeout=10)
                        resp.raise_for_status()
                        result = resp.json()
                    st.write(f"**{img_name}** →", result)
                except Exception as e:
                    st.error(f"Failed for {img_name}: {e}")
    else:
        st.info("Please select at least one image to demo.")