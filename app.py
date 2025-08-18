import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import os
import cv2
import tempfile
import time

# =====================
# Paths
# =====================
RUNS_DIR = Path("runs_ssl")
st.set_page_config(page_title="Sunflower SSL Dashboard", layout="wide")
st.title("🌻 Sunflower Image Detection in Real-Time Dashboard")

# =====================
# Sidebar
# =====================
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.subheader("🧪 Experiment Settings")

exp_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
custom_labels = [
    "BYOL-YOLOv10s", "BYOL-YOLOv11s", "BYOL-YOLOv12s",
    "DINO-YOLOv10s", "DINO-YOLOv11s", "DINO-YOLOv12s"
]
exp_choice_label = st.sidebar.selectbox("🌟 Choose SSL Experiment", custom_labels)
exp_choice = exp_dirs[custom_labels.index(exp_choice_label)]

best_model_path = exp_choice / "weights" / "best.pt"
results_csv = exp_choice / "results.csv"
results_png = exp_choice / "results.png"

st.sidebar.subheader("🤖 Model Configuration")
st.sidebar.markdown(f"**Experiment:** {exp_choice_label}")

st.sidebar.subheader("🔍 Prediction Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# =====================
# Load model
# =====================
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

model = load_model(best_model_path)

# =====================
# IMAGE / VIDEO UPLOAD & ANNOTATION
# =====================
st.subheader("🔍 Test Your Model on a Custom Image/Video")
st.markdown(f"**Using Model:** `{exp_choice_label}`")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","mp4"])
preds_dir = exp_choice / "preds"
os.makedirs(preds_dir, exist_ok=True)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    temp_path = Path(tempfile.mkdtemp()) / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_ext in ["jpg","jpeg","png"]:
        img = Image.open(temp_path).convert("RGB")
        results = model.predict(source=img, conf=conf_threshold, verbose=False)
        annotated_img = results[0].plot(line_width=1, font_size=5)  # smaller bbox and text

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", width=600)
        with col2:
            st.image(annotated_img[:, :, ::-1], caption="Annotated Result", width=600)

        save_path = preds_dir / uploaded_file.name
        Image.fromarray(annotated_img[:, :, ::-1]).save(save_path)
        with open(save_path, "rb") as f:
            st.download_button("💾 Download Annotated Image", f, file_name=uploaded_file.name, mime="image/png")

    elif file_ext == "mp4":
        cap = cv2.VideoCapture(str(temp_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        file_stem = Path(uploaded_file.name).stem
        annotated_video_path = preds_dir / f"annotated_{file_stem}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

        frame_count = 0
        progress_bar = st.progress(0)
        progress_text = st.empty()
        frame_placeholder = st.empty()  # real-time display
        start_time = time.time()

        with st.spinner("Processing video and generating annotations..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot(line_width=1, font_size=4)  # smaller bbox and text
                out.write(annotated_frame[:, :, ::-1])

                # real-time frame display
                frame_placeholder.image(annotated_frame[:, :, ::-1], channels="RGB", width=1000)

                frame_count += 1
                if total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    remaining = max(0, int((elapsed / frame_count) * (total_frames - frame_count)))
                    progress_text.markdown(
                        f"Processing frame {frame_count}/{total_frames} | Estimated time left: {remaining}s"
                    )

        cap.release()
        out.release()
        progress_bar.empty()
        progress_text.empty()
        frame_placeholder.empty()
        st.success("✅ Video annotation complete!")

        # # Display final videos side-by-side
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.markdown("**Original Video**")
        #     st.video(str(temp_path), start_time=0)
        # with col2:
        #     st.markdown("**Annotated Video**")
        #     with open(annotated_video_path, "rb") as f:
        #         st.video(f.read(), start_time=0)

        # Download button
        with open(annotated_video_path, "rb") as f:
            st.download_button(
                "💾 Download Annotated Video",
                f,
                file_name=f"annotated_{file_stem}.mp4",
                mime="video/mp4"
            )


# =====================
# Live Camera Detection (~30 FPS with camera selection)
# =====================
st.subheader("📷 Live Camera Detection (Target: 30 FPS)")

# Sidebar camera options
st.sidebar.subheader("🎥 Camera Source")
camera_type = st.sidebar.radio(
    "Select Camera",
    ["Laptop Webcam", "External USB Webcam", "Mobile Camera (IP/RTSP)"]
)

# For mobile IP cam
mobile_url = None
if camera_type == "Mobile Camera (IP/RTSP)":
    mobile_url = st.sidebar.text_input(
        "Enter Camera URL (e.g., https://0d5975007478.ngrok-free.app)",
        value="https://0d5975007478.ngrok-free.app/video"
    )

    # --------------------
    # Test Camera URL Button
    # --------------------
    if st.sidebar.button("🔍 Test Camera URL"):
        cap_test = cv2.VideoCapture(mobile_url)
        ret, frame = cap_test.read()
        if ret:
            st.sidebar.success("✅ Camera is reachable!")
            st.sidebar.image(frame[:, :, ::-1], channels="RGB", width=300)
        else:
            st.sidebar.error("❌ Could not reach the camera. Check URL or network.")
        cap_test.release()

start_cam = st.button("▶️ Start Live Camera")

if start_cam:
    # Choose capture source
    if camera_type == "Laptop Webcam":
        cap = cv2.VideoCapture(0)
    elif camera_type == "External USB Webcam":
        cap = cv2.VideoCapture(1)  # change index if multiple webcams
    elif camera_type == "Mobile Camera (IP/RTSP)" and mobile_url:
        cap = cv2.VideoCapture(mobile_url)
    else:
        st.error("❌ Please provide a valid camera URL or device")
        cap = None

    if cap and cap.isOpened():
        # Resize and FPS settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        col1, col2 = st.columns([2, 1])
        with col1:
            frame_placeholder = st.empty()
        with col2:
            stop_cam = st.button("⏹️ Stop Camera")
            fps_text = st.empty()

        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read frame. Check camera or URL.")
                break

            # Resize for faster inference
            frame = cv2.resize(frame, (640, 480))

            # Run YOLO detection
            results = model(frame, stream=True, conf=conf_threshold, verbose=False)
            for r in results:
                annotated_frame = r.plot(line_width=2, font_size=12)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Show annotated frame
            frame_placeholder.image(
                annotated_frame[:, :, ::-1], channels="RGB", width=600
            )

            # Show FPS
            fps_text.markdown(f"**FPS:** {fps:.1f}")

            # Lock to ~30 FPS
            elapsed = time.time() - curr_time
            wait_time = max(0, (1/30) - elapsed)
            time.sleep(wait_time)

            if stop_cam:
                break

        cap.release()
        st.success("✅ Camera stopped")


# =====================
# Training Performance (Graphs Smaller)
# =====================
st.subheader("📈 Training Performance Overview")
if results_csv.exists():
    df = pd.read_csv(results_csv)
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    small_figsize = (8,4)  # smaller plots

    # Train/Val Box Loss
    with col1:
        fig, ax = plt.subplots(figsize=small_figsize)
        if "epoch" in df.columns and "train/box_loss" in df.columns and "val/box_loss" in df.columns:
            ax.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
            ax.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title("Train vs Val Box Loss")
            ax.legend(); ax.grid(True)
            st.pyplot(fig)

    # Detailed Loss
    with col2:
        loss_cols = ["epoch", "train/box_loss","train/cls_loss","val/box_loss","val/cls_loss"]
        if all(col in df.columns for col in loss_cols):
            fig2, ax2 = plt.subplots(figsize=small_figsize)
            ax2.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", linewidth=2)
            ax2.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss", linewidth=2)
            ax2.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", linestyle="--")
            ax2.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss", linestyle="--")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Detailed Train/Val Loss")
            ax2.legend(); ax2.grid(True)
            st.pyplot(fig2)

    # Learning Rates
    with col3:
        lr_columns = [col for col in df.columns if col.startswith("lr/pg")]
        if lr_columns:
            fig3, ax3 = plt.subplots(figsize=small_figsize)
            colors = ['purple','orange','green']; styles=['-','--','-.']; markers=['o','s','^']
            for idx,col in enumerate(lr_columns):
                ax3.plot(df["epoch"], df[col], label=col, color=colors[idx%3], linestyle=styles[idx%3],
                         marker=markers[idx%3], markersize=4, linewidth=2)
            ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning Rate"); ax3.set_title("Learning Rates")
            ax3.legend(); ax3.grid(True)
            st.pyplot(fig3)

    # mAP
    with col4:
        map_keys = ["metrics/mAP50(B)","metrics/mAP50-95(B)"]
        map_col = next((k for k in map_keys if k in df.columns), None)
        if map_col:
            fig4, ax4 = plt.subplots(figsize=small_figsize)
            ax4.plot(df["epoch"], df[map_col], label=map_col, color="green", linewidth=2)
            ax4.set_xlabel("Epoch"); ax4.set_ylabel("mAP"); ax4.set_title("mAP During Training")
            ax4.grid(True); ax4.legend(); st.pyplot(fig4)

# Display results.png
st.subheader("🖼️ Results Image")
if results_png.exists():
    img = Image.open(results_png)
    st.image(img, caption="Results Image", use_container_width=True)
