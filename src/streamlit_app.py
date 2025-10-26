import streamlit as st
import cv2, numpy as np, os
from ultralytics import YOLO
from preprocess import full_pipeline
from PIL import Image

st.set_page_config(page_title="Chest X-ray Detector", layout="wide")

st.title("🩺 Chest X‑ray Disease Detection (YOLOv8)")
st.markdown("""
Upload an X‑ray image to preview preprocessing (normalize → CLAHE → denoise → resize)
and run YOLO detection. Place your trained weights at `runs/detect/train/weights/best.pt`.
""")

col_left, col_right = st.columns([1,1])

with col_left:
    up = st.file_uploader("Upload X‑ray (JPG/PNG)", type=["jpg","jpeg","png"])
    conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
    size = st.select_slider("Image size", [512, 640, 768, 896, 1024], value=640)
    run = st.button("Run Detection")

def load_image(file):
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

if up is not None:
    img = load_image(up)
    steps, meta = full_pipeline(img, size=size)
    st.subheader("Preprocessing Steps")
    a,b = st.columns(2)
    c,d = st.columns(2)
    with a: st.image(steps["original"], caption="Original", use_column_width=True, channels="BGR")
    with b: st.image(steps["normalized"], caption="Normalized", use_column_width=True, clamp=True)
    with c: st.image(steps["clahe"], caption="CLAHE", use_column_width=True, clamp=True)
    with d: st.image(steps["denoised"], caption="Denoised", use_column_width=True, clamp=True)
    st.image(steps["resized"], caption=f"Resized/Padded to {size}x{size}", use_column_width=True, clamp=True)

    if run:
        weights = "runs/detect/train/weights/best.pt"
        if not os.path.exists(weights):
            st.error("Weights not found. Train the model first or place weights at runs/detect/train/weights/best.pt")
        else:
            model = YOLO(weights)
            # Predict on uploaded file path (convert to RGB for PIL save)
            temp_path = "temp_input.png"
            cv2.imwrite(temp_path, img)
            results = model.predict(source=temp_path, imgsz=size, conf=conf)
            for r in results:
                im = r.plot()  # numpy BGR
                st.image(im, caption="Detections", use_column_width=True, channels="BGR")
                # Show raw predictions
                if r.boxes is not None and len(r.boxes) > 0:
                    st.write("Raw detections (xyxy, conf, cls):")
                    st.dataframe({
                        "xmin": r.boxes.xyxy[:,0].cpu().numpy(),
                        "ymin": r.boxes.xyxy[:,1].cpu().numpy(),
                        "xmax": r.boxes.xyxy[:,2].cpu().numpy(),
                        "ymax": r.boxes.xyxy[:,3].cpu().numpy(),
                        "conf": r.boxes.conf.cpu().numpy(),
                        "cls": r.boxes.cls.cpu().numpy().astype(int)
                    })
            os.remove(temp_path)
else:
    st.info("Upload an X‑ray image to begin.")
