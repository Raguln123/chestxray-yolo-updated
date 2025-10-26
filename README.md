# Disease Detection and Diagnosis — Chest X-ray (YOLOv8 + Streamlit)

End‑to‑end project for detecting multiple chest conditions in X‑ray images using YOLOv8.
It includes data prep (CSV ➜ YOLO labels), preprocessing/augmentation, training, evaluation,
hyperparameter tuning, and a Streamlit app that shows preprocessing steps and runs inference.

> You will add the dataset zip you shared into `data/raw/` or point to your Google Drive path.

## Folder Layout
```
chestxray-yolo/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── convert_annotations.py   # CSV → YOLO format
│   ├── preprocess.py            # preprocessing utilities + CLI demo
│   ├── anchors_kmeans.py        # optional: anchor box tuning
│   ├── train.py                 # trains YOLOv8 (ultralytics)
│   ├── evaluate.py              # metrics & per-class analysis
│   ├── tune.py                  # hyperparameter sweep (learning rate, batch size, IoU, imgsz)
│   ├── inference.py             # batch/single image inference
│   └── streamlit_app.py         # UI to preview preprocessing & run inference
├── data/
│   ├── dataset.yaml             # YOLO data config (edit paths after preparing data)
│   ├── raw/                     # put original files here (e.g., your Google Drive zip)
│   ├── interim/                 # temporary conversions
│   └── yolo/                    # final YOLO structure: images/{train,val,test}, labels/{...}
├── models/
│   └── yolo.yaml                # (optional) custom model override (e.g., classes)
├── notebooks/
│   └── EDA.ipynb                # starter notebook (optional)
└── scripts/
    ├── prepare_data.sh
    └── run_train.sh
```

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place `Chest_XRay_Dataset/` under `data/raw/` with `Ground_Truth.csv` and `xray_images/`. Ensure a CSV with boxes exists
     (columns can be `image`, `xmin`, `ymin`, `xmax`, `ymax`, `label`; see `src/config.py`).
   - Convert CSV annotations to YOLO format and create the folder tree:
     ```bash
     python src/convert_annotations.py        --csv data/raw/annotations.csv        --images-root data/raw/images        --out-root data/yolo        --val-split 0.15 --test-split 0.10
     ```
   - Edit `data/dataset.yaml` if needed (class names, paths are auto-written).

3. **Train**
   ```bash
   python src/train.py --model yolov8n.pt --imgsz 640 --epochs 50 --batch 16
   ```

4. **Evaluate**
   ```bash
   python src/evaluate.py --weights runs/detect/train/weights/best.pt
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run src/streamlit_app.py
   ```

   - Upload an X‑ray image to see **preprocessing steps** (normalize, CLAHE, denoise, resize)
     and the **detected conditions** with boxes and confidences.

## Notes
- Multi‑label: Each X‑ray can have multiple conditions (CSV can list multiple rows per image).
- Metadata (age, gender, view): Supported in `convert_annotations.py` for stratified split
  and saved alongside images for future research. YOLO itself doesn’t consume metadata,
  but we keep it for EDA and potential fusion.
- Anchor tuning: Use `src/anchors_kmeans.py` to recompute anchors for medical image sizes.

## Troubleshooting
- If images are very large, increase `--imgsz` or downscale in `preprocess.py`.
- If class imbalance is severe, consider `--augment` and class‑balanced sampling.
- For CUDA issues, ensure a matching PyTorch + CUDA toolkit is installed.
