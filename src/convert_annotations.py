import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import COLUMNS, CFG

def norm_box_full(w, h):
    # Full-image normalized box (centered)
    return 0.5, 0.5, 1.0, 1.0

def ensure_dirs(root):
    for split in ["train","val","test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

def expand_multilabel_rows(df, images_root: Path):
    cols = COLUMNS
    sep = CFG.label_sep
    token_no = CFG.no_finding_token

    records = []
    for _, r in df.iterrows():
        img_name = str(r[cols.image]).strip()
        labels_raw = str(r[cols.labels]).strip()
        # skip empty
        if not img_name:
            continue
        labels = [s.strip() for s in labels_raw.split(sep)] if labels_raw else []
        labels = [l for l in labels if l and l != token_no]
        # If only No Finding => no boxes; keep a row with empty labels to allow stratified split by metadata
        records.append({
            "image_path": str((images_root / img_name).resolve()),
            "labels": labels,
            "age": r.get(cols.age, np.nan),
            "gender": r.get(cols.gender, np.nan),
            "view": r.get(cols.view, np.nan),
        })
    return pd.DataFrame.from_records(records)

def main(args):
    csv_path = Path(args.csv)
    images_root = Path(args.images_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out_root)

    df_raw = pd.read_csv(csv_path)
    df = expand_multilabel_rows(df_raw, images_root)

    # Image-level split
    images = df["image_path"].unique()
    train_imgs, tmp = train_test_split(images, test_size=CFG.val_split + CFG.test_split, random_state=CFG.seed)
    relative = CFG.test_split / (CFG.val_split + CFG.test_split) if (CFG.val_split + CFG.test_split)>0 else 0.0
    val_imgs, test_imgs = train_test_split(tmp, test_size=relative, random_state=CFG.seed)

    split_map = {}
    for p in train_imgs: split_map[p] = "train"
    for p in val_imgs: split_map[p] = "val"
    for p in test_imgs: split_map[p] = "test"

    classes = CFG.classes
    class_to_idx = {c:i for i,c in enumerate(classes)}

    from shutil import copy2
    import cv2

    for img_path in tqdm(images):
        split = split_map.get(img_path, "train")
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] cannot read {img_path}")
                continue
            h, w = img.shape[:2]
        except Exception as e:
            print(f"[WARN] OpenCV failed on {img_path}: {e}")
            continue

        # get labels for this image
        row = df[df["image_path"] == img_path].iloc[0]
        labels = row["labels"]

        # Write label file
        lines = []
        if labels:
            x,y,bw,bh = norm_box_full(w, h)
            for l in labels:
                if l not in class_to_idx:
                    # unseen label -> skip but warn once
                    print(f"[WARN] Label '{l}' not in config classes; skipping")
                    continue
                cls = class_to_idx[l]
                lines.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

        img_name = Path(img_path).name
        out_img = out_root / "images" / split / img_name
        out_lbl = out_root / "labels" / split / (Path(img_name).stem + ".txt")

        copy2(img_path, out_img)
        with open(out_lbl, "w") as f:
            f.write("\n".join(lines))

    # dataset.yaml
    yaml_path = Path("data/dataset.yaml")
    yaml_text = f"""
# Auto-generated from image-level labels (full-image boxes)
path: {str(out_root.resolve())}
train: images/train
val: images/val
test: images/test

names:
{os.linesep.join([f"  {i}: {c}" for i,c in enumerate(classes)])}
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f: f.write(yaml_text)
    print(f"[OK] Wrote YOLO dataset to: {out_root}")
    print(f"[OK] Wrote data/dataset.yaml")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Ground_Truth.csv")
    ap.add_argument("--images-root", required=True, help="Folder with all xray images (e.g., Chest_XRay_Dataset/xray_images)")
    ap.add_argument("--out-root", default="data/yolo", help="output root for YOLO structure")
    args = ap.parse_args()
    main(args)
