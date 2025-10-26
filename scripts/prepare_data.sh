#!/usr/bin/env bash
set -e
python src/convert_annotations.py \
  --csv data/raw/Chest_XRay_Dataset/Ground_Truth.csv \
  --images-root data/raw/Chest_XRay_Dataset/xray_images \
  --out-root data/yolo
