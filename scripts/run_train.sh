#!/usr/bin/env bash
set -e
python src/train.py --model yolov8n.pt --data data/dataset.yaml --imgsz 640 --epochs 50 --batch 16 --augment
