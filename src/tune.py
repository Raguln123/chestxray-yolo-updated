import argparse
import itertools
import json
import os
from ultralytics import YOLO

def main(args):
    model_path = args.model
    combos = list(itertools.product(args.lr0, args.batch, args.iou, args.imgsz))
    os.makedirs("runs/tune", exist_ok=True)
    results = []
    for i, (lr, bs, iou, imgsz) in enumerate(combos):
        name = f"lr{lr}_bs{bs}_iou{iou}_sz{imgsz}"
        model = YOLO(model_path)
        r = model.train(data=args.data, imgsz=imgsz, epochs=args.epochs, batch=bs, lr0=lr, iou=iou, project="runs/tune", name=name)
        results.append({"name": name, "lr": lr, "batch": bs, "iou": iou, "imgsz": imgsz})

    with open("runs/tune/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--data", default="data/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr0", nargs="+", type=float, default=[0.0005, 0.001, 0.003])
    ap.add_argument("--batch", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--iou", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[512, 640, 768])
    args = ap.parse_args()
    main(args)
