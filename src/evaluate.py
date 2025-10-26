import argparse
from ultralytics import YOLO

def main(args):
    model = YOLO(args.weights)
    # Test dataset defined in data yaml
    metrics = model.val(data=args.data, imgsz=args.imgsz, conf=0.25, iou=0.6)
    print(metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--data", default="data/dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    main(args)
