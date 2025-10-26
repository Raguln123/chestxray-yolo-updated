import argparse, os
from ultralytics import YOLO

def main(args):
    model = YOLO(args.weights)
    results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, save=args.save, project=args.project, name=args.name)
    for r in results:
        print(r.path, r.boxes)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--source", default="data/yolo/images/test")  # file/dir
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--project", default="runs/infer")
    ap.add_argument("--name", default="pred")
    args = ap.parse_args()
    main(args)
