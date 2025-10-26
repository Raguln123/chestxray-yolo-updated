import argparse
from ultralytics import YOLO

def main(args):
    model = YOLO(args.model)  # e.g., 'yolov8n.pt'
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        optimizer=args.optimizer,
        device=args.device,
        project="runs/detect",
        name="train",
        pretrained=True,
        mosaic=0.8 if args.augment else 0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        fliplr=0.5,
        flipud=0.0
    )
    print(results)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--data", default="data/dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--device", default="")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()
    main(args)
