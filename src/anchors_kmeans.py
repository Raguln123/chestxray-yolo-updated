import argparse
import glob
import numpy as np

def load_boxes(labels_glob):
    boxes = []
    for p in glob.glob(labels_glob):
        with open(p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                _, _, _, w, h = map(float, parts)
                boxes.append([w, h])
    return np.array(boxes)

def kmeans(data, k=9, iters=100):
    # Simple IoU-based kmeans for anchors (wh space)
    centers = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(iters):
        ious = 1 - np.minimum(data[:,None,:]/centers[None,:,:], centers[None,:,:]/data[:,None,:]).min(axis=2)
        labels = np.argmin(ious, axis=1)
        for i in range(k):
            pts = data[labels==i]
            if len(pts): centers[i] = pts.mean(axis=0)
    return centers

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-glob", default="data/yolo/labels/train/*.txt")
    ap.add_argument("--k", type=int, default=9)
    args = ap.parse_args()
    boxes = load_boxes(args.labels_glob)
    centers = kmeans(boxes, k=args.k)
    print("Anchors (w,h normalized to img):")
    for c in centers:
        print(f"{c[0]:.4f},{c[1]:.4f}")
