import cv2
import numpy as np
from typing import Dict, Any, Tuple

def to_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img

def grayscale(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return (img * 255).astype(np.uint8)

def clahe(img, clip=2.0, tile=8):
    g = grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    out = clahe.apply(g)
    return out

def denoise(img, h=5):
    g = grayscale(img)
    return cv2.fastNlMeansDenoising(g, None, h, 7, 21)

def resize_pad(img, size=640):
    g = grayscale(img)
    h, w = g.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=resized.dtype)
    y0 = (size - nh)//2
    x0 = (size - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, (x0, y0, scale)

def full_pipeline(img, size=640, clahe_clip=2.0, clahe_tile=8, denoise_h=5):
    steps = {}
    steps["original"] = img
    steps["grayscale"] = grayscale(img)
    steps["normalized"] = normalize(steps["grayscale"])
    steps["clahe"] = clahe(steps["normalized"], clip=clahe_clip, tile=clahe_tile)
    steps["denoised"] = denoise(steps["clahe"], h=denoise_h)
    out, meta = resize_pad(steps["denoised"], size=size)
    steps["resized"] = out
    return steps, meta
