from __future__ import annotations

import cv2
import numpy as np


def color(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    hue = hsv[..., 0].astype(np.float32) / 180.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    h, _ = np.histogram(hue, bins=12, range=(0.0, 1.0), weights=sat + 0.05)
    s, _ = np.histogram(sat, bins=4, range=(0.0, 1.0))
    v, _ = np.histogram(val, bins=4, range=(0.0, 1.0))
    out = np.concatenate([h, s, v]).astype(np.float32)
    return out / (np.linalg.norm(out) + 1e-12)


def pattern(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)
    ori, _ = np.histogram(ang, bins=8, range=(0.0, 1.0), weights=mag + 1e-4)
    bins, _ = np.histogram(mag, bins=4, range=(0.0, 1.5))
    density = np.asarray([float(np.mean(mag > 0.16)), float(np.std(gray))], np.float32)
    out = np.concatenate([ori.astype(np.float32), bins.astype(np.float32), density])
    return out / (np.linalg.norm(out) + 1e-12)


def head(image: np.ndarray) -> np.ndarray:
    return np.concatenate([color(image), pattern(image)]).astype(np.float32)


def eye(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape[:2]
    if h < 8 or w < 12:
        return np.zeros(6, np.float32)
    band = gray[int(h * 0.42):max(int(h * 0.70), int(h * 0.42) + 1)]
    mid = max(1, band.shape[1] // 2)
    left = band[:, :mid]
    right = band[:, -mid:]
    line = np.mean(band < 0.28)
    edge = cv2.Laplacian(band, cv2.CV_32F).var() / 20.0
    out = np.asarray([
        float(np.mean(left)),
        float(np.mean(right)),
        float(np.std(left)),
        float(np.std(right)),
        float(line),
        float(np.clip(edge, 0.0, 1.0)),
    ], np.float32)
    return out / (np.linalg.norm(out) + 1e-12)


def pack(person: np.ndarray, frame: np.ndarray, box) -> np.ndarray:
    h, w = person.shape[:2]
    if h < 40 or w < 20:
        return np.zeros(112, np.float32)
    upper = person[int(h * 0.12):max(int(h * 0.58), int(h * 0.12) + 1)]
    lower = person[int(h * 0.45):]
    headpart = person[:max(int(h * 0.34), 1)]
    eyepart = person[:max(int(h * 0.34), 1)]
    uppercolor = color(upper)
    lowercolor = color(lower)
    upperpattern = pattern(upper)
    lowerpattern = pattern(lower)
    headpart = head(headpart)
    eyepart = eye(eyepart)

    height, _ = frame.shape[:2]
    _, y1, _, y2 = [float(v) for v in box]
    full = max(1.0, float(y2 - y1))
    uppervis = 1.0 if y1 > height * 0.015 and full >= 60.0 else -1.0
    lowervis = 1.0 if y2 < height * 0.975 and full >= 100.0 else -1.0
    headvis = 1.0 if y1 > height * 0.015 and headpart.shape[0] >= 24 else -1.0
    eyevis = 1.0 if headvis > 0 and eyepart.shape[1] >= 28 else -1.0

    out = np.concatenate([
        uppercolor,
        lowercolor,
        upperpattern,
        lowerpattern,
        headpart,
        eyepart,
        np.asarray([uppervis, lowervis, headvis, eyevis], np.float32),
    ]).astype(np.float32)
    return out / (np.linalg.norm(out) + 1e-12)
