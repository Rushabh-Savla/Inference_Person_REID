from __future__ import annotations

import cv2
import numpy as np


def color(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros(20, np.float32)
    if image.ndim == 3 and image.shape[2] >= 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        gray = np.asarray(image, np.uint8)
        if gray.ndim != 2:
            return np.zeros(20, np.float32)
        hsv = cv2.cvtColor(gray, cv2.COLOR_GRAY2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    hue = hsv[..., 0].astype(np.float32) / 180.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    h, _ = np.histogram(hue, bins=12, range=(0.0, 1.0), weights=sat + 0.05)
    s, _ = np.histogram(sat, bins=4, range=(0.0, 1.0))
    v, _ = np.histogram(val, bins=4, range=(0.0, 1.0))
    out = np.concatenate([h, s, v]).astype(np.float32)
    return out / (np.linalg.norm(out) + 1e-12)


def pattern(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros(14, np.float32)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        return np.zeros(14, np.float32)
    gray = gray.astype(np.float32) / 255.0
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
    if image is None or image.size == 0:
        return np.zeros(6, np.float32)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        return np.zeros(6, np.float32)
    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape[:2]
    if h < 8 or w < 12:
        return np.zeros(6, np.float32)
    y1 = int(h * 0.42)
    y2 = max(int(h * 0.70), y1 + 1)
    band = gray[y1:y2]
    if band.ndim != 2 or band.shape[0] < 2 or band.shape[1] < 2:
        return np.zeros(6, np.float32)
    mid = max(1, band.shape[1] // 2)
    left = band[:, :mid]
    right = band[:, -mid:]
    line = float(np.mean(band < 0.28))
    edge = float(cv2.Laplacian(band, cv2.CV_32F).var() / 20.0)
    out = np.asarray([
        float(np.mean(left)),
        float(np.mean(right)),
        float(np.std(left)),
        float(np.std(right)),
        line,
        float(np.clip(edge, 0.0, 1.0)),
    ], np.float32)
    return out / (np.linalg.norm(out) + 1e-12)


def pack(person: np.ndarray, frame: np.ndarray, box) -> np.ndarray:
    if person is None or person.size == 0 or person.ndim < 2:
        return np.zeros(112, np.float32)
    h, w = person.shape[:2]
    if h < 40 or w < 20:
        return np.zeros(112, np.float32)

    upper = person[int(h * 0.12):max(int(h * 0.58), int(h * 0.12) + 1)]
    lower = person[int(h * 0.45):]
    headpart = person[:max(int(h * 0.34), 1)]
    eyepart = headpart

    uppercolor = color(upper)
    lowercolor = color(lower)
    upperpattern = pattern(upper)
    lowerpattern = pattern(lower)
    headdesc = head(headpart)
    eyedesc = eye(eyepart)

    frameh = int(frame.shape[0]) if frame is not None and frame.ndim >= 2 else 0
    framew = int(frame.shape[1]) if frame is not None and frame.ndim >= 2 else 0
    x1, y1, x2, y2 = [float(v) for v in box]
    full = max(1.0, y2 - y1)

    uppervis = 1.0 if frameh and framew and y1 > frameh * 0.015 and full >= 60.0 else -1.0
    lowervis = 1.0 if frameh and framew and y2 < frameh * 0.975 and full >= 100.0 else -1.0
    headvis = 1.0 if uppervis > 0 and headpart.shape[0] >= 24 and headpart.shape[1] >= 16 else -1.0
    eyevis = 1.0 if headvis > 0 and eyepart.shape[0] >= 24 and eyepart.shape[1] >= 28 else -1.0

    out = np.concatenate([
        uppercolor,
        lowercolor,
        upperpattern,
        lowerpattern,
        headdesc,
        eyedesc,
        np.asarray([uppervis, lowervis, headvis, eyevis], np.float32),
    ]).astype(np.float32)
    return out / (np.linalg.norm(out) + 1e-12)
