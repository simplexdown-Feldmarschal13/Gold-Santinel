from __future__ import annotations

import cv2
import numpy as np


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    # CLAHE tends to help with faint gridlines / candles.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (3, 3), 0)


def edges(gray: np.ndarray) -> np.ndarray:
    # thresholds tuned for screenshots; dynamic adjustment happens upstream via diagnostics.
    return cv2.Canny(gray, 50, 150)

