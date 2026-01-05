from __future__ import annotations

import cv2
import numpy as np


def estimate_plot_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Best-effort plot region detection.
    We look for the largest "structured" area by edge density, ignoring margins where UI labels usually live.

    Returns (x, y, w, h) in pixels, or None if uncertain.
    """
    h, w = gray.shape[:2]
    if h < 200 or w < 200:
        return None

    # Remove typical header/footer/side panels.
    margin_x = int(w * 0.08)
    margin_top = int(h * 0.10)
    margin_bottom = int(h * 0.05)
    roi = gray[margin_top : h - margin_bottom, margin_x : w - margin_x]

    # Edge map and morphology to find a coherent chart pane.
    edges = cv2.Canny(roi, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    x, y, bw, bh = cv2.boundingRect(contours[idx])

    # Map back to full image coords.
    x_full = x + margin_x
    y_full = y + margin_top

    # Reject tiny panes.
    if bw * bh < (w * h) * 0.15:
        return None

    return (int(x_full), int(y_full), int(bw), int(bh))

