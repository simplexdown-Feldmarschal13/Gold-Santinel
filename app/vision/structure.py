from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StructureResult:
    slope: float | None
    slope_strength: float | None
    phase: str
    key_levels_y: list[int]
    liquidity_pools_y: list[tuple[int, int]]  # (y_px, count)
    diagnostics: dict


def _robust_line_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """
    Returns (slope, intercept, r2) for y = slope*x + intercept
    """
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    if len(x) < 5:
        return (0.0, float(np.mean(y) if len(y) else 0.0), 0.0)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2)) + 1e-9
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), float(r2)


def _cluster_1d(values: list[int], tol: int) -> list[list[int]]:
    if not values:
        return []
    values_sorted = sorted(values)
    clusters: list[list[int]] = [[values_sorted[0]]]
    for v in values_sorted[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def analyze_structure(bgr: np.ndarray, plot_bbox: tuple[int, int, int, int] | None) -> StructureResult:
    """
    Pure image-based, best-effort structure read:
    - trend proxy from edge-centroid regression across x
    - phase from slope magnitude + residual dispersion
    - key levels from horizontal line detection (Hough)
    - liquidity pools from repeated swing clusters (very conservative)
    """
    h, w = bgr.shape[:2]
    if plot_bbox is None:
        return StructureResult(
            slope=None,
            slope_strength=None,
            phase="unknown",
            key_levels_y=[],
            liquidity_pools_y=[],
            diagnostics={"reason": "plot_bbox_missing"},
        )

    x0, y0, bw, bh = plot_bbox
    plot = bgr[y0 : y0 + bh, x0 : x0 + bw]
    gray = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 50, 150)

    # Build a per-x "price trace proxy": take median y of edge pixels in each x column.
    ys = []
    xs = []
    for x in range(edge.shape[1]):
        col = edge[:, x]
        y_idx = np.where(col > 0)[0]
        if y_idx.size < max(3, int(edge.shape[0] * 0.01)):
            continue
        ys.append(int(np.median(y_idx)))
        xs.append(x)

    slope = None
    r2 = None
    slope_strength = None
    phase = "unknown"
    if len(xs) >= 40:
        slope, intercept, r2 = _robust_line_fit(np.array(xs), np.array(ys))
        # Normalize: stronger when r2 high and slope not near 0.
        slope_strength = float(min(1.0, max(0.0, r2)))

        # Phase heuristic: slope near 0 + noisy => consolidation; strong slope => expansion.
        slope_abs = abs(slope)
        if slope_abs < 0.05 and r2 < 0.25:
            phase = "consolidation"
        elif slope_abs >= 0.12 and r2 >= 0.35:
            phase = "expansion"
        else:
            phase = "retracement"

    # Horizontal levels via Hough lines in the plot.
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=120, minLineLength=int(bw * 0.25), maxLineGap=10)
    y_candidates: list[int] = []
    if lines is not None:
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            if abs(y2 - y1) <= 2:  # near-horizontal
                y_candidates.append(int((y1 + y2) / 2))

    # Cluster into distinct levels.
    clusters = _cluster_1d(y_candidates, tol=6)
    key_levels_y = [int(np.median(c)) for c in clusters if len(c) >= 2]

    # Liquidity pools (conservative): clusters of many edge-trace points at similar y.
    liquidity_pools: list[tuple[int, int]] = []
    if len(ys) >= 80:
        y_clusters = _cluster_1d(list(map(int, ys)), tol=5)
        for c in y_clusters:
            if len(c) >= max(10, int(len(ys) * 0.12)):
                liquidity_pools.append((int(np.median(c)), int(len(c))))
        # Keep top 3 by count.
        liquidity_pools.sort(key=lambda t: t[1], reverse=True)
        liquidity_pools = liquidity_pools[:3]

    diagnostics = {
        "plot_bbox": plot_bbox,
        "edge_density": float(np.mean(edge > 0)),
        "trace_points": int(len(xs)),
        "r2": None if r2 is None else float(r2),
    }

    return StructureResult(
        slope=None if slope is None else float(slope),
        slope_strength=slope_strength,
        phase=phase,
        key_levels_y=[int(y0 + y) for y in key_levels_y],
        liquidity_pools_y=[(int(y0 + y), int(c)) for (y, c) in liquidity_pools],
        diagnostics=diagnostics,
    )

