from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.plot_region import estimate_plot_bbox
from app.vision.structure import StructureResult, analyze_structure


@dataclass(frozen=True)
class VisionBundle:
    plot_bbox: tuple[int, int, int, int] | None
    plot_quality: float  # 0..1
    struct_primary: StructureResult | None
    struct_alt: StructureResult | None
    direction_stable: bool
    phase_stable: bool
    notes: list[str]


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _plot_quality(image_shape: tuple[int, int], bbox: tuple[int, int, int, int] | None, edge_density: float) -> float:
    if bbox is None:
        return 0.0
    h, w = image_shape[:2]
    _, _, bw, bh = bbox
    area_ratio = float((bw * bh) / max(1.0, float(w * h)))
    # Conservative quality: chart pane should be sizeable, and there should be non-trivial edges.
    q_area = min(1.0, max(0.0, (area_ratio - 0.12) / 0.35))
    q_edges = min(1.0, max(0.0, (edge_density - 0.01) / 0.08))
    return float(0.65 * q_area + 0.35 * q_edges)


def extract_vision(bgr: np.ndarray) -> VisionBundle:
    """
    v1 vision extraction with robustness to light/dark themes.
    Runs structure extraction on two polarities and returns the best one + stability flags.
    """
    notes: list[str] = []
    gray = _to_gray(bgr)
    bbox = estimate_plot_bbox(gray=gray)
    if bbox is None:
        return VisionBundle(
            plot_bbox=None,
            plot_quality=0.0,
            struct_primary=None,
            struct_alt=None,
            direction_stable=False,
            phase_stable=False,
            notes=["plot_bbox_missing"],
        )

    # Variant A: normal
    s_a = analyze_structure(bgr=bgr, plot_bbox=bbox)

    # Variant B: inverted luminance (helps when candles are low-contrast vs background)
    bgr_inv = bgr.copy()
    bgr_inv[:, :, :] = 255 - bgr_inv[:, :, :]
    s_b = analyze_structure(bgr=bgr_inv, plot_bbox=bbox)

    r2_a = float(s_a.diagnostics.get("r2") or 0.0)
    r2_b = float(s_b.diagnostics.get("r2") or 0.0)

    primary, alt = (s_a, s_b) if r2_a >= r2_b else (s_b, s_a)

    def _dir(s: StructureResult) -> str:
        if s.slope is None:
            return "unknown"
        if s.slope <= -0.10:
            return "up"
        if s.slope >= 0.10:
            return "down"
        return "flat"

    dir_primary = _dir(primary)
    dir_alt = _dir(alt)
    direction_stable = dir_primary != "unknown" and dir_primary == dir_alt
    phase_stable = primary.phase != "unknown" and primary.phase == alt.phase

    edge_density = float(primary.diagnostics.get("edge_density") or 0.0)
    pq = _plot_quality(bgr.shape, bbox, edge_density=edge_density)

    if not direction_stable:
        notes.append("direction_unstable_across_preprocess_variants")
    if not phase_stable:
        notes.append("phase_unstable_across_preprocess_variants")

    return VisionBundle(
        plot_bbox=bbox,
        plot_quality=pq,
        struct_primary=primary,
        struct_alt=alt,
        direction_stable=direction_stable,
        phase_stable=phase_stable,
        notes=notes,
    )

