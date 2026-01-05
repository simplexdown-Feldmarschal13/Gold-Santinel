from app.decision.engine import decide_v1
from app.schemas import Bias, DecisionStatus
from app.vision.extract import VisionBundle
from app.vision.structure import StructureResult


def _vision_bundle(
    *,
    plot_bbox=(10, 10, 800, 500),
    plot_quality=0.9,
    r2=0.6,
    edge_density=0.05,
    trace_points=200,
    trace_x_span_ratio=0.9,
    slope=-0.15,
    phase="expansion",
    key_levels=3,
    liquidity=1,
    direction_stable=True,
    phase_stable=True,
):
    s = StructureResult(
        slope=slope,
        slope_strength=None,
        phase=phase,
        key_levels_y=[100 + i * 20 for i in range(key_levels)],
        liquidity_pools_y=[(200, 20)] if liquidity else [],
        diagnostics={
            "plot_bbox": plot_bbox,
            "edge_density": edge_density,
            "trace_points": trace_points,
            "trace_x_span_ratio": trace_x_span_ratio,
            "r2": r2,
        },
    )
    return VisionBundle(
        plot_bbox=plot_bbox,
        plot_quality=plot_quality,
        struct_primary=s,
        struct_alt=s,
        direction_stable=direction_stable,
        phase_stable=phase_stable,
        notes=[],
    )


def test_refusal_invariant_insufficient_data_is_neutral_and_low_confidence():
    v = VisionBundle(
        plot_bbox=None,
        plot_quality=0.0,
        struct_primary=None,
        struct_alt=None,
        direction_stable=False,
        phase_stable=False,
        notes=[],
    )
    d = decide_v1(v)
    assert d.status == DecisionStatus.insufficient_data
    assert d.bias == Bias.neutral
    assert d.confidence <= 25


def test_confidence_monotonicity_more_conflicts_never_increases_confidence():
    v_ok = _vision_bundle(direction_stable=True, r2=0.65)
    v_conflict = _vision_bundle(direction_stable=False, r2=0.65)
    d_ok = decide_v1(v_ok)
    d_conflict = decide_v1(v_conflict)
    assert d_conflict.confidence <= d_ok.confidence


def test_zero_trade_on_weak_evidence_fit_quality():
    v_weak = _vision_bundle(r2=0.10, plot_quality=0.95, edge_density=0.06, trace_points=220, trace_x_span_ratio=0.95)
    d = decide_v1(v_weak)
    assert d.status != DecisionStatus.trade

