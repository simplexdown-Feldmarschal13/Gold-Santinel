from app.decision.engine import decide_mtf_v1, decide_v1
from app.schemas import DecisionStatus
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
    direction_stable=True,
    phase_stable=True,
):
    s = StructureResult(
        slope=slope,
        slope_strength=None,
        phase=phase,
        key_levels_y=[120, 180, 240],
        liquidity_pools_y=[(260, 20)],
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


def test_every_no_trade_has_pattern_and_actionable_advice():
    # Force NO_TRADE via low fit quality (below 0.35 gate) but otherwise valid.
    v = _vision_bundle(r2=0.30, plot_quality=0.95, edge_density=0.06, trace_points=250, trace_x_span_ratio=0.95)
    d = decide_v1(v)
    assert d.status == DecisionStatus.no_trade
    assert len(d.safety.refusal_reasons) >= 1
    assert any(":" in r for r in d.safety.refusal_reasons)  # pattern prefix
    assert any("Action:" in r for r in d.safety.refusal_reasons)


def test_low_fit_quality_never_produces_strong_direction_language():
    v = _vision_bundle(r2=0.20, slope=-0.20, plot_quality=0.95, edge_density=0.06, trace_points=260, trace_x_span_ratio=0.95)
    d = decide_v1(v)
    joined = " ".join(d.reasoning).lower()
    assert "bullish structure intact" not in joined
    assert "bearish structure intact" not in joined
    assert "upward pressure" not in joined
    assert "downward pressure" not in joined


def test_confidence_below_trade_threshold_has_no_near_trade_language():
    v = _vision_bundle(r2=0.35, plot_quality=0.60, edge_density=0.02, trace_points=120, trace_x_span_ratio=0.80)
    d = decide_v1(v)
    if d.confidence < 62:
        joined = " ".join(d.reasoning).lower()
        assert "near trade" not in joined
        assert "trade-eligible" not in joined


def test_mtf_direction_conflict_emits_pattern_and_action():
    htf = _vision_bundle(slope=-0.15, r2=0.75)  # bullish
    ltf = _vision_bundle(slope=0.15, r2=0.75)  # bearish
    d = decide_mtf_v1(htf, ltf)
    assert d.status == DecisionStatus.no_trade
    assert any(r.startswith("HTF_LTF_DIRECTION_CONFLICT:") for r in d.safety.refusal_reasons)
    assert any("Action:" in r for r in d.safety.refusal_reasons)

