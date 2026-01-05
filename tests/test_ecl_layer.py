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


def test_every_no_trade_has_dec_and_rrm():
    v = _vision_bundle(r2=0.30, plot_quality=0.95, edge_density=0.06, trace_points=250, trace_x_span_ratio=0.95)
    d = decide_v1(v)
    assert d.status == DecisionStatus.no_trade
    joined = "\n".join(d.reasoning)
    assert "DEC:" in joined
    assert "RRM:" in joined


def test_cps_never_upgrades_decision_fields():
    # ECL must not change status/bias/confidence. This is enforced structurally:
    v = _vision_bundle(r2=0.30, plot_quality=0.95, edge_density=0.06, trace_points=250, trace_x_span_ratio=0.95)
    d = decide_v1(v)
    assert d.status == DecisionStatus.no_trade
    # CPS is internal-only and must never upgrade to TRADE.
    assert d.status != DecisionStatus.trade


def test_no_hindsight_bias_or_missed_trade_language():
    v = _vision_bundle(r2=0.30, plot_quality=0.95, edge_density=0.06, trace_points=250, trace_x_span_ratio=0.95)
    d = decide_v1(v)
    joined = " ".join(d.reasoning).lower() + " " + " ".join(d.safety.refusal_reasons).lower()
    forbidden = [
        "missed trade",
        "missed opportunity",
        "should have",
        "would have",
        "regret",
        "hindsight",
        "price continued",
    ]
    for f in forbidden:
        assert f not in joined


def test_mtf_no_trade_has_dec_and_rrm_and_no_missed_trade_language():
    htf = _vision_bundle(slope=-0.15, r2=0.75)
    ltf = _vision_bundle(slope=0.15, r2=0.75)
    d = decide_mtf_v1(htf, ltf)
    assert d.status == DecisionStatus.no_trade
    joined = "\n".join(d.reasoning).lower()
    assert "dec:" in joined
    assert "rrm:" in joined
    assert "missed trade" not in joined

