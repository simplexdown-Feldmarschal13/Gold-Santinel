from app.decision.engine import decide_mtf_v1, decide_v1
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


def test_mtf_direction_conflict_forces_no_trade_and_emits_high_conflict():
    htf = _vision_bundle(slope=-0.15, r2=0.75)  # bullish
    ltf = _vision_bundle(slope=0.15, r2=0.75)  # bearish
    d = decide_mtf_v1(htf, ltf)
    assert d.status == DecisionStatus.no_trade
    assert any(c.severity == "high" and c.name == "htf_ltf_direction_conflict" for c in d.safety.conflicts)


def test_mtf_htf_neutral_never_upgrades_to_trade():
    # LTF is not trade-eligible under v1; MTF must not upgrade it.
    htf = _vision_bundle(slope=0.0, r2=0.70)  # neutral-ish
    ltf = _vision_bundle(r2=0.25, plot_quality=0.95, edge_density=0.06, trace_points=220, trace_x_span_ratio=0.95)
    d_ltf = decide_v1(ltf)
    assert d_ltf.status != DecisionStatus.trade

    d = decide_mtf_v1(htf, ltf)
    assert d.status != DecisionStatus.trade


def test_mtf_confidence_monotonicity_direction_conflict_does_not_increase_confidence():
    htf = _vision_bundle(slope=-0.15, r2=0.75)
    ltf_aligned = _vision_bundle(slope=-0.15, r2=0.75)
    ltf_conflict = _vision_bundle(slope=0.15, r2=0.75)
    d_aligned = decide_mtf_v1(htf, ltf_aligned)
    d_conflict = decide_mtf_v1(htf, ltf_conflict)
    assert d_conflict.confidence <= d_aligned.confidence

