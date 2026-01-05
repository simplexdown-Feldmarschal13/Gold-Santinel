from app.decision.engine import decide
from app.schemas import DecisionStatus, VisionFeatures


def test_insufficient_data_when_no_plot_bbox():
    d = decide(VisionFeatures(plot_bbox=None, diagnostics={}))
    assert d.status == DecisionStatus.insufficient_data


def test_no_trade_when_low_confidence():
    f = VisionFeatures(
        plot_bbox=(0, 0, 500, 300),
        slope=0.0,
        phase="consolidation",
        diagnostics={"edge_density": 0.02, "trace_points": 45, "r2": 0.1, "plot_bbox": (0, 0, 500, 300)},
    )
    d = decide(f)
    assert d.status in (DecisionStatus.no_trade, DecisionStatus.insufficient_data)

