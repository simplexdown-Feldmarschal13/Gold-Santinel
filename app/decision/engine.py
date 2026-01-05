from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.schemas import Bias, DecisionStatus, KeyLevel, LiquidityPool, VisionFeatures
from app.vision.structure import StructureResult


@dataclass(frozen=True)
class Decision:
    bias: Bias
    confidence: int
    status: DecisionStatus
    reasoning: list[str]
    features: VisionFeatures


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def decide(features: VisionFeatures) -> Decision:
    """
    Deterministic, explainable decision policy.

    Rules:
    - If plot detection/trace is weak -> INSUFFICIENT_DATA (no guessing)
    - Bias mostly derived from trend proxy slope sign (y increases down in images)
    - Confidence derived from evidence quality and consistency, not from bias strength alone
    - Below threshold -> NO_TRADE
    """
    r: list[str] = []

    diag = features.diagnostics or {}
    edge_density = float(diag.get("edge_density") or 0.0)
    trace_points = int(diag.get("trace_points") or 0)
    r2 = diag.get("r2")
    r2f = float(r2) if isinstance(r2, (int, float)) else 0.0

    # Evidence quality gates
    if features.plot_bbox is None:
        return Decision(
            bias=Bias.neutral,
            confidence=0,
            status=DecisionStatus.insufficient_data,
            reasoning=["INSUFFICIENT DATA: chart pane could not be isolated from the screenshot."],
            features=features,
        )

    if trace_points < 40 or edge_density < 0.01:
        return Decision(
            bias=Bias.neutral,
            confidence=_clamp_int(10 + 20 * edge_density, 0, 25),
            status=DecisionStatus.insufficient_data,
            reasoning=[
                "INSUFFICIENT DATA: the price trace could not be extracted reliably from this screenshot.",
                "Action: upload a higher-resolution chart with clear candles and minimal UI overlays.",
            ],
            features=features,
        )

    # Bias from slope sign (image y axis down => negative slope means rising prices)
    bias = Bias.neutral
    if features.slope is not None:
        if features.slope <= -0.10:
            bias = Bias.bullish
        elif features.slope >= 0.10:
            bias = Bias.bearish
        else:
            bias = Bias.neutral

    if bias == Bias.bullish:
        r.append("Trend proxy indicates upward pressure (price trace slopes up).")
    elif bias == Bias.bearish:
        r.append("Trend proxy indicates downward pressure (price trace slopes down).")
    else:
        r.append("Trend proxy is flat/unclear; directional edge is limited.")

    if features.phase != "unknown":
        r.append(f"Current phase estimate: {features.phase.upper()}.")

    if features.key_levels:
        r.append(f"Detected {len(features.key_levels)} candidate key horizontal levels (context-only).")
    if features.liquidity_pools:
        r.append(f"Detected {len(features.liquidity_pools)} candidate liquidity clusters (context-only).")

    # Confidence scoring: evidence quality + structure consistency.
    # Base from r2 and trace size; add modest boosts for contextual features.
    confidence = 35 + 40 * r2f
    confidence += min(10, trace_points / 25)  # saturates quickly
    if features.phase == "expansion":
        confidence += 6
    if len(features.key_levels) >= 2:
        confidence += 4
    if len(features.liquidity_pools) >= 1:
        confidence += 3

    confidence_i = _clamp_int(confidence, 0, 100)

    # Hard safety: if r2 is weak, refuse to be directional.
    if r2f < 0.20:
        bias = Bias.neutral
        r.append("Model safety gate: structure fit quality is weak; defaulting to NEUTRAL.")

    if confidence_i < settings.confidence_no_trade_threshold:
        status = DecisionStatus.no_trade
        r.append(
            f"NO TRADE: confidence {confidence_i}% is below threshold ({settings.confidence_no_trade_threshold}%)."
        )
    else:
        status = DecisionStatus.trade
        r.append(f"Trade-eligible bias (confidence {confidence_i}%). Execution rules are external to this module.")

    return Decision(
        bias=bias,
        confidence=confidence_i,
        status=status,
        reasoning=r,
        features=features,
    )


def from_structure(struct: StructureResult, timeframe: str | None) -> VisionFeatures:
    key_levels = [KeyLevel(y_px=y, strength=3, kind="unknown") for y in struct.key_levels_y[:6]]
    liquidity = [
        LiquidityPool(y_px=y, count=c, side="unknown") for (y, c) in (struct.liquidity_pools_y or [])[:3]
    ]
    return VisionFeatures(
        timeframe=timeframe,
        plot_bbox=struct.diagnostics.get("plot_bbox"),
        slope=struct.slope,
        slope_strength=struct.slope_strength,
        phase=struct.phase,
        key_levels=key_levels,
        liquidity_pools=liquidity,
        diagnostics=struct.diagnostics,
    )

