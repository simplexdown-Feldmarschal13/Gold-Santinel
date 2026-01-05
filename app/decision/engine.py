from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.schemas import (
    Bias,
    Conflict,
    DecisionStatus,
    DetectorInfo,
    Detectors,
    Features,
    KeyLevel,
    LiquidityPool,
    PlotBBox,
    Safety,
    TrendProxy,
)
from app.vision.extract import VisionBundle


@dataclass(frozen=True)
class Decision:
    status: DecisionStatus
    bias: Bias
    confidence: int
    reasoning: list[str]
    features: Features
    safety: Safety


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def _conflict_penalty(severity: str) -> int:
    # v1 fixed penalties per spec
    if severity == "low":
        return 5
    if severity == "medium":
        return 12
    return 25  # high


def decide_v1(vision: VisionBundle) -> Decision:
    """
    v1 decision engine: deterministic confidence + explicit refusal gates.
    Traceable to spec sections:
    - Confidence formula + penalties
    - Mandatory refusal conditions
    - T_trade/T_dir gating
    """
    reasoning: list[str] = []
    refusal: list[str] = []
    conflicts: list[Conflict] = []

    # ---- Mandatory INSUFFICIENT_DATA gates ----
    if vision.plot_bbox is None or vision.struct_primary is None:
        refusal.append("Chart pane could not be isolated from the screenshot.")
        features = _empty_features_failed()
        return Decision(
            status=DecisionStatus.insufficient_data,
            bias=Bias.neutral,
            confidence=0,
            reasoning=["INSUFFICIENT DATA: " + refusal[0], "Action: upload a full chart pane with clear candles."],
            features=features,
            safety=Safety(refusal_reasons=refusal, conflicts=[]),
        )

    s = vision.struct_primary
    diag = s.diagnostics or {}
    edge_density = float(diag.get("edge_density") or 0.0)
    trace_points = int(diag.get("trace_points") or 0)
    fit_quality = float(diag.get("r2") or 0.0)
    trace_x_span_ratio = float(diag.get("trace_x_span_ratio") or 0.0)

    # Partial chart / insufficient signal gates (spec: partial charts, low signal, cannot compute trend proxy)
    if vision.plot_quality < 0.20:
        refusal.append("Chart pane quality is too low (low visibility or heavy obstruction).")
    if edge_density < 0.01:
        refusal.append("Insufficient visual signal (edge density too low).")
    if trace_points < 40:
        refusal.append("Trend proxy cannot be computed reliably (insufficient trace points).")
    if trace_x_span_ratio < 0.50:
        refusal.append("Partial chart detected (insufficient horizontal span to infer structure).")

    if refusal:
        features = _features_from_vision(vision, bias=Bias.neutral, confidence=0, conflicts=[])
        return Decision(
            status=DecisionStatus.insufficient_data,
            bias=Bias.neutral,
            confidence=min(25, _clamp_int(10 + 200 * edge_density, 0, 25)),
            reasoning=["INSUFFICIENT DATA: " + refusal[0]]
            + (["Action: upload a higher-resolution, full-width chart pane with minimal overlays."] if len(refusal) else []),
            features=features,
            safety=Safety(refusal_reasons=refusal, conflicts=[]),
        )

    # ---- Trend proxy interpretation ----
    # Note: image y axis increases downward; negative slope corresponds to rising price.
    direction = "unknown"
    if s.slope is None:
        direction = "unknown"
    elif s.slope <= -0.10:
        direction = "up"
    elif s.slope >= 0.10:
        direction = "down"
    else:
        direction = "flat"

    # Strength proxy: normalize slope magnitude and multiply by stability factor
    slope_abs = 0.0 if s.slope is None else float(abs(s.slope))
    slope_strength = min(1.0, max(0.0, slope_abs / 0.25))
    stability_factor = 1.0 if vision.direction_stable else 0.6
    strength = float(min(1.0, slope_strength * stability_factor))

    # Phase stability score
    phase_score = 1.0 if vision.phase_stable else 0.5

    # ---- Conflicts (explicit) ----
    if not vision.direction_stable:
        conflicts.append(
            Conflict(
                name="direction_unstable",
                severity="medium",
                evidence=["Direction differs across preprocessing variants."],
            )
        )
    if fit_quality < 0.20:
        conflicts.append(
            Conflict(
                name="weak_fit_quality",
                severity="high",
                evidence=[f"Trend proxy fit quality is weak (fit_quality={fit_quality:.2f})."],
            )
        )

    # ---- Bias (subject to directional gate) ----
    bias = Bias.neutral
    if direction == "up":
        bias = Bias.bullish
    elif direction == "down":
        bias = Bias.bearish
    else:
        bias = Bias.neutral

    # ---- Deterministic confidence (spec formula) ----
    # Evidence scores in [0,1]
    e1_plot = float(min(1.0, max(0.0, vision.plot_quality)))
    e2_fit = float(min(1.0, max(0.0, fit_quality)))
    e3_strength = float(min(1.0, max(0.0, strength)))
    e4_phase = float(min(1.0, max(0.0, phase_score)))
    extras = 0.0
    if len(s.key_levels_y) >= 2:
        extras += 0.6
    if len(s.liquidity_pools_y) >= 1:
        extras += 0.4
    e5_extras = float(min(1.0, max(0.0, extras)))

    # v1 fixed weights (implementation constants; tuned to prefer refusal over false trade)
    c_base = 10.0
    confidence_f = (
        c_base
        + 20.0 * e1_plot
        + 30.0 * e2_fit
        + 15.0 * e3_strength
        + 10.0 * e4_phase
        + 5.0 * e5_extras
    )

    penalty = 0
    for c in conflicts:
        penalty += _conflict_penalty(c.severity)
    confidence_f -= penalty

    confidence = _clamp_int(confidence_f, 0, 100)

    # Directional gate (spec T_dir): below this, bias forced neutral.
    if confidence < settings.t_dir:
        if bias != Bias.neutral:
            conflicts.append(
                Conflict(
                    name="direction_below_threshold",
                    severity="medium",
                    evidence=[f"Confidence below directional threshold (T_dir={settings.t_dir})."],
                )
            )
        bias = Bias.neutral

    # Reasoning (bounded to extracted features)
    if bias == Bias.bullish:
        reasoning.append("Directional bias: BULLISH (trend proxy indicates upward pressure).")
    elif bias == Bias.bearish:
        reasoning.append("Directional bias: BEARISH (trend proxy indicates downward pressure).")
    else:
        reasoning.append("Directional bias: NEUTRAL (insufficient directional edge).")

    reasoning.append(f"Trend proxy fit quality: {fit_quality:.2f}.")
    reasoning.append(f"Market phase estimate: {s.phase.upper()}.")

    # ---- TRADE eligibility gating (spec minimum evidence requirements) ----
    trade_blockers: list[str] = []
    if e1_plot < 0.60:
        trade_blockers.append("Plot isolation quality below minimum for TRADE.")
    if fit_quality < 0.35:
        trade_blockers.append("Trend proxy fit quality below minimum for TRADE.")
    if not vision.direction_stable:
        trade_blockers.append("Directional stability check failed across preprocessing variants.")
    if any(c.severity == "high" for c in conflicts):
        trade_blockers.append("High-severity conflict present.")

    status: DecisionStatus
    if trade_blockers:
        status = DecisionStatus.no_trade
    elif confidence >= settings.t_trade:
        status = DecisionStatus.trade
    else:
        status = DecisionStatus.no_trade

    if status == DecisionStatus.trade:
        reasoning.append(f"TRADE: confidence {confidence}% meets threshold (T_trade={settings.t_trade}).")
    else:
        reasoning.append(f"NO TRADE: confidence {confidence}% below threshold or blocked by safety gates.")
        refusal.extend(trade_blockers)
        if not refusal:
            refusal.append(f"Confidence {confidence}% below trade threshold (T_trade={settings.t_trade}).")

    # Final safety invariants (spec)
    if status == DecisionStatus.insufficient_data:
        bias = Bias.neutral
        confidence = min(confidence, 25)

    features = _features_from_vision(vision, bias=bias, confidence=confidence, conflicts=conflicts)
    return Decision(
        status=status,
        bias=bias,
        confidence=confidence,
        reasoning=reasoning,
        features=features,
        safety=Safety(refusal_reasons=refusal, conflicts=conflicts),
    )


def decide_mtf_v1(htf: VisionBundle, ltf: VisionBundle) -> Decision:
    """
    Multi-Timeframe Confluence Engine (MTF v1).

    Constraints (as per instruction prompt):
    - Deterministic only
    - No schema changes
    - Reuse existing vision outputs and v1 decision
    - Prefer NO_TRADE over false certainty
    """
    h = decide_v1(htf)
    l = decide_v1(ltf)

    # Output remains spec-shaped: we return LTF features as the execution context.
    features = l.features

    carried_conflicts = [*h.safety.conflicts, *l.safety.conflicts]
    mtf_conflicts: list[Conflict] = []
    refusal_reasons: list[str] = []
    reasoning: list[str] = []

    # ---- Directional Alignment Gate (HTF > LTF) ----
    direction_adjust = 0
    if h.bias == Bias.neutral:
        direction_adjust -= 10
        mtf_conflicts.append(
            Conflict(
                name="htf_neutral",
                severity="low",
                evidence=["HTF bias is NEUTRAL → confidence −10."],
            )
        )

    direction_conflict = (h.bias == Bias.bullish and l.bias == Bias.bearish) or (
        h.bias == Bias.bearish and l.bias == Bias.bullish
    )
    if direction_conflict:
        mtf_conflicts.append(
            Conflict(
                name="htf_ltf_direction_conflict",
                severity="high",
                evidence=[f"HTF={h.bias.value}, LTF={l.bias.value}."],
            )
        )

    # ---- Market Phase Compatibility ----
    h_phase = h.features.market_phase
    l_phase = l.features.market_phase

    phase_adjust = 0
    if h_phase == "expansion" and l_phase == "retracement":
        phase_adjust = 5
    elif h_phase == "expansion" and l_phase == "expansion":
        phase_adjust = -8
        mtf_conflicts.append(
            Conflict(
                name="phase_late_move_risk",
                severity="medium",
                evidence=["HTF expansion + LTF expansion → confidence −8 (late move risk)."],
            )
        )
    elif h_phase == "consolidation" and l_phase == "expansion":
        phase_adjust = -12
        mtf_conflicts.append(
            Conflict(
                name="phase_fake_breakout_risk",
                severity="medium",
                evidence=["HTF consolidation + LTF expansion → confidence −12 (fake breakout risk)."],
            )
        )

    # ---- Confidence Composition (Extend v1) ----
    # C_total = C_v1 + C_HTF_bonus + C_LTF_bonus − C_MTF_penalties
    # v1 MTF implementation:
    # - C_v1 = C_ltf
    # - C_HTF_bonus = round(2*(C_htf - C_ltf)/3)  (HTF weighs 2× LTF)
    # - C_LTF_bonus = 0
    # - C_MTF_penalties/bonuses are the explicit rule-table adjustments (direction_adjust, phase_adjust)
    c_ltf = int(l.confidence)
    c_htf = int(h.confidence)
    c_htf_bonus = _clamp_int(2.0 * (float(c_htf) - float(c_ltf)) / 3.0, -100, 100)

    confidence = _clamp_int(float(c_ltf + c_htf_bonus + direction_adjust + phase_adjust), 0, 100)

    # Any HTF conflict caps confidence at ≤55 (v1 MTF rule).
    # Interpreted as: any HIGH MTF directional conflict or any HIGH conflict already present in HTF decision.
    if direction_conflict or any(c.severity == "high" for c in h.safety.conflicts):
        confidence = min(confidence, 55)

    # ---- Status gating ----
    all_conflicts = [*carried_conflicts, *mtf_conflicts]
    high_conflict_present = any(c.severity == "high" for c in all_conflicts)

    if direction_conflict:
        status = DecisionStatus.no_trade
        refusal_reasons.append("HTF/LTF directional conflict detected.")
    elif h.status == DecisionStatus.insufficient_data or l.status == DecisionStatus.insufficient_data:
        status = DecisionStatus.insufficient_data
        refusal_reasons.append("One or more timeframes returned INSUFFICIENT_DATA.")
    elif h.status != DecisionStatus.trade or l.status != DecisionStatus.trade:
        status = DecisionStatus.no_trade
        refusal_reasons.append("One or more timeframes not trade-eligible under v1 safety gates.")
    elif high_conflict_present:
        status = DecisionStatus.no_trade
        refusal_reasons.append("High-severity conflict present.")
    elif confidence >= settings.t_trade:
        status = DecisionStatus.trade
    else:
        status = DecisionStatus.no_trade
        refusal_reasons.append(f"Confidence {confidence}% below trade threshold (T_trade={settings.t_trade}).")

    # Directional gate preserved (T_dir): below this, bias forced neutral.
    bias = l.bias if confidence >= settings.t_dir else Bias.neutral

    # ---- Reasoning (required, human-readable, auditable) ----
    reasoning.append("MTF mode enabled: combining HTF context with LTF execution readiness.")
    reasoning.append(f"HTF bias: {h.bias.value}.")
    reasoning.append(f"LTF bias: {l.bias.value}.")
    if h.bias == Bias.bullish:
        reasoning.append("HTF bullish structure intact")
    elif h.bias == Bias.bearish:
        reasoning.append("HTF bearish structure intact")

    if direction_conflict:
        reasoning.append("HTF/LTF directional conflict detected — trade refused")

    if h_phase and l_phase:
        reasoning.append(f"HTF phase: {h_phase}.")
        reasoning.append(f"LTF phase: {l_phase}.")

    if h_phase == "expansion" and l_phase == "retracement" and not direction_conflict:
        reasoning.append("LTF pullback aligns with HTF trend")

    if h.bias == Bias.neutral:
        reasoning.append("HTF neutral → confidence −10")
    if phase_adjust == 5:
        reasoning.append("HTF expansion + LTF retracement → +5 confidence")
    elif phase_adjust == -8:
        reasoning.append("HTF expansion + LTF expansion → −8 confidence (late move risk)")
    elif phase_adjust == -12:
        reasoning.append("HTF consolidation + LTF expansion → −12 confidence (fake breakout risk)")

    reasoning.append(
        f"Confidence composition: base(LTF)={c_ltf}%, HTF={c_htf}% (2× weight via bonus={c_htf_bonus}), direction_adj={direction_adjust}, phase_adj={phase_adjust}."
    )
    reasoning.append(f"Final confidence: {confidence}%.")
    reasoning.append(f"Final status: {status.value}.")

    # Safety: if not TRADE, refusal reasons must be present.
    if status != DecisionStatus.trade and not refusal_reasons:
        refusal_reasons = ["MTF safety gate triggered."]

    return Decision(
        status=status,
        bias=bias,
        confidence=confidence,
        reasoning=reasoning,
        features=features,
        safety=Safety(refusal_reasons=refusal_reasons, conflicts=all_conflicts),
    )


def _features_from_vision(vision: VisionBundle, bias: Bias, confidence: int, conflicts: list[Conflict]) -> Features:
    # Build spec-shaped features. v1 keeps key level/liquidity classification conservative ("unknown").
    assert vision.plot_bbox is not None and vision.struct_primary is not None
    x, y, w, h = vision.plot_bbox
    s = vision.struct_primary
    diag = s.diagnostics or {}
    fit_quality = float(diag.get("r2") or 0.0)

    # Trend direction per spec
    direction = "unknown"
    if s.slope is None:
        direction = "unknown"
    elif s.slope <= -0.10:
        direction = "up"
    elif s.slope >= 0.10:
        direction = "down"
    else:
        direction = "flat"

    slope_abs = 0.0 if s.slope is None else float(abs(s.slope))
    slope_strength = min(1.0, max(0.0, slope_abs / 0.25))
    stability_factor = 1.0 if vision.direction_stable else 0.6
    strength = float(min(1.0, slope_strength * stability_factor))

    structure_notes: list[str] = []
    structure_status: str = "ok"
    if fit_quality < 0.35 or not vision.direction_stable:
        structure_status = "weak"
    if fit_quality < 0.20:
        structure_status = "failed"
        structure_notes.append("Trend proxy fit quality below minimum.")
    structure_notes.append(f"plot_quality={vision.plot_quality:.2f}")
    for n in vision.notes:
        structure_notes.append(n)

    levels_status: str = "ok" if len(s.key_levels_y) >= 2 else "weak"
    liquidity_status: str = "ok" if len(s.liquidity_pools_y) >= 1 else "weak"

    key_levels = [KeyLevel(y_px=int(y_px), kind="unknown", strength=3) for y_px in s.key_levels_y[:6]]
    liquidity = [
        LiquidityPool(y_px=int(y_px), side="unknown", count=int(cnt)) for (y_px, cnt) in (s.liquidity_pools_y or [])[:3]
    ]

    return Features(
        plot_bbox=PlotBBox(x=int(x), y=int(y), w=int(w), h=int(h)),
        market_phase=s.phase if s.phase in ("expansion", "retracement", "consolidation") else "unknown",
        trend_proxy=TrendProxy(direction=direction, strength=strength, fit_quality=float(min(1.0, max(0.0, fit_quality)))),
        key_levels=key_levels,
        liquidity_pools=liquidity,
        detectors=Detectors(
            structure=DetectorInfo(status=structure_status, notes=structure_notes),
            levels=DetectorInfo(status=levels_status, notes=[]),
            liquidity=DetectorInfo(status=liquidity_status, notes=[]),
        ),
    )


def _empty_features_failed() -> Features:
    # For schema compliance on plot isolation failures, return a minimal placeholder bbox.
    # This path is only used for INSUFFICIENT_DATA responses where plot bbox is missing.
    return Features(
        plot_bbox=PlotBBox(x=0, y=0, w=0, h=0),
        market_phase="unknown",
        trend_proxy=TrendProxy(direction="unknown", strength=0.0, fit_quality=0.0),
        key_levels=[],
        liquidity_pools=[],
        detectors=Detectors(
            structure=DetectorInfo(status="failed", notes=["plot_bbox_missing"]),
            levels=DetectorInfo(status="failed", notes=[]),
            liquidity=DetectorInfo(status="failed", notes=[]),
        ),
    )

