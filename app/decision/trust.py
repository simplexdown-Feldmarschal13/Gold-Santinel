from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from app.config import settings
from app.schemas import Bias, Conflict, DecisionStatus, Safety

logger = logging.getLogger("gold_sentinel.trust")


# -----------------------------
# Standardized refusal patterns
# -----------------------------


@dataclass(frozen=True)
class RefusalTemplate:
    pattern: str
    explanation: str
    action: str


REFUSAL_TEMPLATES: dict[str, RefusalTemplate] = {
    "PLOT_PANE_NOT_FOUND": RefusalTemplate(
        pattern="PLOT_PANE_NOT_FOUND",
        explanation="Chart pane could not be isolated from the screenshot.",
        action="Upload a full chart screenshot with the price pane clearly visible (avoid cropping UI panels over the chart).",
    ),
    "PLOT_QUALITY_BELOW_THRESHOLD": RefusalTemplate(
        pattern="PLOT_QUALITY_BELOW_THRESHOLD",
        explanation="Chart pane quality is below the minimum threshold for reliable structure extraction.",
        action="Re-upload a higher-resolution screenshot with clear candles and minimal overlays; ensure the chart pane occupies most of the image.",
    ),
    "INSUFFICIENT_TRACE_POINTS": RefusalTemplate(
        pattern="INSUFFICIENT_TRACE_POINTS",
        explanation="Insufficient trace points to compute a stable trend proxy.",
        action="Upload a clearer screenshot (higher resolution/contrast) showing candles across the full width of the pane.",
    ),
    "INSUFFICIENT_TRACE_SPAN": RefusalTemplate(
        pattern="INSUFFICIENT_TRACE_SPAN",
        explanation="Partial chart detected: insufficient horizontal span to infer structure.",
        action="Upload a full-width chart view (avoid zooming/cropping to the last few candles).",
    ),
    "LOW_VISUAL_SIGNAL": RefusalTemplate(
        pattern="LOW_VISUAL_SIGNAL",
        explanation="Insufficient visual signal for reliable extraction (low edge/contrast).",
        action="Increase chart contrast (theme or candle colors) and re-upload; avoid heavy compression.",
    ),
    "TREND_FIT_UNSTABLE": RefusalTemplate(
        pattern="TREND_FIT_UNSTABLE",
        explanation="Trend proxy fit quality is below the minimum reliability threshold.",
        action="Re-upload with clearer candles or a less cluttered chart; if possible, include more visible swings.",
    ),
    "DIRECTION_UNSTABLE": RefusalTemplate(
        pattern="DIRECTION_UNSTABLE",
        explanation="Direction is unstable across preprocessing variants; evidence is not consistent.",
        action="Re-upload with improved clarity; remove drawings/overlays that can distort edge extraction.",
    ),
    "EVIDENCE_CONFLICT_CORE": RefusalTemplate(
        pattern="EVIDENCE_CONFLICT_CORE",
        explanation="Core evidence is internally conflicting; the system refuses to average contradictions.",
        action="Wait for cleaner structure to form; re-check the chart and re-upload after consolidation resolves.",
    ),
    "HTF_LTF_DIRECTION_CONFLICT": RefusalTemplate(
        pattern="HTF_LTF_DIRECTION_CONFLICT",
        explanation="HTF and LTF directional biases are in conflict.",
        action="Do not force a trade. Wait for LTF to realign with HTF or reassess HTF structure with a cleaner screenshot.",
    ),
    "HTF_NEUTRAL": RefusalTemplate(
        pattern="HTF_NEUTRAL",
        explanation="HTF bias is neutral; higher timeframe does not validate direction.",
        action="Treat LTF moves as lower-conviction. Wait for clearer HTF structure or use a cleaner HTF screenshot and retry.",
    ),
    "CONFIDENCE_BELOW_TRADE_THRESHOLD": RefusalTemplate(
        pattern="CONFIDENCE_BELOW_TRADE_THRESHOLD",
        explanation="Confidence is below the trade threshold.",
        action="Treat this as context only. Wait for stronger, cleaner evidence and re-run analysis.",
    ),
    "MTF_SAFETY_GATE": RefusalTemplate(
        pattern="MTF_SAFETY_GATE",
        explanation="MTF safety gate blocked the trade due to insufficient confluence.",
        action="Seek clearer HTF context and LTF confirmation before re-running analysis.",
    ),
}


def format_refusal(pattern: str) -> str:
    t = REFUSAL_TEMPLATES.get(pattern)
    if not t:
        # Controlled fallback vocabulary
        t = RefusalTemplate(
            pattern="EVIDENCE_CONFLICT_CORE",
            explanation="Core evidence is insufficient or conflicting.",
            action="Re-upload a clearer chart and re-run analysis; if uncertainty persists, remain flat.",
        )
    return f"{t.pattern}: {t.explanation} Action: {t.action}"


# -----------------------------
# RQI (internal only)
# -----------------------------


_VAGUE_PHRASES = (
    "something",
    "issue",
    "weak",
    "unclear",
    "maybe",
    "possibly",
    "might",
    "could be",
    "likely",
)


def compute_rqi(status: DecisionStatus, refusal_reasons: list[str], reasoning: list[str]) -> int | None:
    """
    Refusal Quality Index (internal-only).
    RQI is computed for NO_TRADE and INSUFFICIENT_DATA only.
    """
    if status not in (DecisionStatus.no_trade, DecisionStatus.insufficient_data):
        return None

    score = 50

    # Quantity of refusal reasons
    score += min(20, 6 * len(refusal_reasons))

    # Actionability
    actionable = any("Action:" in r for r in refusal_reasons) or any("Action:" in s for s in reasoning)
    score += 15 if actionable else -15

    # Specificity: presence of standardized pattern tokens
    has_pattern = any(re.match(r"^[A-Z0-9_]+:", r.strip()) for r in refusal_reasons)
    score += 10 if has_pattern else -10

    # Penalize vagueness in refusal wording
    vagueness = 0
    joined = " ".join(refusal_reasons + reasoning).lower()
    for p in _VAGUE_PHRASES:
        if p in joined:
            vagueness += 1
    score -= min(20, 4 * vagueness)

    # Penalize hand-waving / near-trade implication
    if "near trade" in joined or "almost" in joined or "close to trade" in joined:
        score -= 15

    return int(max(0, min(100, score)))


# -----------------------------
# DCS (internal only)
# -----------------------------


def compute_dcs(direction_stable: bool, phase_stable: bool) -> float:
    """
    Decision Consistency Score (internal-only).
    Deterministic score based on agreement across preprocessing variants.
    """
    return float((0.6 if direction_stable else 0.0) + (0.4 if phase_stable else 0.0))


# -----------------------------
# Anti-hallucination guardrails
# -----------------------------


_STRONG_DIRECTION_PHRASES = (
    "bullish structure intact",
    "bearish structure intact",
    "upward pressure",
    "downward pressure",
    "trend is bullish",
    "trend is bearish",
    "strong trend",
)


def enforce_guardrails(
    *,
    reasoning: list[str],
    fit_quality: float,
    plot_quality: float,
    confidence: int,
) -> list[str]:
    """
    Deterministic text guardrails.
    - If fit_quality < 0.35: prohibit strong directional statements.
    - If plot_quality < 0.6: prohibit trend naming.
    - If confidence < T_trade: prohibit 'near trade' implications.
    """
    out: list[str] = []
    for line in reasoning:
        low = line.lower()

        if fit_quality < 0.35 and any(p in low for p in _STRONG_DIRECTION_PHRASES):
            continue

        if plot_quality < 0.6 and ("trend" in low or "structure intact" in low):
            continue

        if confidence < settings.t_trade and ("trade-eligible" in low or "near trade" in low or "almost" in low):
            continue

        out.append(line)

    # Ensure refusal outputs still have a clear summary line
    if not out:
        out = ["Decision rationale withheld due to low evidence quality."]
    return out


# -----------------------------
# Trust Telemetry (internal only)
# -----------------------------


_telemetry = {"TRADE": 0, "NO_TRADE": 0, "INSUFFICIENT_DATA": 0}


def record_telemetry(status: DecisionStatus) -> None:
    key = status.value
    if key in _telemetry:
        _telemetry[key] += 1
    logger.info("decision=%s totals=%s", key, dict(_telemetry))


# -----------------------------
# Decision shaping helpers
# -----------------------------


def ensure_refusal_quality(
    *,
    status: DecisionStatus,
    refusal_patterns: Iterable[str],
    existing_refusals: list[str],
    reasoning: list[str],
) -> tuple[list[str], list[str], int | None]:
    """
    Ensures refusal reasons use standardized patterns and include actionable advice.
    Returns (refusal_reasons, expanded_reasoning, rqi).
    """
    refusal_reasons = [format_refusal(p) for p in dict.fromkeys(refusal_patterns)]
    # Preserve any already-populated refusal strings if present (but still guarantee a pattern).
    for r in existing_refusals:
        if r and r not in refusal_reasons:
            refusal_reasons.append(r)

    # Guarantee at least one refusal reason for NO_TRADE/INSUFFICIENT_DATA
    if status in (DecisionStatus.no_trade, DecisionStatus.insufficient_data) and not refusal_reasons:
        refusal_reasons = [format_refusal("EVIDENCE_CONFLICT_CORE")]

    rqi = compute_rqi(status, refusal_reasons, reasoning)

    # If RQI is low, expand reasoning deterministically with explicit actions.
    expanded = list(reasoning)
    if rqi is not None and rqi < 70:
        expanded.append("Refusal rationale (standardized):")
        for rr in refusal_reasons[:5]:
            expanded.append(f"- {rr}")

    return refusal_reasons, expanded, rqi


def dcs_conflict_if_unstable(direction_stable: bool, phase_stable: bool) -> Conflict | None:
    if direction_stable and phase_stable:
        return None
    sev = "medium" if not direction_stable else "low"
    evidence: list[str] = []
    if not direction_stable:
        evidence.append("Direction disagrees across preprocessing variants.")
    if not phase_stable:
        evidence.append("Market phase disagrees across preprocessing variants.")
    return Conflict(name="decision_consistency_instability", severity=sev, evidence=evidence)

