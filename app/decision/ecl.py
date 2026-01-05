from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.decision.trust import compute_rqi
from app.schemas import Bias, Conflict, DecisionStatus, Features, Safety

logger = logging.getLogger("gold_sentinel.ecl")


# -----------------------------
# Decision Explainability Capsule (DEC)
# -----------------------------


@dataclass(frozen=True)
class DecisionExplainabilityCapsule:
    decision_summary: str
    dominant_blocker: str
    secondary_blockers: list[str]
    what_would_change: list[str]


def _extract_patterns(refusal_reasons: list[str]) -> list[str]:
    patterns: list[str] = []
    for r in refusal_reasons:
        m = re.match(r"^([A-Z0-9_]+):", r.strip())
        if m:
            patterns.append(m.group(1))
    return list(dict.fromkeys(patterns))


def _dominant_blocker(patterns: list[str], conflicts: list[Conflict]) -> str:
    # Deterministic priority: any HIGH conflict first, then first refusal pattern, else fallback.
    for c in conflicts:
        if c.severity == "high":
            return c.name
    if patterns:
        return patterns[0]
    return "EVIDENCE_CONFLICT_CORE"


def _secondary_blockers(patterns: list[str], conflicts: list[Conflict], dominant: str) -> list[str]:
    secs: list[str] = []
    for c in conflicts:
        if c.name != dominant:
            secs.append(c.name)
    for p in patterns:
        if p != dominant and p not in secs:
            secs.append(p)
    return secs[:5]


def _what_would_change(patterns: list[str]) -> list[str]:
    """
    Deterministic re-entry conditions (exact conditions only).
    This is NOT a promise; it is a checklist of gates that must pass.
    """
    conds: list[str] = []

    # Base trade-eligibility gates (v1)
    # Note: these are conditions, not forecasts.
    base = [
        f"confidence >= {settings.t_trade}",
        "no high-severity conflicts",
        "plot_quality >= 0.60",
        "fit_quality >= 0.35",
        "direction stability across preprocessing variants == true",
    ]

    # Pattern-specific gating
    if "PLOT_PANE_NOT_FOUND" in patterns:
        conds.append("plot_bbox detected (chart pane isolated) == true")
    if "PLOT_QUALITY_BELOW_THRESHOLD" in patterns:
        conds.append("plot_quality >= 0.60")
    if "LOW_VISUAL_SIGNAL" in patterns:
        conds.append("edge_density >= 0.01")
    if "INSUFFICIENT_TRACE_POINTS" in patterns:
        conds.append("trace_points >= 40")
    if "INSUFFICIENT_TRACE_SPAN" in patterns:
        conds.append("trace_x_span_ratio >= 0.50")
    if "TREND_FIT_UNSTABLE" in patterns:
        conds.append("fit_quality >= 0.35")
    if "DIRECTION_UNSTABLE" in patterns:
        conds.append("direction stability across preprocessing variants == true")
    if "HTF_LTF_DIRECTION_CONFLICT" in patterns:
        conds.append("HTF bias == LTF bias")
        conds.append("no high-severity conflicts")
    if "HTF_NEUTRAL" in patterns:
        conds.append("HTF bias != neutral")
    if "CONFIDENCE_BELOW_TRADE_THRESHOLD" in patterns:
        conds.append(f"confidence >= {settings.t_trade}")

    # Always include base gates (deduped) for clarity on what is required.
    for b in base:
        if b not in conds:
            conds.append(b)

    return conds[:8]


def build_dec(status: DecisionStatus, bias: Bias, safety: Safety) -> DecisionExplainabilityCapsule:
    patterns = _extract_patterns(safety.refusal_reasons)
    dominant = _dominant_blocker(patterns, safety.conflicts)
    secondary = _secondary_blockers(patterns, safety.conflicts, dominant=dominant)

    if status == DecisionStatus.trade:
        summary = "Decision: TRADE. v1 safety gates are satisfied for this input."
    elif status == DecisionStatus.insufficient_data:
        summary = "Decision: INSUFFICIENT_DATA. The screenshot does not meet minimum evidence requirements."
    else:
        # NO_TRADE
        if bias == Bias.neutral:
            summary = "Decision: NO_TRADE. Directional edge is not validated under v1 safety gates."
        else:
            summary = "Decision: NO_TRADE. Evidence does not meet v1 safety gates."

    return DecisionExplainabilityCapsule(
        decision_summary=summary,
        dominant_blocker=dominant,
        secondary_blockers=secondary,
        what_would_change=_what_would_change(patterns),
    )


def render_dec(dec: DecisionExplainabilityCapsule) -> list[str]:
    lines: list[str] = []
    lines.append(f"DEC: {dec.decision_summary}")
    lines.append(f"DEC: dominant_blocker={dec.dominant_blocker}")
    if dec.secondary_blockers:
        lines.append("DEC: secondary_blockers=" + ", ".join(dec.secondary_blockers))
    if dec.what_would_change:
        lines.append("DEC: what_would_change (exact conditions):")
        for c in dec.what_would_change:
            lines.append(f"- {c}")
    return lines


# -----------------------------
# Re-entry Readiness Matrix (RRM)
# -----------------------------


def render_rrm(status: DecisionStatus, safety: Safety) -> list[str]:
    if status != DecisionStatus.no_trade:
        return []
    patterns = _extract_patterns(safety.refusal_reasons)
    conds = _what_would_change(patterns)
    return ["RRM: reconsider only if all conditions hold:"] + [f"- {c}" for c in conds]


# -----------------------------
# Capital Protection Score (CPS) (internal-only)
# -----------------------------


def compute_cps(status: DecisionStatus, features: Features, safety: Safety) -> float | None:
    """
    CPS ∈ [0,1], internal-only.
    Represents avoided risk when refusing (NO_TRADE / INSUFFICIENT_DATA).
    Must not influence decision fields.
    """
    if status == DecisionStatus.trade:
        return None

    plot_q = _extract_plot_quality(features.detectors.structure.notes)
    fit_q = float(features.trend_proxy.fit_quality)

    # Base avoided-risk proxy: lower evidence quality => higher avoided risk by refusing.
    base = 0.5 * (1.0 - plot_q) + 0.5 * (1.0 - fit_q)

    # Conflict severity increases avoided-risk (deterministic).
    bump = 0.0
    for c in safety.conflicts:
        if c.severity == "high":
            bump += 0.25
        elif c.severity == "medium":
            bump += 0.12
        else:
            bump += 0.05

    cps = max(0.0, min(1.0, base + bump))
    return float(cps)


# -----------------------------
# Missed Opportunity Firewall (MOF)
# -----------------------------


_FORBIDDEN_PHRASES = (
    "missed trade",
    "missed opportunity",
    "should have",
    "would have",
    "if you had",
    "regret",
    "hindsight",
    "price continued",
)


def enforce_mof(reasoning: list[str]) -> list[str]:
    """
    Deterministic language firewall: removes hindsight/regret language.
    """
    out: list[str] = []
    for line in reasoning:
        low = line.lower()
        if any(p in low for p in _FORBIDDEN_PHRASES):
            continue
        out.append(line)
    return out


# -----------------------------
# Investor-grade audit trace (internal-only)
# -----------------------------


def evidence_hash(features: Features, safety: Safety, status: DecisionStatus, bias: Bias, confidence: int) -> str:
    snapshot: dict[str, Any] = {
        "status": status.value,
        "bias": bias.value,
        "confidence": confidence,
        "market_phase": features.market_phase,
        "trend": {
            "direction": features.trend_proxy.direction,
            "strength": features.trend_proxy.strength,
            "fit_quality": features.trend_proxy.fit_quality,
        },
        "plot_quality": _extract_plot_quality(features.detectors.structure.notes),
        "refusal_patterns": _extract_patterns(safety.refusal_reasons),
        "conflicts": [{"name": c.name, "severity": c.severity} for c in safety.conflicts],
    }
    raw = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def blocker_vector(safety: Safety) -> list[str]:
    return _extract_patterns(safety.refusal_reasons) + [c.name for c in safety.conflicts]


def trust_snapshot(features: Features, safety: Safety, status: DecisionStatus, reasoning: list[str]) -> dict[str, Any]:
    pq = _extract_plot_quality(features.detectors.structure.notes)
    # Infer stability from presence of the DCS conflict.
    dcs_conflict = next((c for c in safety.conflicts if c.name == "decision_consistency_instability"), None)
    direction_stable = True
    phase_stable = True
    if dcs_conflict:
        ev = " ".join(dcs_conflict.evidence).lower()
        if "direction" in ev:
            direction_stable = False
        if "phase" in ev:
            phase_stable = False
    dcs = (0.6 if direction_stable else 0.0) + (0.4 if phase_stable else 0.0)

    return {
        "plot_quality": pq,
        "fit_quality": float(features.trend_proxy.fit_quality),
        "dcs": float(dcs),
        "rqi": compute_rqi(status, safety.refusal_reasons, reasoning),
    }


def emit_audit_trace(
    *,
    status: DecisionStatus,
    bias: Bias,
    confidence: int,
    features: Features,
    safety: Safety,
    reasoning: list[str],
) -> None:
    h = evidence_hash(features, safety, status, bias, confidence)
    vec = blocker_vector(safety)
    snap = trust_snapshot(features, safety, status, reasoning)
    cps = compute_cps(status, features, safety)
    payload = {
        "evidence_hash": h,
        "blocker_vector": vec,
        "trust_snapshot": snap,
        "cps": cps,
    }
    logger.info("audit_trace=%s", json.dumps(payload, sort_keys=True))


def _extract_plot_quality(notes: list[str]) -> float:
    for n in notes:
        if n.startswith("plot_quality="):
            try:
                return float(n.split("=", 1)[1])
            except Exception:
                return 0.0
    return 0.0

