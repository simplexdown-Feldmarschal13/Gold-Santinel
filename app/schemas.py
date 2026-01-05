from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---- Enums (v1 spec) ----


class Bias(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class DecisionStatus(str, Enum):
    trade = "TRADE"
    no_trade = "NO_TRADE"
    insufficient_data = "INSUFFICIENT_DATA"


Timeframe = Literal["MN1", "W1", "D1", "H4", "H1", "M15", "M5", "unknown"]
TimeframeSource = Literal["ocr", "hint", "unknown"]
MarketPhase = Literal["expansion", "retracement", "consolidation", "unknown"]
TrendDirection = Literal["up", "down", "flat", "unknown"]
DetectorStatus = Literal["ok", "weak", "failed"]
LevelKind = Literal["support", "resistance", "unknown"]
LiquiditySide = Literal["buy_side", "sell_side", "unknown"]
ConflictSeverity = Literal["low", "medium", "high"]


# ---- Request contract (v1 spec) ----


class ClientContext(BaseModel):
    # Opaque audit/debug context; backend treats as optional metadata.
    platform: str | None = None
    broker: str | None = None
    theme: str | None = None
    resolution: str | None = None
    notes: str | None = None


# ---- Response contract (v1 spec) ----


class TimeframeInfo(BaseModel):
    detected: Timeframe
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: TimeframeSource


class PlotBBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class TrendProxy(BaseModel):
    direction: TrendDirection
    strength: float = Field(..., ge=0.0, le=1.0)
    fit_quality: float = Field(..., ge=0.0, le=1.0)


class KeyLevel(BaseModel):
    y_px: int
    kind: LevelKind
    strength: int = Field(..., ge=1, le=5)


class LiquidityPool(BaseModel):
    y_px: int
    side: LiquiditySide
    count: int = Field(..., ge=2)


class DetectorInfo(BaseModel):
    status: DetectorStatus
    notes: list[str] = []


class Detectors(BaseModel):
    structure: DetectorInfo
    levels: DetectorInfo
    liquidity: DetectorInfo


class Features(BaseModel):
    plot_bbox: PlotBBox
    market_phase: MarketPhase
    trend_proxy: TrendProxy
    key_levels: list[KeyLevel] = []
    liquidity_pools: list[LiquidityPool] = []
    detectors: Detectors


class Conflict(BaseModel):
    name: str
    severity: ConflictSeverity
    evidence: list[str] = []


class Safety(BaseModel):
    refusal_reasons: list[str] = []
    conflicts: list[Conflict] = []


class VersionInfo(BaseModel):
    api: Literal["v1"] = "v1"
    vision_model: str
    decision_model: str


class AnalysisResponse(BaseModel):
    analysis_id: str
    instrument: Literal["XAUUSD"] = "XAUUSD"
    timeframe: TimeframeInfo
    status: DecisionStatus
    bias: Bias
    confidence: int = Field(..., ge=0, le=100)
    reasoning: list[str]
    features: Features
    safety: Safety
    version: VersionInfo
