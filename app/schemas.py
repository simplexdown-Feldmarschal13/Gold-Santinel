from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Bias(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class DecisionStatus(str, Enum):
    trade = "TRADE"
    no_trade = "NO_TRADE"
    insufficient_data = "INSUFFICIENT_DATA"


class KeyLevel(BaseModel):
    y_px: int = Field(..., description="Horizontal level in image pixel coordinates (top=0).")
    strength: int = Field(..., ge=1, le=5, description="Relative strength 1-5.")
    kind: Literal["support", "resistance", "unknown"] = "unknown"


class LiquidityPool(BaseModel):
    y_px: int
    count: int = Field(..., ge=2, description="How many swing touches cluster at this level.")
    side: Literal["buy_side", "sell_side", "unknown"] = "unknown"


class VisionFeatures(BaseModel):
    timeframe: str | None = None
    plot_bbox: tuple[int, int, int, int] | None = None  # x, y, w, h
    slope: float | None = None  # px per px (x->y)
    slope_strength: float | None = None  # 0..1-ish
    phase: Literal["expansion", "retracement", "consolidation", "unknown"] = "unknown"
    key_levels: list[KeyLevel] = []
    liquidity_pools: list[LiquidityPool] = []
    diagnostics: dict[str, Any] = {}


class AnalysisRequest(BaseModel):
    timeframe_hint: str | None = Field(
        default=None,
        description="Optional user-provided timeframe (e.g., D1/H4/H1/M15). Used when OCR is unreliable.",
    )


class AnalysisResponse(BaseModel):
    instrument: Literal["XAUUSD"] = "XAUUSD"
    bias: Bias
    confidence: int = Field(..., ge=0, le=100)
    status: DecisionStatus
    reasoning: list[str]
    features: VisionFeatures
