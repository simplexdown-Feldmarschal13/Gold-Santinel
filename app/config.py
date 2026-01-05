from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "Gold Sentinel"
    max_upload_mb: int = 15
    # v1 thresholds (frozen by spec)
    t_trade: int = 62
    t_dir: int = 40
    # version pins (must be returned in API output)
    vision_model: str = "vision_v1.0.0"
    decision_model: str = "decision_v1.0.0"


settings = Settings()
