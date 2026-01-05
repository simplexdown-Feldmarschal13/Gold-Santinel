from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "Gold Sentinel"
    max_upload_mb: int = 15
    confidence_no_trade_threshold: int = 62


settings = Settings()
