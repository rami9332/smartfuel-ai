from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """
    Einfache Konfiguration (über Umgebungsvariablen steuerbar).
    """

    alert_threshold: float = float(os.getenv("SMARTFUEL_ALERT_THRESHOLD", 0.4))
    alert_critical_threshold: float = float(os.getenv("SMARTFUEL_ALERT_CRITICAL_THRESHOLD", 0.7))
    scoring_default_top_n: int = int(os.getenv("SMARTFUEL_SCORING_TOP_N", 50))
    dashboard_limit: int = int(os.getenv("SMARTFUEL_DASHBOARD_LIMIT", 100))
    scoring_interval_minutes: int = int(os.getenv("SMARTFUEL_SCORING_INTERVAL_MIN", 30))
    data_directory: str = os.getenv("SMARTFUEL_DATA_DIR", "Data/Vaporix_side_a.csv")
    smtp_host: str = os.getenv("SMARTFUEL_SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMARTFUEL_SMTP_PORT", 587))
    smtp_username: str = os.getenv("SMARTFUEL_SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMARTFUEL_SMTP_PASSWORD", "")
    smtp_sender: str = os.getenv("SMARTFUEL_SMTP_SENDER", "alerts@example.com")
    smtp_recipients: str = os.getenv("SMARTFUEL_SMTP_RECIPIENTS", "")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
