from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.services.data_ingest import ingest_vaporix_directory
from app.services.notifications import notify_critical
from app.services.scoring import persist_scores, score_fault_risk


async def run_pipeline(
    session,
    *,
    data_dir: Optional[str | Path] = None,
    top_n: Optional[int] = None,
    latest_only: bool = False,
    threshold_override: Optional[float] = None,
    critical_threshold: Optional[float] = None,
    model_path: Optional[str | Path] = None,
    metrics_path: Optional[str | Path] = None,
) -> dict:
    settings = get_settings()
    directory = Path(data_dir or settings.data_directory)
    model_path = Path(model_path or "reports/models/fault_model_gbm.joblib")
    metrics_path = Path(metrics_path or "reports/fault_model_gbm_metrics.json")

    ingested = await ingest_vaporix_directory(session, directory)

    df = await score_fault_risk(
        model_path=model_path,
        metrics_path=metrics_path,
        latest_only=latest_only,
        top_n=top_n or settings.scoring_default_top_n,
        threshold_override=threshold_override,
        session=session,
    )

    alerts = []
    run_log = None
    if not df.empty:
        alerts, run_log = await persist_scores(
            session,
            df,
            model_path=model_path,
            metrics_path=metrics_path,
            warning_threshold=threshold_override if threshold_override is not None else settings.alert_threshold,
            critical_threshold=critical_threshold if critical_threshold is not None else settings.alert_critical_threshold,
            log_run=True,
        )
        await notify_critical(
            [
                {
                    "sensor_no": alert.sensor_no,
                    "fault_probability": alert.fault_probability,
                    "severity": alert.severity,
                    "date": alert.date,
                    "source_file": alert.source_file,
                }
                for alert in alerts
            ]
        )

    return {
        "ingested": ingested,
        "scored": len(df),
        "alerts_saved": len(alerts),
        "run_log_id": getattr(run_log, "id", None),
    }
