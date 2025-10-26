from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import math

import joblib
import pandas as pd
from sqlalchemy import delete

from app.services.logic import compute_vaporix_features, fetch_vaporix_dataframe
from app.models.models import VaporixAlert, ScoringRun


@dataclass
class ScoringArtifacts:
    model_path: Path
    metrics_path: Path
    best_threshold: float
    numeric_features: list[str]
    categorical_features: list[str]


def load_artifacts(model_path: Path | str, metrics_path: Path | str) -> ScoringArtifacts:
    metrics_file = Path(metrics_path)
    with metrics_file.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    threshold = metrics.get("precision_recall_curve", {}).get("best_threshold", 0.5)
    numeric = metrics.get("numeric_features")
    categorical = metrics.get("categorical_features")
    if not isinstance(numeric, list) or not isinstance(categorical, list):
        raise ValueError("Metrics JSON must contain 'numeric_features' and 'categorical_features'.")

    return ScoringArtifacts(
        model_path=Path(model_path),
        metrics_path=metrics_file,
        best_threshold=float(threshold),
        numeric_features=[str(c) for c in numeric],
        categorical_features=[str(c) for c in categorical],
    )


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add missing columns with NA to keep pipeline compatible."""
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def prepare_feature_frame(df: pd.DataFrame, artifacts: ScoringArtifacts) -> pd.DataFrame:
    """
    Ensure that the feature DataFrame matches the expected columns of the trained model.
    """
    df = df.copy()
    df = _ensure_columns(df, artifacts.numeric_features + artifacts.categorical_features)
    feature_cols = artifacts.numeric_features + artifacts.categorical_features
    return df[feature_cols]


def select_latest_per_sensor(df: pd.DataFrame) -> pd.DataFrame:
    if "sensor_no" not in df.columns or df.empty:
        return df
    if "date" in df.columns:
        df_sorted = df.sort_values(["sensor_no", "date"])
    else:
        df_sorted = df.sort_values(["sensor_no"])
    return df_sorted.groupby("sensor_no", dropna=False).tail(1)


async def score_fault_risk(
    *,
    model_path: Path | str,
    metrics_path: Path | str,
    latest_only: bool = True,
    top_n: Optional[int] = None,
    threshold_override: Optional[float] = None,
    session=None,
) -> pd.DataFrame:
    """
    Load measurements, compute features, run the model, and return scored rows.
    """

    artifacts = load_artifacts(model_path, metrics_path)
    model = joblib.load(artifacts.model_path)

    if session is None:
        raise ValueError("An AsyncSession must be provided for scoring.")

    df_raw = await fetch_vaporix_dataframe(session)
    if df_raw.empty:
        return pd.DataFrame()

    features = compute_vaporix_features(df_raw)
    feature_frame = prepare_feature_frame(features, artifacts)

    # Align index for mapping back to metadata
    feature_frame = feature_frame.copy()
    feature_frame.index = features.index

    proba = model.predict_proba(feature_frame)[:, 1]
    decision_threshold = (
        artifacts.best_threshold if threshold_override is None else float(threshold_override)
    )
    proba = proba.astype(float)
    prediction = proba >= decision_threshold

    result = features.assign(
        fault_probability=proba,
        fault_prediction=prediction,
        decision_threshold=decision_threshold,
    )

    if latest_only:
        result = select_latest_per_sensor(result)

    result = result.sort_values("fault_probability", ascending=False)

    if top_n is not None:
        result = result.head(int(top_n))

    return result.reset_index(drop=True)


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, pd.Series):
        return value.iloc[0]
    if pd.isna(value):
        return None
    return value


async def persist_scores(
    session,
    scores: pd.DataFrame,
    *,
    model_path: Path | str,
    metrics_path: Path | str,
    warning_threshold: Optional[float] = None,
    critical_threshold: Optional[float] = None,
    log_run: bool = False,
) -> tuple[list[VaporixAlert], Optional[ScoringRun]]:
    """
    Speichert die übergebenen Scores in der DB und gibt die gespeicherten Datensätze zurück.
    """
    if scores.empty:
        return [], None

    saved: list[VaporixAlert] = []
    run_log: Optional[ScoringRun] = None
    if log_run:
        run_log = ScoringRun(
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )
        session.add(run_log)
        await session.flush()

    for _, row in scores.iterrows():
        sensor_no = _clean_value(row.get("sensor_no"))
        date_value = _clean_value(row.get("date"))
        if isinstance(date_value, pd.Timestamp):
            date_value = date_value.date()
        elif hasattr(date_value, "date"):
            date_value = date_value.date()

        if sensor_no and date_value:
            await session.execute(
                delete(VaporixAlert).where(
                    VaporixAlert.sensor_no == str(sensor_no),
                    VaporixAlert.date == date_value,
                )
            )

        probability = float(row["fault_probability"])
        severity = None
        if critical_threshold is not None and probability >= critical_threshold:
            severity = "critical"
        elif warning_threshold is not None and probability >= warning_threshold:
            severity = "warning"

        alert = VaporixAlert(
            sensor_no=str(sensor_no) if sensor_no is not None else None,
            date=date_value,
            fault_probability=probability,
            fault_prediction=bool(row["fault_prediction"]),
            fault_event=bool(row["fault_event"]) if not pd.isna(row.get("fault_event")) else None,
            fault_active=bool(row["fault_active"]) if not pd.isna(row.get("fault_active")) else None,
            days_since_last_fault=_clean_value(row.get("days_since_last_fault")),
            recovery_rate_mean=_clean_value(row.get("recovery_rate_mean")),
            temperature_c=_clean_value(row.get("temperature_c")),
            vapour_flow_rate=_clean_value(row.get("vapour_flow_rate")),
            source_file=_clean_value(row.get("source_file")),
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            severity=severity,
        )
        session.add(alert)
        saved.append(alert)

    await session.commit()
    if run_log is not None:
        run_log.alerts_saved = len(saved)
        run_log.finished_at = datetime.utcnow()
        await session.merge(run_log)
        await session.commit()
        await session.refresh(run_log)
    return saved, run_log
