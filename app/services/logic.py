from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import VaporixMeasurement


@dataclass
class VaporixLabelConfig:
    """Parameter für einfache Heuristik-basierte Label."""

    min_recovery_mean: float = 95.0          # Prozent
    max_temperature_c: float = 35.0          # °C
    min_vapour_flow: float = 5.0            # l/min
    recovery_drop_threshold: float = -2.0   # Prozentpunkte ggü. Vortag


async def fetch_vaporix_dataframe(db: AsyncSession, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Lädt Messdaten aus der DB und gibt sie als DataFrame zurück.
    """

    stmt = select(VaporixMeasurement).order_by(
        VaporixMeasurement.sensor_no, VaporixMeasurement.date
    )
    if limit:
        stmt = stmt.limit(limit)

    res = await db.execute(stmt)
    rows = res.scalars().all()

    records: List[Dict] = []
    for row in rows:
        payload = row.__dict__.copy()
        payload.pop("_sa_instance_state", None)
        records.append(payload)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    date_cols = ["date", "created_at"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    numeric_cols = [
        "error_counter",
        "fuel_transfers",
        "recovery_rate",
        "recovery_rate_mean",
        "temperature_c",
        "vc_status",
        "vapour_flow_rate",
        "fuel_flow_rate",
        "recovery_rate_pos",
        "recovery_rate_neg",
        "pulse_rate",
        "pcm",
        "flg",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_vaporix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erweitert den DataFrame um Gleitmittelwerte & Differenzen.
    """

    if df.empty:
        return df

    data = df.sort_values(["sensor_no", "date"]).copy()
    grouped = data.groupby("sensor_no", dropna=False)

    def _rolling_mean(series: pd.Series) -> pd.Series:
        return series.rolling(window=3, min_periods=1).mean()

    for col, target in [
        ("recovery_rate", "rec_rate_ma3"),
        ("recovery_rate_mean", "rec_rate_mean_ma3"),
        ("temperature_c", "temperature_ma3"),
        ("vapour_flow_rate", "vapour_flow_ma3"),
    ]:
        if col in data.columns:
            data[target] = grouped[col].transform(_rolling_mean)

    # Längere Gleitfenster (7) und Standardabweichungen
    if "recovery_rate_mean" in data.columns:
        data["rec_rate_mean_ma7"] = grouped["recovery_rate_mean"].transform(
            lambda s: s.rolling(window=7, min_periods=1).mean()
        )
        data["rec_rate_mean_std3"] = grouped["recovery_rate_mean"].transform(
            lambda s: s.rolling(window=3, min_periods=2).std()
        )
        data["rec_rate_mean_std7"] = grouped["recovery_rate_mean"].transform(
            lambda s: s.rolling(window=7, min_periods=2).std()
        )

    if "temperature_c" in data.columns:
        data["temperature_ma7"] = grouped["temperature_c"].transform(
            lambda s: s.rolling(window=7, min_periods=1).mean()
        )
        data["temperature_std3"] = grouped["temperature_c"].transform(
            lambda s: s.rolling(window=3, min_periods=2).std()
        )

    if "vapour_flow_rate" in data.columns:
        data["vapour_flow_ma7"] = grouped["vapour_flow_rate"].transform(
            lambda s: s.rolling(window=7, min_periods=1).mean()
        )
        data["vapour_flow_std3"] = grouped["vapour_flow_rate"].transform(
            lambda s: s.rolling(window=3, min_periods=2).std()
        )

    # Lags und Differenzen
    for col in [
        "recovery_rate_mean",
        "recovery_rate",
        "temperature_c",
        "vapour_flow_rate",
        "fuel_flow_rate",
    ]:
        if col in data.columns:
            data[f"{col}_lag1"] = grouped[col].shift(1)
            data[f"{col}_lag3"] = grouped[col].shift(3)
            data[f"{col}_diff1"] = data[col] - data[f"{col}_lag1"]

    if "recovery_rate_mean" in data.columns:
        data["recovery_drop"] = grouped["recovery_rate_mean"].transform(lambda s: s.diff())
    else:
        data["recovery_drop"] = pd.NA

    if "error_counter" in data.columns:
        data["error_delta"] = grouped["error_counter"].transform(lambda s: s.diff().fillna(0))
        data["fault_event"] = data["error_delta"] > 0
        data["fault_active"] = data["error_counter"].fillna(0) > 0
    else:
        data["error_delta"] = 0
        data["fault_event"] = False
        data["fault_active"] = False

    if "date" in data.columns:
        data["days_since_prev"] = grouped["date"].diff().dt.days
    else:
        data["days_since_prev"] = pd.NA

    # Zeit seit letztem Fault-Event je Sensor
    if "date" in data.columns:
        date_series = pd.to_datetime(data["date"])
        last_fault = (
            date_series.where(data["fault_event"])
            .groupby(data["sensor_no"], dropna=False)
            .ffill()
        )
        data["days_since_last_fault"] = (date_series - last_fault).dt.days
    else:
        data["days_since_last_fault"] = pd.NA

    # Rolling Fault-Kennzahlen
    data["fault_event"] = data["fault_event"].astype(bool)
    data["fault_active"] = data["fault_active"].astype(bool)
    data["fault_event_roll7"] = grouped["fault_event"].transform(
        lambda s: s.astype(int).rolling(window=7, min_periods=1).sum()
    )
    data["fault_event_roll30"] = grouped["fault_event"].transform(
        lambda s: s.astype(int).rolling(window=30, min_periods=1).sum()
    )
    data["fault_active_roll3"] = grouped["fault_active"].transform(
        lambda s: s.astype(int).rolling(window=3, min_periods=1).sum()
    )

    return data


def label_vaporix_anomalies(
    df: pd.DataFrame, config: Optional[VaporixLabelConfig] = None
) -> pd.DataFrame:
    """
    Kennzeichnet Zeilen mit einfacher Heuristik als Warnung/Anomalie.
    """

    if df.empty:
        return df

    cfg = config or VaporixLabelConfig()
    data = df.copy()

    def _bool_series(condition: pd.Series) -> pd.Series:
        return condition.fillna(False)

    cond_error = _bool_series(data["fault_active"])
    cond_fault_event = _bool_series(data["fault_event"])
    cond_recovery_low = _bool_series(data["recovery_rate_mean"] < cfg.min_recovery_mean)
    cond_recovery_drop = _bool_series(data["recovery_drop"] <= cfg.recovery_drop_threshold)
    cond_temp_high = _bool_series(data["temperature_c"] >= cfg.max_temperature_c)
    cond_flow_low = _bool_series(data["vapour_flow_rate"] <= cfg.min_vapour_flow)

    reason_map = {
        "fault_event": cond_fault_event,
        "fault_active": cond_error,
        "low_recovery": cond_recovery_low,
        "recovery_drop": cond_recovery_drop,
        "high_temperature": cond_temp_high,
        "low_vapour_flow": cond_flow_low,
    }

    alert_reasons: List[List[str]] = []
    anomaly_score: List[int] = []
    for idx in range(len(data)):
        reasons_for_row = [key for key, series in reason_map.items() if series.iloc[idx]]
        alert_reasons.append(reasons_for_row)
        score = len(reasons_for_row) + (2 if cond_fault_event.iloc[idx] else 0)
        anomaly_score.append(score)

    data["alert_reasons"] = alert_reasons
    data["anomaly_score"] = anomaly_score
    data["is_alert"] = data["alert_reasons"].apply(bool)
    data["alert_reason_str"] = data["alert_reasons"].apply(lambda items: ", ".join(items))

    def _severity(row: pd.Series) -> str:
        if row.get("fault_event"):
            return "critical"
        if row.get("fault_active"):
            return "high"
        score = row.get("anomaly_score", 0) or 0
        if score >= 3:
            return "high"
        if score == 2:
            return "medium"
        if score == 1:
            return "low"
        return "normal"

    data["severity"] = data.apply(_severity, axis=1)
    return data


def summarise_vaporix_health(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Ergebnisse je Sensor für ein schnelles Health-Dashboard.
    """

    if df.empty:
        return df

    grouped = df.groupby("sensor_no", dropna=False)
    summary = grouped.agg(
        total_rows=("id", "count"),
        alerts=("is_alert", "sum"),
        fault_events=("fault_event", "sum"),
        fault_active_rows=("fault_active", "sum"),
        min_recovery=("recovery_rate_mean", "min"),
        mean_recovery=("recovery_rate_mean", "mean"),
        max_temperature=("temperature_c", "max"),
        error_events=("error_counter", lambda s: (s.fillna(0) > 0).sum()),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )

    summary["alert_ratio"] = summary["alerts"] / summary["total_rows"]
    alert_dates = (
        df[df["is_alert"]].groupby("sensor_no", dropna=False)["date"].max()
        if "is_alert" in df.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    summary["last_alert"] = summary.index.map(alert_dates)
    return summary.reset_index()
