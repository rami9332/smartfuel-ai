# app/main.py
import math
import os
from datetime import datetime, timedelta
from typing import Any, Optional, List
from fastapi import FastAPI, Depends, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, and_, func, text, or_
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import engine, get_db
from app.models.models import Base, PumpData, VaporixAlert, ScoringRun, VaporixMeasurement
from app.schemas.scoring import ScoreResult
from app.services.notifications import notify_critical
from app.services.scoring import persist_scores, score_fault_risk


def sanitize_floats(data):
    if isinstance(data, dict):
        return {k: sanitize_floats(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_floats(v) for v in data]
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if hasattr(data, "isoformat"):
        try:
            return data.isoformat()
        except Exception:
            pass
    if hasattr(data, "item"):
        try:
            return sanitize_floats(data.item())
        except Exception:
            pass
    return data


def dataframe_to_dicts(df):
    if df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        records.append(sanitize_floats({col: row[col] for col in df.columns}))
    return records


def alert_to_dict(alert: VaporixAlert) -> dict:
    return sanitize_floats(
        {
            "id": alert.id,
            "sensor_no": alert.sensor_no,
            "date": alert.date,
            "fault_probability": alert.fault_probability,
            "fault_prediction": alert.fault_prediction,
            "fault_event": alert.fault_event,
            "fault_active": alert.fault_active,
            "days_since_last_fault": alert.days_since_last_fault,
            "recovery_rate_mean": alert.recovery_rate_mean,
            "temperature_c": alert.temperature_c,
            "vapour_flow_rate": alert.vapour_flow_rate,
            "source_file": alert.source_file,
            "model_path": alert.model_path,
            "metrics_path": alert.metrics_path,
            "severity": alert.severity,
            "scored_at": alert.scored_at,
        }
    )


def summarize_alerts(alerts: list[dict]) -> dict:
    total = len(alerts)
    critical = sum(1 for a in alerts if a.get("severity") == "critical")
    warning = sum(1 for a in alerts if a.get("severity") == "warning")
    info = total - critical - warning
    return {
        "total": total,
        "critical": critical,
        "warning": warning,
        "info": info,
        "threshold": settings.alert_threshold,
        "critical_threshold": settings.alert_critical_threshold,
    }

# ------ Security (einfacher Header-Key) ------
API_KEY = os.getenv("API_KEY", "dev-1234")

async def require_key(request: Request):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ------ App ------
settings = get_settings()

app = FastAPI(title="SmartFuel AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    # Tabellen erzeugen (nur beim ersten Start relevant)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Ensure new columns exist (SQLite ALTER TABLE ADD COLUMN if missing)
        for stmt in [
            "ALTER TABLE vaporix_alerts ADD COLUMN severity TEXT",
            "ALTER TABLE vaporix_alerts ADD COLUMN model_path TEXT",
            "ALTER TABLE vaporix_alerts ADD COLUMN metrics_path TEXT",
            "ALTER TABLE vaporix_alerts ADD COLUMN scored_at DATETIME"
        ]:
            try:
                await conn.execute(text(stmt))
            except OperationalError:
                pass

@app.get("/")
def root():
    return {"message": "SmartFuel AI Backend läuft!"}

@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    total_alerts = await db.execute(
        select(func.count()).select_from(VaporixAlert)
    )
    total_measurements = await db.execute(
        select(func.count()).select_from(VaporixMeasurement)
    )
    critical_count = await db.execute(
        select(func.count()).where(VaporixAlert.severity == "critical")
    )
    last_run = await db.execute(
        select(ScoringRun.started_at, ScoringRun.alerts_saved)
        .order_by(desc(ScoringRun.started_at))
        .limit(1)
    )
    last_run_row = last_run.first()

    return {
        "status": "ok",
        "alerts": total_alerts.scalar() or 0,
        "measurements": total_measurements.scalar() or 0,
        "critical_alerts": critical_count.scalar() or 0,
        "last_run": {
            "started_at": last_run_row[0] if last_run_row else None,
            "alerts_saved": last_run_row[1] if last_run_row else None,
        },
    }

# ------ Schemas ------
class PumpDataIn(BaseModel):
    pump_id: str = Field(..., example="HH-Station-03")
    temperature: Optional[float] = Field(None, example=41.2)
    pressure:    Optional[float] = Field(None, example=6.8)
    vibration:   Optional[float] = Field(None, example=2.1)
    status:      Optional[str]   = Field("OK", example="OK")

class PumpDataOut(PumpDataIn):
    id: int
    created_at: str

    class Config:
        from_attributes = True

# ------ Endpoints ------
@app.post("/data/upload", dependencies=[Depends(require_key)], response_model=PumpDataOut)
async def upload(point: PumpDataIn, db: AsyncSession = Depends(get_db)):
    row = PumpData(
        pump_id=point.pump_id,
        temperature=point.temperature,
        pressure=point.pressure,
        vibration=point.vibration,
        status=point.status or "OK",
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row

@app.get("/data/latest", dependencies=[Depends(require_key)], response_model=List[PumpDataOut])
async def latest(limit: int = 50, db: AsyncSession = Depends(get_db)):
    q = select(PumpData).order_by(desc(PumpData.id)).limit(limit)
    res = await db.execute(q)
    return list(res.scalars().all())


@app.get(
    "/scoring/latest",
    dependencies=[Depends(require_key)],
)
async def scoring_latest(
    model: str = Query(
        "reports/models/fault_model_gbm.joblib",
        description="Pfad zum trainierten Modell (Joblib)",
    ),
    metrics: str = Query(
        "reports/fault_model_gbm_metrics.json",
        description="Pfad zur Metrik-JSON (enthält Threshold)",
    ),
    top_n: int = Query(20, ge=1, le=500, description="Anzahl Top-Ergebnisse"),
    latest_only: bool = Query(True, description="Nur letzte Messung je Sensor"),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optionaler Override für den Entscheidungsschwellenwert",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Berechnet Störungswahrscheinlichkeiten und liefert die Top-N Ergebnisse."""

    try:
        scores = await score_fault_risk(
            model_path=model,
            metrics_path=metrics,
            latest_only=latest_only,
            top_n=top_n,
            threshold_override=threshold,
            session=db,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Nur Teilmenge an Feldern für API-Ausgabe auswählen
    subset_cols = {
        "sensor_no",
        "date",
        "fault_probability",
        "fault_prediction",
        "fault_event",
        "fault_active",
        "days_since_last_fault",
        "recovery_rate_mean",
        "temperature_c",
        "vapour_flow_rate",
        "source_file",
        "created_at",
    }

    if scores.empty:
        return []

    available_cols = [c for c in scores.columns if c in subset_cols]
    payload = scores[available_cols].copy()
    payload = payload.where(payload.notna(), None)

    results: list[dict[str, Any]] = []
    for _, row in payload.iterrows():
        data = row.to_dict()
        sensor_no = data.get("sensor_no")
        if sensor_no is not None:
            data["sensor_no"] = str(sensor_no)
        validated = ScoreResult.model_validate(data).model_dump(mode="json")
        results.append(sanitize_floats(validated))

    return results


@app.post(
    "/scoring/run",
    dependencies=[Depends(require_key)],
)
async def scoring_run(
    model: str = Query(
        "reports/models/fault_model_gbm.joblib",
        description="Pfad zum trainierten Modell (Joblib)",
    ),
    metrics: str = Query(
        "reports/fault_model_gbm_metrics.json",
        description="Pfad zur Metrik-JSON (enthält Threshold)",
    ),
    top_n: int = Query(
        default=settings.scoring_default_top_n,
        ge=1,
        le=500,
        description="Anzahl Top-Ergebnisse",
    ),
    latest_only: bool = Query(True, description="Nur letzte Messung je Sensor"),
    persist: bool = Query(True, description="Speichert die Ergebnisse in der DB"),
    threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optionaler Override für den Entscheidungsschwellenwert",
    ),
    critical_threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Optionaler Override für kritische Severity",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Führt das Scoring aus und speichert optional die Ergebnisse."""

    scores = await score_fault_risk(
        model_path=model,
        metrics_path=metrics,
        latest_only=latest_only,
        top_n=top_n,
        threshold_override=threshold,
        session=db,
    )

    response: dict[str, Any] = {
        "count": int(len(scores)),
        "results": dataframe_to_dicts(scores),
        "persisted": False,
    }

    if persist and not scores.empty:
        alerts, run_log = await persist_scores(
            db,
            scores,
            model_path=model,
            metrics_path=metrics,
            warning_threshold=threshold if threshold is not None else settings.alert_threshold,
            critical_threshold=critical_threshold if critical_threshold is not None else settings.alert_critical_threshold,
            log_run=True,
        )
        response["persisted"] = True
        response["saved_count"] = len(alerts)
        response["alerts"] = [alert_to_dict(alert) for alert in alerts]
        if run_log is not None:
            response["run_log"] = sanitize_floats(
                {
                    "id": run_log.id,
                    "started_at": run_log.started_at,
                    "finished_at": run_log.finished_at,
                    "alerts_saved": run_log.alerts_saved,
                    "threshold": run_log.threshold,
                    "critical_threshold": run_log.critical_threshold,
                }
            )
        await notify_critical(response["alerts"])

    return response


@app.get(
    "/scoring/alerts",
    dependencies=[Depends(require_key)],
)
async def scoring_alerts(
    limit: int = Query(
        default=settings.dashboard_limit,
        ge=1,
        le=1000,
        description="Anzahl gespeicherter Alerts",
    ),
    offset: int = Query(0, ge=0, description="Offset für Pagination"),
    sensor_no: Optional[str] = Query(None, description="Optionaler Sensorfilter"),
    min_probability: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Nur Alerts oberhalb dieser Wahrscheinlichkeit"
    ),
    severity: Optional[str] = Query(
        None,
        description="Optionaler Severity-Filter (critical, warning, info)",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Liefert gespeicherte Alert-Datensätze aus der Datenbank."""

    filters = []
    if sensor_no:
        filters.append(VaporixAlert.sensor_no == sensor_no)
    if min_probability is not None:
        filters.append(VaporixAlert.fault_probability >= min_probability)
    if severity:
        if severity.lower() == "info":
            filters.append(or_(VaporixAlert.severity == "info", VaporixAlert.severity.is_(None)))
        else:
            filters.append(VaporixAlert.severity == severity.lower())

    total_stmt = select(func.count()).select_from(VaporixAlert)
    if filters:
        total_stmt = total_stmt.where(and_(*filters))
    total_res = await db.execute(total_stmt)
    total_count = total_res.scalar_one()

    stmt = select(VaporixAlert)
    if filters:
        stmt = stmt.where(and_(*filters))
    stmt = stmt.order_by(desc(VaporixAlert.scored_at)).offset(offset).limit(limit)

    res = await db.execute(stmt)
    alerts = [alert_to_dict(alert) for alert in res.scalars().all()]
    page_summary = summarize_alerts(alerts)
    page_summary["page_total"] = page_summary.get("total", len(alerts))
    page_summary["total"] = total_count
    return {
        "items": alerts,
        "summary": page_summary,
        "offset": offset,
        "limit": limit,
        "total": total_count,
        "has_next": offset + limit < total_count,
        "next_offset": offset + limit if offset + limit < total_count else None,
    }


@app.get(
    "/scoring/alerts/latest-per-sensor",
    dependencies=[Depends(require_key)],
)
async def scoring_alerts_latest_per_sensor(
    min_probability: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Nur Alerts oberhalb dieser Wahrscheinlichkeit"
    ),
    db: AsyncSession = Depends(get_db),
):
    """Liefert den jeweils letzten gespeicherten Alert je Sensor."""

    res = await db.execute(select(VaporixAlert).order_by(VaporixAlert.sensor_no, desc(VaporixAlert.scored_at)))
    alerts = res.scalars().all()

    latest_by_sensor: dict[str, VaporixAlert] = {}
    for alert in alerts:
        key = alert.sensor_no or "_unknown"
        if min_probability is not None and alert.fault_probability < min_probability:
            continue
        current = latest_by_sensor.get(key)
        if current is None or (alert.scored_at and alert.scored_at > current.scored_at):
            latest_by_sensor[key] = alert

    alerts = [alert_to_dict(alert) for alert in latest_by_sensor.values()]
    return {
        "items": alerts,
        "summary": summarize_alerts(alerts),
    }


@app.get(
    "/scoring/alerts/sensor/{sensor_no}",
    dependencies=[Depends(require_key)],
)
async def scoring_alerts_by_sensor(
    sensor_no: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(VaporixAlert)
        .where(VaporixAlert.sensor_no == sensor_no)
        .order_by(desc(VaporixAlert.scored_at))
        .offset(offset)
        .limit(limit)
    )
    res = await db.execute(stmt)
    alerts = [alert_to_dict(alert) for alert in res.scalars().all()]

    trend_stmt = (
        select(VaporixAlert.date, func.avg(VaporixAlert.fault_probability))
        .where(VaporixAlert.sensor_no == sensor_no, VaporixAlert.date != None)
        .group_by(VaporixAlert.date)
        .order_by(VaporixAlert.date)
    )
    trend_res = await db.execute(trend_stmt)
    trend = [
        {
            "date": row[0],
            "avg_probability": row[1],
        }
        for row in trend_res.all()
    ]

    return {
        "items": alerts,
        "summary": summarize_alerts(alerts),
        "trend": sanitize_floats(trend),
    }


@app.get(
    "/scoring/alerts/metrics",
    dependencies=[Depends(require_key)],
)
async def scoring_alerts_metrics(
    days: int = Query(7, ge=1, le=365, description="Zeitraum für Trend/Recent-Stats in Tagen"),
    db: AsyncSession = Depends(get_db),
):
    base_stmt = select(VaporixAlert.severity, func.count()).group_by(VaporixAlert.severity)
    res_total = await db.execute(base_stmt)
    total_by_severity = {row[0] or "info": row[1] for row in res_total.all()}

    since = datetime.utcnow() - timedelta(days=days)
    recent_stmt = (
        select(VaporixAlert.severity, func.count())
        .where(VaporixAlert.scored_at >= since)
        .group_by(VaporixAlert.severity)
    )
    res_recent = await db.execute(recent_stmt)
    recent_by_severity = {row[0] or "info": row[1] for row in res_recent.all()}

    top_sensors_stmt = (
        select(
            VaporixAlert.sensor_no,
            func.count().label("count"),
            func.max(VaporixAlert.fault_probability).label("max_probability"),
        )
        .group_by(VaporixAlert.sensor_no)
        .order_by(desc("count"))
        .limit(10)
    )
    top_sensors_res = await db.execute(top_sensors_stmt)
    top_sensors = [
        {
            "sensor_no": row[0] or "",
            "count": row[1],
            "max_probability": row[2],
        }
        for row in top_sensors_res.all()
    ]

    trend_stmt = (
        select(
            VaporixAlert.date,
            func.avg(VaporixAlert.fault_probability).label("avg_probability"),
        )
        .where(VaporixAlert.date != None)
        .group_by(VaporixAlert.date)
        .order_by(VaporixAlert.date)
        .limit(60)
    )
    trend_res = await db.execute(trend_stmt)
    trend = [
        {
            "date": row[0],
            "avg_probability": row[1],
        }
        for row in trend_res.all()
    ]

    return sanitize_floats(
        {
            "total_by_severity": total_by_severity,
            "recent_by_severity": recent_by_severity,
            "top_sensors": top_sensors,
            "trend": trend,
        }
    )


@app.get(
    "/dashboard/alerts",
    dependencies=[Depends(require_key)],
    response_class=HTMLResponse,
)
async def dashboard_alerts(
    request: Request,
    limit: int = Query(None, description="Anzahl Einträge", ge=1, le=500),
    sensor_no: Optional[str] = Query(None),
    min_probability: Optional[float] = Query(None, ge=0.0, le=1.0),
    auto_refresh: int = Query(0, ge=0, le=1, description="Auto Refresh"),
    db: AsyncSession = Depends(get_db),
):
    limit_value = limit or settings.dashboard_limit
    alerts_payload = await scoring_alerts(
        limit=limit_value,
        offset=0,
        sensor_no=sensor_no,
        min_probability=min_probability,
        db=db,
    )
    alerts = alerts_payload["items"]
    summary = alerts_payload["summary"]
    # Statt serverseitigem Template verweisen wir auf das React-Frontend
    html = f"""
    <!DOCTYPE html>
    <html lang='de'>
      <head>
        <meta charset='utf-8'/>
        <title>SmartFuel Dashboard</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #050507; color: #f5f5f7; display:flex; align-items:center; justify-content:center; height: 100vh; margin:0; }}
          .card {{ text-align:center; background:#0f0f15; padding:2rem 3rem; border-radius:1rem; border:1px solid rgba(255,255,255,0.08); box-shadow:0 20px 40px rgba(0,0,0,0.35); }}
          a {{ color:#61dafb; text-decoration:none; font-weight:600; }}
        </style>
      </head>
      <body>
        <div class='card'>
          <h1>SmartFuel React Dashboard</h1>
          <p>Bitte nutze das React-Frontend unter<br/><strong><a href='http://localhost:5173'>http://localhost:5173</a></strong></p>
          <p>API-Key erforderlich: <code>x-api-key: dev-1234</code></p>
          <p>Backend liefert Rohdaten über <code>/scoring/alerts</code> &amp; <code>/scoring/alerts/metrics</code>.</p>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get(
    "/scoring/run-logs",
    dependencies=[Depends(require_key)],
)
async def scoring_run_logs(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(ScoringRun).order_by(desc(ScoringRun.started_at)).limit(limit)
    )
    runs = []
    for run in res.scalars().all():
        runs.append(
            sanitize_floats(
                {
                    "id": run.id,
                    "started_at": run.started_at,
                    "finished_at": run.finished_at,
                    "alerts_saved": run.alerts_saved,
                    "model_path": run.model_path,
                    "metrics_path": run.metrics_path,
                    "threshold": run.threshold,
                    "critical_threshold": run.critical_threshold,
                    "note": run.note,
                }
            )
        )
    return {"items": runs}
