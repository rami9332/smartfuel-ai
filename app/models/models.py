# app/models/models.py
from datetime import datetime
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Boolean

Base = declarative_base()

class PumpData(Base):
    __tablename__ = "pump_data"

    id          = Column(Integer, primary_key=True, index=True)
    pump_id     = Column(String, index=True, nullable=False)   # z.B. „HH-Station-03“
    temperature = Column(Float, nullable=True)                  # °C
    pressure    = Column(Float, nullable=True)                  # bar
    vibration   = Column(Float, nullable=True)                  # mm/s
    status      = Column(String, default="OK")                  # optionales Feld vom Sensor
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)


class VaporixMeasurement(Base):
    """
    History/Telemetry-Export der VAPORIX-Steuerung.
    Enthält neben Messreihen auch die Metaangaben je Exportdatei.
    """

    __tablename__ = "vaporix_measurements"

    id                 = Column(Integer, primary_key=True)
    source_file        = Column(String, nullable=False)              # Datei/Quelle des Imports
    serial_no          = Column(String, index=True, nullable=True)
    hardware           = Column(String, nullable=True)
    software           = Column(String, nullable=True)
    protocol           = Column(String, nullable=True)
    pulse_rate         = Column(Integer, nullable=True)
    country_code       = Column(String, nullable=True)
    pcm                = Column(Integer, nullable=True)
    flg                = Column(Integer, nullable=True)
    side               = Column(String, nullable=True)
    status_text        = Column(String, nullable=True)

    sensor_no          = Column(String, index=True, nullable=True)
    date               = Column(Date, index=True, nullable=True)
    error_counter      = Column(Integer, nullable=True)
    fuel_transfers     = Column(Integer, nullable=True)
    recovery_rate      = Column(Float, nullable=True)
    recovery_rate_mean = Column(Float, nullable=True)
    temperature_c      = Column(Float, nullable=True)
    vc_status          = Column(Integer, nullable=True)
    vapour_flow_rate   = Column(Float, nullable=True)
    fuel_flow_rate     = Column(Float, nullable=True)
    recovery_rate_pos  = Column(Float, nullable=True)
    recovery_rate_neg  = Column(Float, nullable=True)

    created_at         = Column(DateTime, default=datetime.utcnow, index=True)


class VaporixAlert(Base):
    """
    Persistierte Ergebnisse aus dem Scoring-Service.
    """

    __tablename__ = "vaporix_alerts"

    id                = Column(Integer, primary_key=True)
    sensor_no         = Column(String, index=True, nullable=True)
    date              = Column(Date, index=True, nullable=True)
    fault_probability = Column(Float, nullable=False)
    fault_prediction  = Column(Boolean, nullable=False, default=False)
    fault_event       = Column(Boolean, nullable=True)
    fault_active      = Column(Boolean, nullable=True)
    days_since_last_fault = Column(Float, nullable=True)
    recovery_rate_mean    = Column(Float, nullable=True)
    temperature_c         = Column(Float, nullable=True)
    vapour_flow_rate      = Column(Float, nullable=True)
    source_file           = Column(String, nullable=True)
    model_path            = Column(String, nullable=True)
    metrics_path          = Column(String, nullable=True)
    severity              = Column(String, nullable=True)
    scored_at             = Column(DateTime, default=datetime.utcnow, index=True)


class ScoringRun(Base):
    """
    Protokolliert ausgeführte Scoring-Läufe (z. B. Scheduler).
    """

    __tablename__ = "scoring_runs"

    id             = Column(Integer, primary_key=True)
    started_at     = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at    = Column(DateTime, nullable=True)
    alerts_saved   = Column(Integer, default=0)
    model_path     = Column(String, nullable=True)
    metrics_path   = Column(String, nullable=True)
    threshold      = Column(Float, nullable=True)
    critical_threshold = Column(Float, nullable=True)
    note           = Column(String, nullable=True)
