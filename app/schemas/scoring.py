from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ScoreResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True, arbitrary_types_allowed=True)

    sensor_no: Optional[str] = Field(None, description="Externe Sensor-ID")
    date: Optional[date] = Field(None, description="Messdatum")
    fault_probability: float = Field(..., ge=0.0, le=1.0, description="Vorhersagewahrscheinlichkeit")
    fault_prediction: bool = Field(..., description="True wenn Wahrscheinlichkeit >= Schwelle")
    fault_event: Optional[bool] = Field(None, description="Historischer Fault-Event (falls bekannt)")
    fault_active: Optional[bool] = Field(None, description="Historisch aktiver Fehlerzähler")
    days_since_last_fault: Optional[float] = Field(None)
    rec_rate_mean: Optional[float] = Field(None, alias="recovery_rate_mean")
    temperature_c: Optional[float] = Field(None)
    vapour_flow_rate: Optional[float] = Field(None)
    source_file: Optional[str] = Field(None)
    created_at: Optional[datetime] = Field(None)
