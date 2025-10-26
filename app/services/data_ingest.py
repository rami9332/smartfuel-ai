from __future__ import annotations

import csv
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import VaporixMeasurement

# Mapping der Meta-Feldnamen (englisch & deutsch) auf unsere Spaltennamen.
META_KEY_MAP: Mapping[str, str] = {
    "Serial no.": "serial_no",
    "Gerätenr.": "serial_no",
    "Hardware": "hardware",
    "Software": "software",
    "Protocol": "protocol",
    "Protokoll": "protocol",
    "Pulse rate": "pulse_rate",
    "Pulsrate": "pulse_rate",
    "Country code": "country_code",
    "Ländercode": "country_code",
    "PCM": "pcm",
    "FLG": "flg",
    "Side": "side",
    "Seite": "side",
    "Status": "status_text",
}

# Mapping der Messwert-Header auf unsere Spalten.
COLUMN_KEY_MAP: Mapping[str, str] = {
    "Sensor no.": "sensor_no",
    "Sensornr.": "sensor_no",
    "Date": "date",
    "Datum": "date",
    "Error counter": "error_counter",
    "Fehlerzähler": "error_counter",
    "Fuel transfers": "fuel_transfers",
    "Betankungszähler": "fuel_transfers",
    "Recovery rate in %": "recovery_rate",
    "Rückführrate in %": "recovery_rate",
    "Mean value of recovery rate in %": "recovery_rate_mean",
    "Mittelwert der Rückführrate in %": "recovery_rate_mean",
    "Temperature in °C": "temperature_c",
    "Temperature in ¡C": "temperature_c",
    "Gastemperatur in °C": "temperature_c",
    "Gastemperatur in ¡C": "temperature_c",
    "VC status": "vc_status",
    "GK-Status": "vc_status",
    "Vapour flow rate in l/min": "vapour_flow_rate",
    "Gasfluß in l/min": "vapour_flow_rate",
    "Gasfluss in l/min": "vapour_flow_rate",
    "Fuel flow rate in l/min": "fuel_flow_rate",
    "Kraftstofffluß in l/min": "fuel_flow_rate",
    "Kraftstofffluss in l/min": "fuel_flow_rate",
    "Recovery rate (+)": "recovery_rate_pos",
    "Rückführrate (+)": "recovery_rate_pos",
    "Recovery rate (-)": "recovery_rate_neg",
    "Rückführrate (-)": "recovery_rate_neg",
}


def _read_rows(path: Path) -> List[List[str]]:
    """Read the CSV rows using a best-effort encoding detection."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.reader(fh, delimiter=";")
                return [[cell.strip() for cell in row] for row in reader]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, "Unable to decode file")


def _parse_meta_row(row: Iterable[str], meta: Dict[str, Any]) -> None:
    """Update meta dict in-place by reading alternating key/value cells."""
    cells = list(row)
    for idx in range(0, len(cells), 2):
        key = cells[idx].strip()
        if not key:
            continue
        value = cells[idx + 1].strip() if idx + 1 < len(cells) else ""
        mapped = META_KEY_MAP.get(key)
        if not mapped:
            continue
        meta[mapped] = value


def _to_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    value = value.replace(",", ".")
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    value = value.replace("%", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def _parse_date(value: str) -> Optional[date]:
    value = value.strip()
    if not value:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


COLUMN_CONVERTERS: Mapping[str, Any] = {
    "sensor_no": lambda v: v.strip() or None,
    "date": _parse_date,
    "error_counter": _to_int,
    "fuel_transfers": _to_int,
    "recovery_rate": _to_float,
    "recovery_rate_mean": _to_float,
    "temperature_c": _to_float,
    "vc_status": _to_int,
    "vapour_flow_rate": _to_float,
    "fuel_flow_rate": _to_float,
    "recovery_rate_pos": _to_float,
    "recovery_rate_neg": _to_float,
}

META_CONVERTERS: Mapping[str, Any] = {
    "serial_no": lambda v: v.strip() or None,
    "hardware": lambda v: v.strip() or None,
    "software": lambda v: v.strip() or None,
    "protocol": lambda v: v.strip() or None,
    "pulse_rate": _to_int,
    "country_code": lambda v: v.strip() or None,
    "pcm": _to_int,
    "flg": _to_int,
    "side": lambda v: v.strip() or None,
    "status_text": lambda v: v.strip() or None,
}


def parse_vaporix_csv(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse one Vaporix CSV export and return meta information + rows."""
    rows = _read_rows(path)
    meta: Dict[str, Any] = {"source_file": path.name}
    header_cells: Optional[List[str]] = None
    data: List[Dict[str, Any]] = []

    for row in rows:
        if not any(cell for cell in row):
            continue
        first = row[0]
        if header_cells is None:
            if first in META_KEY_MAP:
                _parse_meta_row(row, meta)
                continue
            if first.lower().startswith(("history", "historie")):
                meta["title"] = first
                continue
            if first in ("Seite", "Side"):
                _parse_meta_row(row, meta)
                continue
            # Header row?
            mapped = [COLUMN_KEY_MAP.get(cell, "") for cell in row]
            if any(mapped):
                header_cells = row
            continue

        # After we have a header we can map the values.
        row_data: Dict[str, Any] = {}
        for cell, header in zip(row, header_cells):
            key = COLUMN_KEY_MAP.get(header)
            if not key:
                continue
            converter = COLUMN_CONVERTERS.get(key, lambda v: v.strip() or None)
            row_data[key] = converter(cell)

        if any(row_data.values()):
            data.append(row_data)

    # Normalize meta values with converters
    for key, converter in META_CONVERTERS.items():
        value = meta.get(key)
        if value is None:
            continue
        meta[key] = converter(value)  # type: ignore[arg-type]

    return meta, data


async def ingest_vaporix_file(db: AsyncSession, file_path: Path | str) -> int:
    """
    Parse a single CSV file and store the measurements.

    Returns the number of imported rows. Existing rows for the same source_file
    are removed before import to allow idempotent re-runs.
    """
    path = Path(file_path)
    meta, rows = parse_vaporix_csv(path)
    source_file = meta["source_file"]

    await db.execute(
        delete(VaporixMeasurement).where(VaporixMeasurement.source_file == source_file)
    )

    count = 0
    for payload in rows:
        record = VaporixMeasurement(
            source_file=source_file,
            serial_no=meta.get("serial_no"),
            hardware=meta.get("hardware"),
            software=meta.get("software"),
            protocol=meta.get("protocol"),
            pulse_rate=meta.get("pulse_rate"),
            country_code=meta.get("country_code"),
            pcm=meta.get("pcm"),
            flg=meta.get("flg"),
            side=meta.get("side"),
            status_text=meta.get("status_text"),
            **payload,
        )
        db.add(record)
        count += 1

    await db.commit()
    return count


async def ingest_vaporix_directory(
    db: AsyncSession, directory: Path | str, pattern: str = "*.csv"
) -> int:
    """
    Import all CSV files in a directory.

    Returns the total number of imported rows.
    """
    total = 0
    directory_path = Path(directory)
    for csv_path in sorted(directory_path.glob(pattern)):
        if csv_path.name.startswith("._"):
            # macOS erstellt für einige Tools Resource-Fork-Dateien – überspringen.
            continue
        total += await ingest_vaporix_file(db, csv_path)
    return total
