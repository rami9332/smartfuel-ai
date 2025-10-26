# app/services/data_ingest.py
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import PumpData

async def ingest_csv(file_path: str, db: AsyncSession):
    """
    Liest CSV-Daten (z. B. von Zapfsäulen) ein und speichert sie in der DB.
    """
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        entry = PumpData(
            pump_id=row.get("pump_id", "Unknown"),
            temperature=row.get("temperature"),
            pressure=row.get("pressure"),
            vibration=row.get("vibration"),
            status=row.get("status", "OK")
        )
        db.add(entry)

    await db.commit()
    print(f"{len(df)} Datensätze erfolgreich importiert.")