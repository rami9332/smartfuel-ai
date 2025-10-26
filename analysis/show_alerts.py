#!/usr/bin/env python
"""Kleiner Snapshot der gespeicherten Alerts."""

from __future__ import annotations

import asyncio

from app.db.session import AsyncSessionLocal
from app.models.models import VaporixAlert


async def main() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            VaporixAlert.__table__.select().order_by(VaporixAlert.scored_at.desc()).limit(10)
        )
        rows = result.fetchall()
        if not rows:
            print("Keine Alerts gespeichert.")
            return
        print(f"Top {len(rows)} Alerts:\n")
        for r in rows:
            print(
                f"Sensor={r.sensor_no} prob={r.fault_probability:.3f} severity={r.severity} "
                f"date={r.date} source={r.source_file}"
            )


if __name__ == "__main__":
    asyncio.run(main())
