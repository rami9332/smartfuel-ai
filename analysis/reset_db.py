#!/usr/bin/env python
"""Hilfsskript: setzt die SQLite-Datenbank zurück."""

from __future__ import annotations

import asyncio

from app.db.session import engine


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(lambda connection: connection.exec_driver_sql("DROP TABLE IF EXISTS vaporix_alerts"))
        await conn.run_sync(lambda connection: connection.exec_driver_sql("DROP TABLE IF EXISTS vaporix_measurements"))
        await conn.run_sync(lambda connection: connection.exec_driver_sql("DROP TABLE IF EXISTS scoring_runs"))
        await conn.run_sync(lambda connection: connection.exec_driver_sql("DROP TABLE IF EXISTS pump_data"))
    print("SQLite-DB zurückgesetzt.")


if __name__ == "__main__":
    asyncio.run(main())
