#!/usr/bin/env python
"""Exportiert Alerts als CSV oder JSON."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from app.db.session import AsyncSessionLocal
from app.models.models import VaporixAlert, ScoringRun


async def fetch_alerts(limit: int) -> list[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            VaporixAlert.__table__.select().order_by(VaporixAlert.scored_at.desc()).limit(limit)
        )
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]


async def fetch_runs(limit: int) -> list[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            ScoringRun.__table__.select().order_by(ScoringRun.started_at.desc()).limit(limit)
        )
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]


def write_output(data: list[dict], path: Path, fmt: str) -> None:
    if fmt == "json":
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    else:
        if not data:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


async def main():
    parser = argparse.ArgumentParser(description="Export alerts and runs")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--output", default="reports/latest_alerts.csv")
    parser.add_argument("--runs-output", default=None)
    args = parser.parse_args()

    alerts = await fetch_alerts(args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(alerts, output_path, args.format)
    print(f"Alerts exportiert: {len(alerts)} -> {output_path}")

    if args.runs_output:
        runs = await fetch_runs(args.limit)
        runs_path = Path(args.runs_output)
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        write_output(runs, runs_path, args.format)
        print(f"Runs exportiert: {len(runs)} -> {runs_path}")


if __name__ == "__main__":
    asyncio.run(main())
