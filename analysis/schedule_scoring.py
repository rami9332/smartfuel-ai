#!/usr/bin/env python
"""Einfacher Scheduler für wiederkehrende Scoring-Läufe."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.db.session import AsyncSessionLocal, engine
from app.models.models import Base
from app.services.pipeline import run_pipeline


async def run_once(model: Path | str, metrics: Path | str, top_n: int, latest_only: bool, data_dir: Path | str) -> dict:
    async with AsyncSessionLocal() as session:
        result = await run_pipeline(
            session,
            data_dir=data_dir,
            top_n=top_n,
            latest_only=latest_only,
            model_path=model,
            metrics_path=metrics,
        )
    return result


async def scheduler_loop(args: argparse.Namespace) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    interval = max(0, args.interval_minutes) * 60
    while True:
        start = datetime.utcnow()
        result = await run_once(
            args.model,
            args.metrics,
            args.top_n,
            not args.no_latest_only,
            args.data_dir,
        )
        print(
            f"[{start.isoformat()}] Pipeline abgeschlossen – "
            f"ingested={result['ingested']} scored={result['scored']} alerts={result['alerts_saved']}"
        )
        if interval <= 0:
            break
        await asyncio.sleep(interval)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Periodisches Scoring starten")
    parser.add_argument("--model", default="reports/models/fault_model_gbm.joblib")
    parser.add_argument("--metrics", default="reports/fault_model_gbm_metrics.json")
    parser.add_argument("--top-n", type=int, default=settings.scoring_default_top_n)
    parser.add_argument("--interval-minutes", type=int, default=settings.scoring_interval_minutes)
    parser.add_argument("--data-dir", default=settings.data_directory)
    parser.add_argument("--no-latest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(scheduler_loop(args))


if __name__ == "__main__":
    main()
