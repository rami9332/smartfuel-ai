#!/usr/bin/env python
"""Manueller Pipeline-Run: Ingestion + Scoring + Alerts."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.session import AsyncSessionLocal, engine
from app.models.models import Base
from app.services.pipeline import run_pipeline


async def run(args: argparse.Namespace) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        result = await run_pipeline(
            session,
            data_dir=args.data_dir,
            top_n=args.top_n,
            latest_only=args.latest_only,
            threshold_override=args.threshold,
            critical_threshold=args.critical_threshold,
            model_path=args.model,
            metrics_path=args.metrics,
        )
        print(
            "Pipeline Result:",
            f"ingested={result['ingested']}",
            f"scored={result['scored']}",
            f"alerts_saved={result['alerts_saved']}",
            f"run_log_id={result['run_log_id']}",
        )


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run ingestion + scoring pipeline")
    parser.add_argument("--data-dir", default=settings.data_directory)
    parser.add_argument("--model", default="reports/models/fault_model_gbm.joblib")
    parser.add_argument("--metrics", default="reports/fault_model_gbm_metrics.json")
    parser.add_argument("--top-n", type=int, default=settings.scoring_default_top_n)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--critical-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
