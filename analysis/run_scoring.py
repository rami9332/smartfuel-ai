#!/usr/bin/env python
"""
Batch scoring helper for Vaporix fault prediction.

Example:
    PYTHONPATH=. MPLCONFIGDIR=.matplotlib-cache \\
        .venv/bin/python analysis/run_scoring.py \\
        --model reports/models/fault_model_gbm.joblib \\
        --metrics reports/fault_model_gbm_metrics.json \\
        --top-n 20 --output reports/latest_scores.csv
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

import pandas as pd

from app.db.session import AsyncSessionLocal, engine
from app.models.models import Base
from app.services.scoring import score_fault_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Vaporix measurements with the trained model")
    parser.add_argument("--model", type=Path, required=True, help="Pfad zum joblib-Modell")
    parser.add_argument("--metrics", type=Path, required=True, help="Pfad zur Metrik-JSON")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Anzahl Top-Ergebnisse (nach Wahrscheinlichkeit) in der Ausgabe",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        default=True,
        help="Nur die jeweils letzte Messung je Sensor bewerten (default: True)",
    )
    parser.add_argument(
        "--no-latest-only",
        dest="latest_only",
        action="store_false",
        help="Setzen, um alle Messungen zu bewerten.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionaler Pfad für CSV-Export der Scores",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> pd.DataFrame:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        scores = await score_fault_risk(
            model_path=args.model,
            metrics_path=args.metrics,
            latest_only=args.latest_only,
            top_n=args.top_n,
            session=session,
        )
    return scores


def main() -> None:
    args = parse_args()
    scores = asyncio.run(_run(args))

    if scores.empty:
        print("Keine Daten gefunden.")
        return

    pd.set_option("display.max_columns", None)
    print("=== Top Ergebnisse ===")
    print(scores.head(args.top_n).to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.output, index=False)
        print(f"Scores gespeichert unter {args.output}")


if __name__ == "__main__":
    main()
