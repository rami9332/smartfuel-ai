#!/usr/bin/env python
"""
Schnelles Explorationsskript für VAPORIX-Messdaten.

Beispiel:
    poetry run python analysis/vaporix_exploration.py --plots-dir reports/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine

from app.db.session import DB_URL
from app.services.logic import (
    VaporixLabelConfig,
    compute_vaporix_features,
    label_vaporix_anomalies,
    summarise_vaporix_health,
)


def _sync_db_url(url: str) -> str:
    """Wandelt async-SQLAlchemy URLs in synchrone Gegenstücke um."""
    if url.startswith("sqlite+aiosqlite"):
        return url.replace("+aiosqlite", "")
    return url


def load_dataframe(db_url: str, limit: Optional[int] = None) -> pd.DataFrame:
    engine = create_engine(_sync_db_url(db_url))
    query = "SELECT * FROM vaporix_measurements ORDER BY date, sensor_no"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql(query, engine, parse_dates=["date", "created_at"])
    return df


def summarise(df: pd.DataFrame) -> dict:
    sensors = df["sensor_no"].dropna().unique()
    return {
        "rows": len(df),
        "sensors": len(sensors),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "serial_numbers": sorted(df["serial_no"].dropna().unique()),
    }


def print_summary(info: dict) -> None:
    print("=== Vaporix Dataset Summary ===")
    for key, value in info.items():
        print(f"{key:>15}: {value}")


def plot_series(df: pd.DataFrame, sensors: Iterable[str], output_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # Lazy import, damit Skript ohne Matplotlib lauffähig bleibt
    except ModuleNotFoundError:
        print("Matplotlib nicht installiert – überspringe Plot-Erzeugung.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    plotted = False

    for sensor in sensors:
        subset = df[df["sensor_no"] == sensor].sort_values("date")
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["date"], subset["recovery_rate_mean"], label="Recovery rate (mean)", color="tab:blue")
        ax.set_ylabel("Recovery rate Mean (%)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax.twinx()
        ax2.plot(subset["date"], subset["temperature_c"], label="Temperature °C", color="tab:red")
        ax2.set_ylabel("Temperature (°C)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax.set_title(f"Sensor {sensor}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{sensor}_timeseries.png", dpi=150)
        plt.close(fig)
        plotted = True

    return plotted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vaporix dataset exploration")
    parser.add_argument("--db-url", default=DB_URL, help="SQLAlchemy DB URL (default aus Umgebungsvariable)")
    parser.add_argument("--limit", type=int, help="Optional: nur die ersten N Zeilen laden")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Wenn gesetzt, speichert Zeitreihenplots je Sensor in diesem Ordner",
    )
    parser.add_argument(
        "--top-sensors",
        type=int,
        default=3,
        help="Anzahl Sensoren mit den meisten Einträgen für Plots (Standard: 3)",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        help="Pfad für exportierte Feature-Tabelle (CSV)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optionaler Export für Sensor-Health-Summary",
    )
    parser.add_argument(
        "--min-recovery-mean",
        type=float,
        default=VaporixLabelConfig().min_recovery_mean,
        help="Schwellwert für minimal akzeptable Recovery-Rate in % (Default 95).",
    )
    parser.add_argument(
        "--recovery-drop-threshold",
        type=float,
        default=VaporixLabelConfig().recovery_drop_threshold,
        help="Grenze für negativen Sprung der Recovery-Rate (Default -2).",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=VaporixLabelConfig().max_temperature_c,
        help="Maximale Gastemperatur vor Warnung (Default 35 °C).",
    )
    parser.add_argument(
        "--min-vapour-flow",
        type=float,
        default=VaporixLabelConfig().min_vapour_flow,
        help="Untergrenze für Gasfluss l/min (Default 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.db_url, limit=args.limit)
    summary = summarise(df)
    print_summary(summary)

    cfg = VaporixLabelConfig(
        min_recovery_mean=args.min_recovery_mean,
        recovery_drop_threshold=args.recovery_drop_threshold,
        max_temperature_c=args.max_temperature,
        min_vapour_flow=args.min_vapour_flow,
    )

    feats = compute_vaporix_features(df)
    labeled = label_vaporix_anomalies(feats, cfg)
    health = summarise_vaporix_health(labeled)

    print("\n=== Alert Overview ===")
    if not health.empty:
        display_cols = [
            "sensor_no",
            "fault_events",
            "fault_active_rows",
            "alerts",
            "alert_ratio",
            "last_alert",
        ]
        print(
            health[display_cols]
            .sort_values(["fault_events", "alerts"], ascending=False)
            .to_string(index=False)
        )
    else:
        print("Keine Daten vorhanden.")

    if args.features_out:
        args.features_out.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_csv(args.features_out, index=False)
        print(f"Features exportiert nach {args.features_out}")

    if args.summary_out and not health.empty:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        health.to_csv(args.summary_out, index=False)
        print(f"Sensor-Summary exportiert nach {args.summary_out}")

    if args.plots_dir:
        sensor_counts = df["sensor_no"].value_counts().nlargest(args.top_sensors).index.tolist()
        if plot_series(labeled, sensor_counts, args.plots_dir):
            print(f"Plots gespeichert unter {args.plots_dir}")


if __name__ == "__main__":
    main()
