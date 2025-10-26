# SmartFuel AI Backend

## Vaporix CSV Import
- Export jedes VAPORIX-Steuerungsblatt als `CSV UTF-8` (Semikolon-getrennt) und lege die Dateien in `Data/<ordner>/`.
- Beispiel: `Data/Vaporix_side_a.csv/messung 1.csv`.
- Starte dann einmalig den Importer (setzt die Tabellen automatisch auf):

```bash
.venv/bin/python - <<'PY'
import asyncio
from pathlib import Path
from app.db.session import engine, AsyncSessionLocal
from app.models.models import Base
from app.services.data_ingest import ingest_vaporix_directory

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        count = await ingest_vaporix_directory(session, Path("Data/Vaporix_side_a.csv"))
    print("Imported rows:", count)

asyncio.run(main())
PY
```

- Die Daten landen in der Tabelle `vaporix_measurements`. Mehrfaches Ausführen überschreibt bestehende Einträge derselben Datei.

## Datenexploration
- Erstes Explorationsskript: `analysis/vaporix_exploration.py`
- Beispielaufruf (Plots + Feature-Export):

```bash
PYTHONPATH=. .venv/bin/python analysis/vaporix_exploration.py \
  --plots-dir reports/plots \
  --features-out reports/vaporix_features.csv \
  --summary-out reports/vaporix_summary.csv
```

- Konsolenausgabe enthält Überblick (Zeilen, Sensoren, Zeitspanne) plus Alert-Zusammenfassung.
- Export `reports/vaporix_features.csv` enthält Features inkl. Heuristik-Labels (`is_alert`, `severity`, `alert_reasons`) sowie echte Störungsindikatoren (`fault_event`, `fault_active`, `error_delta`). Der Summary-Export fasst Alerts und Fehlerevents je Sensor zusammen.
- Plots (sofern `matplotlib` installiert) zeigen `Recovery rate` und Temperatur pro Sensor.
- Schwellwerte für die Heuristik lassen sich via CLI anpassen, z. B. `--min-recovery-mean 93 --max-temperature 32`.

## Modelltraining
- Baseline-Training-Skript: `analysis/train_fault_model.py`
- Führt ein RandomForest-Modell auf den Features aus, bewertet es und kann Artefakte speichern:

```bash
PYTHONPATH=. .venv/bin/python analysis/train_fault_model.py \
  --metrics-out reports/fault_model_metrics.json \
  --model-out reports/models/fault_model.joblib
```

- `reports/fault_model_metrics.json` enthält ROC-AUC, Classification-Report, Konfusionsmatrix und die wichtigsten Features.
- Das gespeicherte Modell (`reports/models/fault_model.joblib`) lässt sich später in einer API oder einem Batch-Job verwenden.
- Standardmäßig werden Feature-Spalten mit hohem Leakage-Risiko (`fault_active`, `error_counter`, …) ausgeschlossen und ein Sensor-basierter Group-Split genutzt. Bei Bedarf lassen sich diese Einstellungen via `--exclude-features` bzw. `--test-size` anpassen.
- Für unausgeglichene Klassen kann mit `--use-smote` ein Oversampling (SMOTE) aktiviert werden; Voraussetzung ist `pip install imbalanced-learn`.
- Alternative Klassifikatoren können über `--model-type` gewählt werden (`random_forest`, `hist_gradient_boosting`, `logistic`). Je Lauf werden zusätzlich Precision-Recall-Kurve, bester F1-Schwellenwert und per-Sensor-Metriken mit exportiert.
- Beispiel für Gradient Boosting inkl. SMOTE:

```bash
PYTHONPATH=. .venv/bin/python analysis/train_fault_model.py \
  --model-type hist_gradient_boosting --use-smote \
  --metrics-out reports/fault_model_gbm_metrics.json \
  --model-out reports/models/fault_model_gbm.joblib
```

- Vergleich (aktuelle Daten, Stand Skriptausführung):
  * RandomForest + SMOTE: Accuracy ~0.95, Recall_fault ~0.43, ROC-AUC ~0.98
  * HistGradientBoosting + SMOTE: Accuracy ~0.98, Recall_fault ~0.79, ROC-AUC ~0.99
  * Logistic Regression + SMOTE: Accuracy ~0.85, Recall_fault ~0.64, ROC-AUC ~0.92
- Die Metrik-JSON (`precision_recall_curve.best_threshold`) liefert einen sinnvollen Schwellenwert für Alarme.

## Batch Scoring
- Skript `analysis/run_scoring.py` berechnet aktuelle Störungswahrscheinlichkeiten direkt aus der Datenbank.
- Beispiel (nutzt das Gradient-Boosting-Modell und exportiert die Top 20 in eine CSV):

```bash
PYTHONPATH=. .venv/bin/python analysis/run_scoring.py \
  --model reports/models/fault_model_gbm.joblib \
  --metrics reports/fault_model_gbm_metrics.json \
  --top-n 20 \
  --output reports/latest_scores.csv
```

- Standardmäßig wird pro Sensor nur die letzte Messung bewertet (`--latest-only`). Mit `--no-latest-only` lassen sich alle Messungen auswerten.
- Ausgabe enthält u. a. Wahrscheinlichkeit (`fault_probability`) und binären Alert (`fault_prediction`) basierend auf dem gespeicherten Schwellenwert.

### API-Endpunkt
- GET `/scoring/latest` (Header `x-api-key` erforderlich)
  * Query-Parameter: `model` (joblib-Pfad), `metrics` (JSON mit Threshold), `top_n` (Default 20), `latest_only` (Default true)
  * Antwort: Liste von `ScoreResult`-Objekten mit den relevantesten Feldern (`sensor_no`, `fault_probability`, `fault_prediction`, `days_since_last_fault`, etc.).
  * Beispielaufruf:

```bash
curl -H "x-api-key: dev-1234" \
  "http://localhost:8000/scoring/latest?model=reports/models/fault_model_gbm.joblib&metrics=reports/fault_model_gbm_metrics.json&top_n=10"
```

- Endpunkt verwendet das gespeicherte `best_threshold` aus der Metrik-JSON und antwortet nur mit den Top-N Einträgen sortiert nach Wahrscheinlichkeit.
- POST `/scoring/run` führt das Scoring aus und speichert (Standard) die Ergebnisse in der Tabelle `vaporix_alerts`. Rückgabe enthält die Zugriffs-Resultate sowie die persistierten Datensätze.
- GET `/scoring/alerts` liefert gespeicherte Alerts; optional mit `sensor_no=`, `min_probability=` und `severity=` filterbar. Antwort enthält `items` + `summary` (Severity-Zahlen).
- GET `/scoring/alerts/latest-per-sensor` gibt den jeweils letzten gespeicherten Alert je Sensor zurück (Antwort ebenfalls mit `items` + `summary`).
- GET `/scoring/alerts/metrics` liefert aggregierte Auswertungen (Severity total/recent, Top-Sensoren, Trenddaten für Charts).
- GET `/scoring/run-logs` zeigt die letzten Scoring-Läufe (inkl. Thresholds & Anzahl gespeicherter Alerts).
- Dashboard (Server): `GET /dashboard/alerts` verweist auf das React-Frontend (`http://localhost:5173`).
- React-Frontend bietet Severity-Summary, interaktive Charts, Filter, Pagination und Sensor-Detail-Overlay (nutzt `/scoring/alerts` sowie `/scoring/alerts/sensor/{sensor_no}`).
- POST `/scoring/run` + Scheduler (`analysis/schedule_scoring.py`) nutzen Standard-Schwellen (Umgebungsvariablen `SMARTFUEL_ALERT_THRESHOLD`, `SMARTFUEL_ALERT_CRITICAL_THRESHOLD`).
- Dashboard: `GET /dashboard/alerts` rendert ein einfaches HTML mit Filtermöglichkeiten (ebenfalls `x-api-key` nötig).

### Automatisierte Läufe
- Pipeline einmalig starten:

```bash
PYTHONPATH=. .venv/bin/python analysis/run_pipeline.py \
  --data-dir Data/Vaporix_side_a.csv \
  --model reports/models/fault_model_gbm.joblib \
  --metrics reports/fault_model_gbm_metrics.json \
  --top-n 200
```

- Dauerhafte Ausführung (inkl. Ingestion + Scoring): `analysis/schedule_scoring.py`

```bash
PYTHONPATH=. .venv/bin/python analysis/schedule_scoring.py \
  --model reports/models/fault_model_gbm.joblib \
  --metrics reports/fault_model_gbm_metrics.json \
  --top-n 200 \
  --data-dir Data/Vaporix_side_a.csv \
  --interval-minutes 30
```

- Umgebungsvariablen (optional in `.env`):
  * `SMARTFUEL_ALERT_THRESHOLD` (Standard 0.4)
  * `SMARTFUEL_ALERT_CRITICAL_THRESHOLD` (Standard 0.7)
  * `SMARTFUEL_SCORING_INTERVAL_MIN` (Standard 30)
  * `SMARTFUEL_SCORING_TOP_N` (Standard 50)
  * `SMARTFUEL_DATA_DIR` (Standard `Data/Vaporix_side_a.csv`)

### Lokaler Start (ohne Sandbox)
1. Backend: `PYTHONPATH=. .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000`
2. Frontend: `cd frontend && npm install && npm run dev` → `http://localhost:5173`

Sandbox-Hinweis: In dieser Umgebung sind Ports gesperrt (`operation not permitted`). Für den Produktivbetrieb das Projekt auf einer eigenen Maschine oder in Docker/Kubernetes starten.

### Benachrichtigungen
- Optionaler Slack-Webhook: `SMARTFUEL_SLACK_WEBHOOK` → kritische Alerts werden via `POST` an das Webhook gesendet.
- E-Mail Benachrichtigungen (für kritische Alerts):
  * `SMARTFUEL_SMTP_HOST`, `SMARTFUEL_SMTP_PORT`
  * `SMARTFUEL_SMTP_USERNAME`, `SMARTFUEL_SMTP_PASSWORD`
  * `SMARTFUEL_SMTP_SENDER`, `SMARTFUEL_SMTP_RECIPIENTS` (Komma-separiert)

### Docker
- Build & Run:

```bash
docker compose build
docker compose up
```

- Die Compose-Datei mountet `reports/`, `Data/` und `smartfuel.sqlite` als Volumes. Anpassungen können über `.env` oder Environment-Variablen erfolgen.

### Frontend (Vite + React)
- Projekt liegt unter `frontend/`

```bash
cd frontend
npm install
npm run dev
```

- Standard-URL: `http://localhost:5173` (CORS freigegeben). Setze `.env` im Frontend (z. B. `.env.local`) mit `VITE_API_URL=http://localhost:8000` und optional `VITE_API_KEY`.
- Das Dashboard bietet Filter, Pagination, Auto-Refresh sowie Charts. Datenbasis: `/scoring/alerts`, `/scoring/alerts/metrics`.

### Tests
- Schneller Smoke-Test:

```bash
PYTHONPATH=. .venv/bin/python -m unittest app.tests.test_basic
```

- Für weiterführende Tests bitte `pytest`, `pytest-asyncio`, `httpx` installieren und zusätzliche Suites ergänzen. Beachte bei allen CLI-Befehlen das Präfix `PYTHONPATH=.`, damit das Paket `app` gefunden wird.

### Continuous Integration
- GitHub Actions Workflow (`.github/workflows/ci.yml`) führt bei Push/PR automatisch `python -m unittest discover app/tests` aus.
- Stelle sicher, dass `requirements.txt` aktuell gehalten wird, damit die Pipeline erfolgreich bleibt.

### Reporting & CLI
- Alerts exportieren (CSV/JSON):

```bash
PYTHONPATH=. .venv/bin/python analysis/export_alerts.py \
  --limit 200 --format csv \
  --output reports/alert_export.csv \
  --runs-output reports/run_export.csv
```

- Alerts in der Konsole ansehen:

```bash
PYTHONPATH=. .venv/bin/python analysis/show_alerts.py
```

- Datenbank zurücksetzen:

```bash
PYTHONPATH=. .venv/bin/python analysis/reset_db.py
```

Hinweis: In der Sandbox können keine Ports gebunden werden (`operation not permitted`). Für einen laufenden FastAPI-Server daher lokal oder in einer geeigneten Umgebung deployen.
