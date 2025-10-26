import unittest
from pathlib import Path

import pandas as pd

from app.services.scoring import persist_scores
from app.db.session import AsyncSessionLocal
from app.models.models import VaporixAlert


class ScoringLogicTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async with AsyncSessionLocal() as session:
            await session.execute(VaporixAlert.__table__.delete())
            await session.commit()

    async def test_persist_scores_assigns_severity(self):
        data = pd.DataFrame(
            [
                {
                    "sensor_no": "TEST-1",
                    "date": pd.Timestamp("2024-01-01"),
                    "fault_probability": 0.8,
                    "fault_prediction": True,
                    "fault_event": False,
                    "fault_active": False,
                },
                {
                    "sensor_no": "TEST-2",
                    "date": pd.Timestamp("2024-01-02"),
                    "fault_probability": 0.5,
                    "fault_prediction": True,
                    "fault_event": False,
                    "fault_active": False,
                },
            ]
        )
        async with AsyncSessionLocal() as session:
            alerts, run = await persist_scores(
                session,
                data,
                model_path=Path("model.joblib"),
                metrics_path=Path("metrics.json"),
                warning_threshold=0.3,
                critical_threshold=0.7,
                log_run=True,
            )

        self.assertEqual(len(alerts), 2)
        severity = sorted([a.severity for a in alerts])
        self.assertEqual(severity, ["critical", "warning"])
        self.assertIsNotNone(run)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
