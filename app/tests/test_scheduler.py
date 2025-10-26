import unittest
from pathlib import Path

from app.db.session import AsyncSessionLocal
from app.models.models import ScoringRun, VaporixAlert
from app.services.scoring import score_fault_risk, persist_scores


class SchedulerLogicTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async with AsyncSessionLocal() as session:
            await session.execute(VaporixAlert.__table__.delete())
            await session.execute(ScoringRun.__table__.delete())
            await session.commit()

    async def test_run_log_created(self):
        async with AsyncSessionLocal() as session:
            scores = await score_fault_risk(
                model_path=Path("reports/models/fault_model_gbm.joblib"),
                metrics_path=Path("reports/fault_model_gbm_metrics.json"),
                latest_only=True,
                top_n=5,
                session=session,
            )
            alerts, run = await persist_scores(
                session,
                scores,
                model_path=Path("reports/models/fault_model_gbm.joblib"),
                metrics_path=Path("reports/fault_model_gbm_metrics.json"),
                warning_threshold=0.3,
                critical_threshold=0.7,
                log_run=True,
            )

        self.assertIsNotNone(run)
        self.assertGreaterEqual(run.alerts_saved, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
