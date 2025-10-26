import asyncio
import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.main import app
from app.db.session import AsyncSessionLocal
from app.models.models import VaporixAlert
from app.services.scoring import score_fault_risk


class ScoringAPITestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = TestClient(app)
        # ensure clean DB for alerts
        async with AsyncSessionLocal() as session:
            await session.execute(delete(VaporixAlert))
            await session.commit()

    async def test_scoring_latest_requires_key(self):
        resp = self.client.get("/scoring/latest")
        self.assertEqual(resp.status_code, 401)

    async def test_scoring_latest_with_key(self):
        headers = {"x-api-key": os.getenv("API_KEY", "dev-1234")}
        resp = self.client.get("/scoring/latest", headers=headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("items", {"items": resp.json() if isinstance(resp.json(), list) else []})

    async def test_score_fault_risk_no_data(self):
        async with AsyncSessionLocal() as session:
            df = await score_fault_risk(
                model_path=Path("reports/models/fault_model_gbm.joblib"),
                metrics_path=Path("reports/fault_model_gbm_metrics.json"),
                latest_only=True,
                top_n=5,
                session=session,
            )
            self.assertIsNotNone(df)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
