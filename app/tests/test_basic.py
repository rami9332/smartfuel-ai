import unittest

from fastapi.testclient import TestClient

from app.main import app


class BasicAPITestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("SmartFuel", resp.json().get("message", ""))

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
