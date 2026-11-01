import unittest
from app import app


class TestStudentApp(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        data = {
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 90,
            "previous_score": 88
        }

        response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)

    def test_invalid_prediction(self):
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()