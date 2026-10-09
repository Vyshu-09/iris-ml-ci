import unittest
 
from app import app
 
 
class TestPredictionApplication(unittest.TestCase):
 
    def setUp(self):
        self.client = app.test_client()
 
    def test_health_endpoint(self):
        response = self.client.get("/")
 
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
 
    def test_setosa_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )
 
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "Iris-setosa"
        )
 
    def test_virginica_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "sepal_length": 6.7,
                "sepal_width": 3.0,
                "petal_length": 5.2,
                "petal_width": 2.3
            }
        )
 
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
           "Iris-setosa"
        )
 
    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5
            }
        )
 
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())
 
 
if __name__ == "__main__":
    unittest.main()
