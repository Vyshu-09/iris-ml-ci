
import json
import os
import unittest
import joblib
import pandas as pd

class TestIrisPipeline(unittest.TestCase):

    def test_dataset_created(self):
        data = pd.read_csv(
            "iris_data.csv",
            header=None,
            skip_blank_lines=True
        )
        self.assertEqual(len(data.dropna(how="all")), 150)

    def test_model_created(self):
        self.assertTrue(os.path.exists("iris_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_valid(self):
        with open("metrics.json") as file:
            metrics = json.load(file)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_model_prediction(self):
        model = joblib.load("iris_model.pkl")
        sample = pd.DataFrame([{
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }])
        prediction = model.predict(sample)[0]
        self.assertEqual(prediction, "Iris-setosa")

if __name__ == "__main__":
    unittest.main()
