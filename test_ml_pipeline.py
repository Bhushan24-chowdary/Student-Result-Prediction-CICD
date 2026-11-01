import unittest
import os
import pickle
import subprocess
import sys


class TestMLPipeline(unittest.TestCase):

    def test_model_training(self):

        result = subprocess.run(
            [sys.executable, "train_model.py"],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)

        self.assertTrue(
            os.path.exists("student_result_model.pkl")
        )

    def test_model_loading(self):

        self.assertTrue(
            os.path.exists("student_result_model.pkl")
        )

        with open("student_result_model.pkl", "rb") as file:
            model = pickle.load(file)

        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()