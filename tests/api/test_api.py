import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from challenge import app

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("challenge.model.DelayModel.predict")
    def test_should_get_predict(self, mock_predict):
        # Mocking the model prediction
        mock_predict.return_value = [0]

        data = [
            {
                "OPERA": "Aerolineas Argentinas",
                "TIPOVUELO": "N",
                "MES": 3,
                "Fecha_I": "2023-01-01T12:00:00",
                "Fecha_O": "2023-01-01T14:00:00"
            }
        ]
        response = self.client.post("/predict", json=data)
        print(response.json())  # Log the response details
        print(response.status_code)  # Log the status code
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})

    @patch("challenge.model.DelayModel.predict")
    def test_should_failed_unkown_column_1(self, mock_predict):
        mock_predict.return_value = [0]

        data = [
            {
                "OPERA": "Aerolineas Argentinas",
                "TIPOVUELO": "N",
                "MES": 13  # Invalid MES
            }
        ]
        response = self.client.post("/predict", json=data)
        print(response.json())  # Log the response details
        print(response.status_code)  # Log the status code
        self.assertEqual(response.status_code, 400)

    @patch("challenge.model.DelayModel.predict")
    def test_should_failed_unkown_column_2(self, mock_predict):
        mock_predict.return_value = [0]

        data = [
            {
                "OPERA": "Aerolineas Argentinas",
                "TIPOVUELO": "O",  # Invalid TIPOVUELO
                "MES": 13  # Invalid MES
            }
        ]
        response = self.client.post("/predict", json=data)
        print(response.json())  # Log the response details
        print(response.status_code)  # Log the status code
        self.assertEqual(response.status_code, 400)

    @patch("challenge.model.DelayModel.predict")
    def test_should_failed_unkown_column_3(self, mock_predict):
        mock_predict.return_value = [0]

        data = [
            {
                "OPERA": "Argentinas",  # Invalid OPERA
                "TIPOVUELO": "O",  # Invalid TIPOVUELO
                "MES": 13  # Invalid MES
            }
        ]
        response = self.client.post("/predict", json=data)
        print(response.json())  # Log the response details
        print(response.status_code)  # Log the status code
        self.assertEqual(response.status_code, 400)
