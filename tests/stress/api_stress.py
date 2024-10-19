from locust import HttpUser, task

class StressUser(HttpUser):
    
    @task
    def predict_argentinas(self):
        self.client.post(
            "/predict", 
            json=[
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3,
                    "Fecha_I": "2023-01-01T12:00:00",
                    "Fecha_O": "2023-01-01T14:00:00"
                }
            ]
        )

    @task
    def predict_latam(self):
        self.client.post(
            "/predict", 
            json=[
                {
                    "OPERA": "Grupo LATAM", 
                    "TIPOVUELO": "N", 
                    "MES": 3,
                    "Fecha_I": "2023-01-01T12:00:00",
                    "Fecha_O": "2023-01-01T14:00:00"
                }
            ]
        )
