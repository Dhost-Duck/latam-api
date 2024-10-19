from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from .model import DelayModel
import pandas as pd

app = FastAPI()

# Initialize your model
model = DelayModel()

# Mock function to simulate loading a model
def load_model():
    features = pd.DataFrame({
        'min_diff': [30, 60, 120],
        'high_season': [1, 0, 1],
        'period_day_morning': [1, 0, 0],
        'period_day_afternoon': [0, 1, 0]
    })
    target = pd.Series([0, 1, 0])
    model.fit(features, target)

# Load the model
load_model()

# Define the structure of the data expected in the POST request
class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    Fecha_I: Optional[datetime] = None
    Fecha_O: Optional[datetime] = None

# Add validation logic
def validate_flight_data(flights: List[FlightData]):
    for flight in flights:
        if flight.MES < 1 or flight.MES > 12:
            raise HTTPException(status_code=400, detail=f"Invalid MES value: {flight.MES}")
        if flight.TIPOVUELO not in ["N", "I"]:  # Assuming "N" and "I" are valid
            raise HTTPException(status_code=400, detail=f"Invalid TIPOVUELO value: {flight.TIPOVUELO}")
        if not flight.OPERA:  # Make sure OPERA is not empty
            raise HTTPException(status_code=400, detail="OPERA field cannot be empty")

# Define the prediction endpoint
@app.post("/predict")
async def predict(flights: List[FlightData]):
    try:
        # Validate input
        validate_flight_data(flights)

        # Convert the flight data into a pandas DataFrame
        flight_data = pd.DataFrame([{
            "OPERA": flight.OPERA,
            "TIPOVUELO": flight.TIPOVUELO,
            "MES": flight.MES,
            "Fecha_I": flight.Fecha_I.isoformat() if flight.Fecha_I else None,
            "Fecha_O": flight.Fecha_O.isoformat() if flight.Fecha_O else None
        } for flight in flights])

        # Preprocess the data using the model's preprocess method
        preprocessed_data = model.preprocess(flight_data)

        # Make predictions using the model
        prediction = model.predict(preprocessed_data)

        return {"predict": prediction}

    except Exception as e:
        # Log the error
        print(f"Error: {str(e)}")  # Log the error details
        raise HTTPException(status_code=400, detail=str(e))
