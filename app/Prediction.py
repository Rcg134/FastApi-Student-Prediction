from http.client import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LinearRegression
import numpy as np

# Define a Pydantic model for the request body
class PredictionRequest(BaseModel):
    data: List[dict]  # List of dictionaries containing 'Activity' and 'Grade'

# Define a Pydantic model for the response body
class PredictionResponse(BaseModel):
    predictions: List[float]  # List of predicted grades for the next activities

# Initialize FastAPI
app = FastAPI()

# Function to predict grades for the next activities using scikit-learn
def predict_next_activities(data: List[dict]) -> List[float]:
    # Extract features and target
    X = np.array([record['Activity'] for record in data]).reshape(-1, 1)
    y = np.array([record['Grade'] for record in data])

    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict grades for the next activities
    next_activities = np.array(range(data[-1]['Activity'] + 1, data[-1]['Activity'] + 5)).reshape(-1, 1)
    predictions = model.predict(next_activities)

    return predictions.tolist()

# Define a POST endpoint to receive input data and return predictions
@app.post("/predict/")
async def predict_grades(request: PredictionRequest):
    data = request.data

    # Check if data is provided
    if not data:
        raise HTTPException(status_code=400, detail="Data not provided")

    # Check if there are at least 2 records for prediction
    if len(data) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction")

    # Predict grades for the next activities
    predictions = predict_next_activities(data)

    # Generate the list of activities for the response
    next_activities = [data[-1]['Activity'] + i + 1 for i in range(4)]

    # Combine the activities and predictions into a list of dictionaries
    result = [{"Activity": activity, "Grade": grade} for activity, grade in zip(next_activities, predictions)]

    # Return the predictions along with the activities
    return {"predictions": result}
