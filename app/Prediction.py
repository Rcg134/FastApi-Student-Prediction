from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define a Pydantic model for the request body
class PredictionRequest(BaseModel):
    data: List[dict]  # List of dictionaries containing 'Activity' and 'Grade'

# Define a Pydantic model for the response body
class PredictionResponse(BaseModel):
    predictions: List[str]  # List of predicted grades for the next activities

# Initialize FastAPI
app = FastAPI()

# Define the grading scale
grading_scale = ['E', 'D', 'C', 'B', 'A']

# Function to predict grades for the next activities using scikit-learn
def predict_next_grades(data: List[dict]) -> List[str]:
    # Extract features and target
    X = np.array([record['Activity'] for record in data]).reshape(-1, 1)
    y = np.array([record['Grade'] for record in data])

    # Perform label encoding for grades
    label_encoder = LabelEncoder()
    label_encoder.fit(grading_scale)
    y_encoded = label_encoder.transform(y)

    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(X, y_encoded)

    # Predict grades for the next activities
    next_activities = np.array(range(data[-1]['Activity'] + 1, data[-1]['Activity'] + 10)).reshape(-1, 1)
    predictions_encoded = model.predict(next_activities)

    # Round predictions and ensure they fall within the grading scale range
    rounded_predictions = np.round(predictions_encoded).astype(int)
    predictions_clipped = np.clip(rounded_predictions, 0, len(grading_scale) - 1)

    # Decode the predicted grades
    predictions = label_encoder.inverse_transform(predictions_clipped)

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
    predicted_grades = predict_next_grades(data)

    # Generate the list of activities for the response
    next_activities = [data[-1]['Activity'] + i + 1 for i in range(10)]

    # Combine the activities and predictions into a list of dictionaries
    result = [{"Activity": activity, "Grade": grade} for activity, grade in zip(next_activities, predicted_grades)]

    # Return the predictions along with the activities
    return {"predictions": result}
