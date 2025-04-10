# API/main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

# Load your trained model
model = joblib.load("model/model_NB.pkl")
label = joblib.load("model/label_encoders.pkl")
onehot = joblib.load("model/onehot_encoder.pkl")

# Define FastAPI API
app = FastAPI()

# Define input data model
class InputData(BaseModel):
    features: list  # or more specific: List[float]

@app.post("/predict/")
async def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
