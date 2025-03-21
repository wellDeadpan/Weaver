# app/main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

# Load your trained model
model = joblib.load("app/model.pkl")

# Define FastAPI app
app = FastAPI()

# Define input data schema
class SymptomInput(BaseModel):
    symptoms_text: str

@app.post("/predict/")
async def predict_disease(input_data: SymptomInput):
    # Simple preprocessing (can expand)
    text = input_data.symptoms_text
    prediction = model.predict([text])[0]
    return {"predicted_disease": prediction}
