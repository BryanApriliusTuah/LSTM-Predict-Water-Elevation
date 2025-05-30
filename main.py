from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

# Load the model and scaler
model = tf.keras.models.load_model('lstm_model.keras')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

# Request format
class SequenceInput(BaseModel):
    sequence: list  # expects a list of 7 float values

@app.post("/predict")
def predict(input_data: SequenceInput):
    sequence = np.array(input_data.sequence).reshape(-1, 1)
    scaled = scaler.transform(sequence)
    X = np.array([scaled])  # shape: (1, 7, 1)

    prediction_scaled = model.predict(X)
    prediction = scaler.inverse_transform(prediction_scaled)
    print(f"Scaled prediction: {float(prediction_scaled[0][0])}")
    print(f"Inverse transformed prediction: {float(prediction[0][0])}")

    return {"predicted_water_level": float(prediction[0][0])}
