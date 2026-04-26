import gradio as gr
import pandas as pd
import pickle
import numpy as np
import joblib


best_model = joblib.load("diabetes_prediction.pkl")   

def predict_diabetes(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = best_model.predict(input_data)

    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        "number","number","number","number",
        "number","number","number","number"
    ],
    outputs="text",
    title="Diabetes Prediction App"
)

interface.launch()