import gradio as gr
import numpy as np
import pickle
import os

# Load model using pickle
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    best_model = pickle.load(f)

# Prediction function
def predict_diabetes(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = best_model.predict(input_data)

    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Gradio interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Diabetes Prediction App"
)

interface.launch()