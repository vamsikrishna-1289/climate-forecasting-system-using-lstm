from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from model import train_model, predict_future
from preprocess import preprocess_data


app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv(r"C:\Users\DELL\PycharmProjects\Deep Learning\climate_change_data.csv")
df = preprocess_data(df)

# Train the LSTM model
model, scaler, pca = train_model(df)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file:
            filepath = "climate_change_data.csv"
            uploaded_file.save(filepath)
            global df, model, scaler, pca
            df = pd.read_csv(filepath)
            df = preprocess_data(df)
            model, scaler, pca = train_model(df)
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    future_steps = 12  # Predict 12 months ahead
    predictions = predict_future(model, df, scaler, pca, future_steps)
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    app.run(debug=True)
