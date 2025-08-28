from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# load the model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where main.py is
MODEL_PATH = os.path.join(BASE_DIR, "../model/iris_model.pkl")

model = joblib.load(MODEL_PATH)
# model = joblib.load("../model/iris_model.pkl")

# create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Flower Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() # Expect json input
    features = np.array(data["features"]).reshape(1, -1) # 2d array
    prediction = model.predict(features)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)