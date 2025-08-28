from flask import Flask, request, jsonify
import joblib
import numpy as np

# load the model
model = joblib.load("../model/iris_model.pkl")

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