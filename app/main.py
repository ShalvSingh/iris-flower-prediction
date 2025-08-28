from flask import Flask, request, render_template ,jsonify
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
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # data = request.get_json() # Expect json input
    # features = np.array(data["features"]).reshape(1, -1) # 2d array
    # prediction = model.predict(features)[0]

    # return jsonify({"prediction": int(prediction)})
    if request.method == "POST":
        try:
            # Read form values
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])

            # Make prediction
            features = np.array([[sl, sw, pl, pw]])
            prediction = model.predict(features)[0]

            return render_template("index.html", prediction=prediction)
        except Exception as e:
            return str(e)
if __name__ == "__main__":
    app.run(debug=True)