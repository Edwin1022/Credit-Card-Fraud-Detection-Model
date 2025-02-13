import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("credit_card_fraud_detection_model.sav", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/")
def home():
    return "Credit Card Fraud Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get the JSON data from the api request
        data = request.get_json()

        input_data = pd.DataFrame([data])

        # check if input is provided
        if not data:
            return jsonify({"error": "Input data not provided"}), 400

        # validate input columns
        required_columns = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", 
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", 
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]

        if not all(col in input_data.columns for col in required_columns):
            return jsonify({"error": f"Required columns missing. Required columns: {required_columns}"}), 400

        # make prediction
        prediction = model.predict(input_data)

        # Debugging statements
        print("Input data:", input_data)
        print("Prediction:", prediction)

        # response
        response = {
            "prediction": "Fraud" if prediction[0] >= 0.9968540885963141 else "Normal"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(debug=True)