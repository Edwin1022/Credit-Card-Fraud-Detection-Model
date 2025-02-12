import pickle
import pandas as pd
from flask import Flask, request, jsonify
import logging
from logging.handlers import SocketHandler

# Create Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = app.logger
logger.setLevel(logging.INFO)

# Load model
try:
    with open("credit_card_fraud_detection_model.sav", "rb") as model_file:
        model = pickle.load(model_file)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route("/")
def home():
    logger.info("Home endpoint accessed")
    return "Credit Card Fraud Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Prediction endpoint accessed")
    try:
        # get the JSON data from the api request
        data = request.get_json()
        logger.info(f"Received data: {data}")

        input_data = pd.DataFrame([data])
        logger.info(f"Created DataFrame with shape: {input_data.shape}")

        # check if input is provided
        if not data:
            logger.warning("No input data provided")
            return jsonify({"error": "Input data not provided"}), 400

        # validate input columns
        required_columns = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", 
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", 
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]

        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            logger.warning(f"Missing columns in input data: {missing_columns}")
            return jsonify({"error": f"Required columns missing. Required columns: {required_columns}"}), 400

        # make prediction
        logger.info("Making prediction")
        prediction = model.predict(input_data)
        logger.info(f"Prediction result: {prediction}")

        # response
        response = {
            "prediction": "Fraud" if prediction[0] >= 0.9968540885963141 else "Normal"
        }
        logger.info(f"Final response: {response}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    logger.info("Starting the Flask application")
    app.run(debug=True)