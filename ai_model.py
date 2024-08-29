import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
import logging
from werkzeug.exceptions import HTTPException
import requests
from sklearn.metrics import confusion_matrix
import joblib

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define countries for prediction mapping
COUNTRIES = ["Australia", "Canada", "Germany", "UK", "US"]

# Load model dynamically
def load_model():
    try:
        # Fetch model from a remote server
        response = requests.get("https://your-model-server.com/latest-model.pkl")
        response.raise_for_status()
        model = pickle.loads(response.content)
        logger.info("Model successfully loaded from remote server.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

model = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]

        # Make prediction
        prediction = model.predict(final_features)
        output = prediction[0]

        # Get country name
        country = COUNTRIES[output]

        # AI-driven prediction explanation
        explanation = get_prediction_explanation(final_features[0], model)

        # Log the prediction
        logger.info(f"Prediction made: {country}")

        # Prepare response with prediction and explanation
        prediction_text = f"Likely country: {country}"
        explanation_text = f"Explanation: {explanation}"

        return render_template("index.html", prediction_text=prediction_text, explanation_text=explanation_text)

    except ValueError as e:
        logger.error(f"Value error: {e}")
        return render_template("index.html", prediction_text="Error: Invalid input.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return render_template("index.html", prediction_text="Error: An unexpected error occurred.")

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    logger.error(f"HTTP exception occurred: {e}")
    return jsonify({"error": str(e.description)}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": "An unexpected error occurred."}), 500

def get_prediction_explanation(features, model):
    """
    AI-driven function to provide explanation for predictions.
    Placeholder function - should be replaced with actual implementation.
    """
    # This is a placeholder for the explanation logic.
    # For real use, you would integrate with an AI model or service that can provide explanations.
    # Example: LIME, SHAP, or a custom explanation model.
    return f"The model predicted {COUNTRIES[np.argmax(model.predict([features]))]} based on the provided features."

if __name__ == "__main__":
    app.run(debug=True)
