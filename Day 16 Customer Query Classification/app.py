from flask import Flask, request, jsonify
import joblib
import re


# Create Flask app
app = Flask(__name__)


# Load model files
model = joblib.load("Day 16 Customer Query Classification\\models\\banking_classifier.pkl")
vectorizer = joblib.load("Day 16 Customer Query Classification\\models\\tfidf_vectorizer.pkl")
encoder = joblib.load("Day 16 Customer Query Classification\\models\\label_encoder.pkl")


# Text preprocessing (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Bank Customer Query Classification API is running"
    })


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({
            "error": "Please provide a query"
        }), 400


    query = data["query"]


    # Clean text
    cleaned_query = clean_text(query)


    # Convert to TF-IDF vector
    vector = vectorizer.transform([cleaned_query])


    # Predict category
    prediction = model.predict(vector)


    # Convert numeric label to actual category
    category = encoder.inverse_transform(prediction)[0]


    return jsonify({
        "query": query,
        "predicted_category": category
    })


# Run server
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )