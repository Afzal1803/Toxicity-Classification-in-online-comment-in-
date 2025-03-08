from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences  # Updated import

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the trained AI model
model = tf.keras.models.load_model("tanglish_classification_model.h5")  # Ensure this file exists

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess input text
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])  # Convert text to numbers
    padded_seq = pad_sequences(seq, maxlen=100)  # Adjust maxlen based on training
    return padded_seq

@app.route("/")
def home():
    return "Flask server is running! Use /predict to check comments."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    comment = data.get("comment", "")

    if not comment:
        return jsonify({"error": "No comment provided"}), 400

    processed_comment = preprocess_text(comment)
    prediction = model.predict(processed_comment)

    is_toxic = prediction[0][0] > 0.5  # Adjust threshold if needed

    return jsonify({"toxic": bool(is_toxic)})

if __name__ == "__main__":
    app.run(debug=True)
