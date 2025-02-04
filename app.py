import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import json

# Load the Keras model
model = tf.keras.models.load_model("models/snakeimageclassifier.keras")

# Load class names dynamically from JSON file
try:
    with open("models/class_names.json", "r") as f:
        snake_names = json.load(f)  # Load class names as an array
except FileNotFoundError:
    snake_names = ["Unknown"]  # Fallback if file is missing

# Define a transformation for the image
def preprocess_image(image):
    # Resize the image to match model input size (150x150)
    image = image.resize((150, 150))  
    image = np.array(image)  # Convert to numpy array
    
    # Convert grayscale to RGB
    if image.ndim == 2:  
        image = np.stack([image] * 3, axis=-1)  # Stack grayscale across 3 channels
    
    # Normalize the image to [0, 1]
    image = image / 255.0

    # Ensure image has 3 channels
    if image.shape[-1] != 3:
        image = np.repeat(image, 3, axis=-1)  
    
    # Add batch dimension for model prediction
    image = np.expand_dims(image, axis=0)  
    
    return image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        # Process the image
        image = Image.open(file.stream)
        image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        # Get the predicted snake name
        snake_name = snake_names[predicted_class] if predicted_class < len(snake_names) else "Unknown"

        return jsonify({"prediction": snake_name, "confidence": float(confidence)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
