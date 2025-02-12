import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import json
import cv2
import os

# Load the Keras model
model = tf.keras.models.load_model("models/snakeimageclassifier.keras")

# Load class names dynamically from JSON file
try:
    with open("models/snake_class_names.json", "r") as f:
        snake_names = json.load(f)
except FileNotFoundError:
    snake_names = ["Unknown"]

# Define image preprocessing
def preprocess_image(image):
    image = image.resize((150, 150))  
    image = np.array(image)
    
    if image.ndim == 2:  
        image = np.stack([image] * 3, axis=-1)  
    
    image = image / 255.0

    if image.shape[-1] != 3:
        image = np.repeat(image, 3, axis=-1)  
    
    image = np.expand_dims(image, axis=0)  
    return image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        image = Image.open(file.stream)
        image = preprocess_image(image)

        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        snake_name = snake_names[predicted_class] if predicted_class < len(snake_names) else "Unknown"

        return jsonify({"prediction": snake_name, "confidence": float(confidence)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        video_path = "temp_video.mp4"
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        frame_skip = 10  # Process every 10th frame for speed
        frame_count = 0

        class_counts = {}
        class_confidences = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  

            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image = preprocess_image(image)

                predictions = model.predict(image)
                predicted_class = np.argmax(predictions)
                confidence = predictions[0][predicted_class]

                print(f"Frame {frame_count}: {snake_names[predicted_class]} ({confidence})")
                snake_name = snake_names[predicted_class] if predicted_class < len(snake_names) else "Unknown"

                # Store class occurrences
                if snake_name not in class_counts:
                    class_counts[snake_name] = 0
                    class_confidences[snake_name] = 0.0
                
                class_counts[snake_name] += 1
                class_confidences[snake_name] += confidence

            frame_count += 1

        cap.release()
        os.remove(video_path)  # Delete temp video after processing

        # Determine the most common prediction
        if not class_counts:
            return jsonify({"error": "No valid frames processed. Check video format and model compatibility."})

        final_class = max(class_counts, key=class_counts.get)
        avg_confidence = class_confidences[final_class] / class_counts[final_class]

        print(final_class, avg_confidence)
    

        return jsonify({"prediction": final_class, "confidence": float(avg_confidence)})

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
