import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'braintumor.h5'
model = None
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define class labels corresponding to the training data
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Format class names for display
def format_class_name(class_name):
    return class_name.replace('_', ' ').title()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400

    try:
        # Read the image file using OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file.'}), 400

        # Preprocess the image
        img = cv2.resize(img, (150, 150))
        # Note: Model was trained without dividing by 255.0, so we pass raw pixel values
        img_array = np.reshape(img, (1, 150, 150, 3))

        # Predict
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        predicted_class = labels[class_idx]
        formatted_class = format_class_name(predicted_class)

        return jsonify({
            'class': predicted_class,
            'formatted_class': formatted_class,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the app
    app.run(debug=True, port=5000)
