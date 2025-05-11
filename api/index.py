from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image
import io
import os

# Load model
model = tf.keras.models.load_model('flower-classification.keras')

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Sesuaikan dengan class Anda

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flower Classification API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    try:
        image = Image.open(file)
        image = image.resize((180, 180))
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_label = class_names[np.argmax(score)]

        return jsonify({
            "prediction": predicted_label,
            "confidence": f"{100 * np.max(score):.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
