from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import os
import numpy as np
from PIL import Image  # Correct import for PIL

# Suppress unnecessary TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model_v-03.h5')
if model_path:
    print(model_path)
    model = tf.keras.models.load_model(model_path)
else:
    print('failed in loading model')

@app.route('/')
def home():
    return jsonify({'message': 'Flask API is running!'})

@app.route('/predict', methods=['POST'])
def pred():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        # Process the image
        img = Image.open(file.stream).convert('RGB')  # Load the image and convert to RGB
        img = img.resize((256, 256))  # Resize to match the model's input shape
        img_array = np.array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        print(f"Raw prediction: {prediction}")
        predicted_class = 'Cat' if prediction[0][0] < 0.5 else 'Dog'
        print(f"Predicted class: {predicted_class}")

        return jsonify({'class': predicted_class}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
