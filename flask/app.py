
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# from PIL import Image
# import io

app = Flask(__name__)
CORS(app)

# Load your trained model
# model = tensorflow.keras.models.load_model(os.path.join('artifacts\\trained_model','model_v-03.h5'))

# model = tensorflow.keras.models.load_model('C:\\flask_deploy\\flask\\flask\\model\\model_v-03.h5')  ## loading the model

model = tensorflow.keras.models.load_model('/app/model/model_v-03.h5')


@app.route('/predict', methods=['POST'])
def pred():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        # Process the image
        image = image.open(file.stream).convert('RGB')
        image = image.resize((256, 256))  # Resize to match your model's input shape
        # image_array = np.array(image) / 255.0  # Normalize the image
        image_array = np.array(image) # no need to normalize 
        # if do normalize he output will different
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image_array)
        print(prediction)
        print()
        print(prediction[0][0])
        # prediction output 2 dimension array [[0.5134971]]
        predicted_class = 'Cat' if prediction[0][0] < 0.5 else 'Dog'
        print(f"Predicted class: {predicted_class}")

        return jsonify({'class': predicted_class}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
