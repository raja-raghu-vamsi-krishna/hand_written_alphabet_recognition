import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

# Define the function to make predictions with the loaded model
def predict_with_loaded_model(model, image_data):
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(BytesIO(image_data))

    # Resize the image to match the model's expected input size (e.g., 15x15 pixels)
    resized_image = image.resize((15, 15))

    # Convert the resized image to grayscale and flatten it
    image_array = np.array(resized_image).flatten()

    # Make predictions using the loaded model
    predictions = model.predict([image_array])

    return predictions

# Load the previously dumped random forest model
model_path = 'rf_model.pkl'

with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Create a Flask app
app = Flask(__name__)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read the image data from the POST request
        image_data = request.files['image'].read()

        # Make predictions using the loaded model
        predictions = predict_with_loaded_model(loaded_model, image_data)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
