import streamlit as st
import pickle
import numpy as np
from PIL import Image
from io import BytesIO

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

# Streamlit app
st.title("Image Prediction App")

# Upload an image file using Streamlit's file uploader
image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image:
    # Read the image data as bytes
    image_data = image.read()

    # Display the uploaded image
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Make predictions when a user clicks a button
    if st.button("Predict"):
        # Make predictions using the loaded model
        predictions = predict_with_loaded_model(loaded_model, image_data)

        # Display the predictions
        st.write(f"Predictions: {predictions}")





