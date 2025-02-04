import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# Load the trained model
model_path = r"C:\Users\jay\Desktop\fake_logo_detection\fake_logo_detection\fake_logo_detector.keras"
model = tf.keras.models.load_model(model_path)

# Image Preprocessing Function
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize images to 224x224
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to predict the image
def predict_image(image):
    img_array = preprocess_image(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Genuine"
    else:
        return "Fake"

# Streamlit UI
st.title("Fake Logo Detection")
st.write("Upload an image to classify it as Genuine or Fake.")

# File uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image using PIL
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the image
    if st.button("Classify Image"):
        result = predict_image(image)
        st.write(f"The image is classified as: {result}")
