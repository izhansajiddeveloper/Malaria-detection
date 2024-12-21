import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved Keras model
model = tf.keras.models.load_model("final_malaria_model (2).keras")

# Define image preprocessing function
def preprocess_image(image):
    """
    Preprocess the uploaded image for the model.
    - Resize to (150, 150).
    - Normalize pixel values to [0, 1].
    - Expand dimensions to match model input.
    """
    image = image.resize((150, 150))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define prediction function
def predict_image(image):
    """
    Predict whether the image is 'Infected' or 'Uninfected'.
    """
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]  # Get the prediction score
    
    # Invert: "Infected" = prediction <= 0.5, "Uninfected" = prediction > 0.5
    return "Uninfected" if prediction > 0.5 else "Infected"  # Inverted labels

# Streamlit UI
st.title("Malaria Detection App")
st.write("Upload an image of a cell, and the app will predict if it is infected with malaria or not.")

# File uploader
uploaded_file = st.file_uploader("Upload Cell Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB format
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Predict and display the result
    st.write("Processing...")
    prediction = predict_image(image)
    st.success(f"Prediction: {prediction}")
