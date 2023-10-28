import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout

# Load your pre-trained model for age and gender prediction
model_path = "age_gender_predictor.h5"  # Replace with the path to your model
model = keras.models.load_model(model_path)

# Function to predict age and gender


def predict_age_and_gender(image):
    # Resize the image to (200, 200) and preprocess it
    image = cv2.resize(image, (200, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Make predictions using the loaded model
    predictions = model.predict(image)

    # Extract age and gender predictions
    age_prediction = int(predictions[0][0])
    gender_prediction = "Male" if predictions[1][0] >= 0.5 else "Female"

    return age_prediction, gender_prediction


# Streamlit app
st.title("Age and Gender Prediction App")

uploaded_image = st.file_uploader(
    "Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    age, gender = predict_age_and_gender(image)

    if age and gender:
        st.write(f"Predicted Age: {age} years")
        st.write(f"Predicted Gender: {gender}")
