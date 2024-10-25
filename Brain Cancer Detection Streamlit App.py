import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('/Users/purplerain/Desktop/Jupyter/brain Cancer Detection.h5')

# Function to predict tumor
def predict_tumor(image, model):
    if image.mode != "RGB":  # Check if the image is not already in RGB
        image = image.convert("RGB")  # Convert grayscale to RGB
    image = image.resize((256, 256))  # Resize image to the input size of the model (256x256)
    img_array = img_to_array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
st.title("Brain Tumor Detection")

# File uploader to allow user to upload any size image
uploaded_file = st.file_uploader("Upload a brain scan image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Button to trigger prediction
    if st.button("Predict"):
        # Predict if the tumor is detected
        prediction = predict_tumor(image, model)

        # Assuming model gives a probability, and using threshold 0.5
        if prediction[0][0] > 0.5:
            st.error("Tumor Detected! Consult Your Doctor ASAP.")
        else:
            st.success("No Tumor Detected.")
