import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

MODEL_PATH = 'alexnet_image_classifier.h5'
IMG_HEIGHT = 227
IMG_WIDTH = 227

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() 

model = load_model()

def predict_image(img_file):
    '''Prediction Function'''
    if img_file is None:
        return None, None

    try:
        img = Image.open(img_file).convert('RGB')
        img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))

        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array /= 255.0 

        prediction_raw = model.predict(img_array, verbose=0)
        probability = prediction_raw[0][0]

        if probability > 0.5:
            result = "AI-generated"
        else:
            result = "Human-generated"

        return result, probability

    except Exception as e:
        st.error(f"Error processing image or making prediction: {e}")
        return None, None

# --- Streamlit UI ---
st.set_page_config(page_title="AI vs. Human Image Classifier", layout="centered")

st.title("🧠 AI vs. Human-Generated Image Classifier")
st.markdown("Upload an image to determine if it's likely AI-generated or human-generated.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Get prediction
    result, probability = predict_image(uploaded_file)

    if result is not None:
        st.subheader("Prediction:")
        if result == "AI-generated":
            st.error(f"This image is likely **{result}**")
        else:
            st.success(f"This image is likely **{result}**")

        st.write(f"Confidence: **{probability:.4f}**")

        # Display a simple gauge/progress bar for confidence
        if probability is not None:
            st.markdown("--- Say, you want to see how 'sure' the model is: ---")
            if result == "AI-generated":
                st.progress(float(probability), text=f"AI-generated likelihood: {probability:.1%}")
            else:
                st.progress(float(1 - probability), text=f"Human-generated likelihood: {(1 - probability):.1%}")

else:
    st.info("Please upload an image to get a classification.")

print("Streamlit app 'app.py' created.")