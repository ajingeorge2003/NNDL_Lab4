from pyngrok import ngrok
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("intel_cnn_model.h5")
    return model

model = load_model()

# Class labels
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Streamlit UI
st.set_page_config(page_title="Intel Image Classifier", page_icon="🌍")
st.title("🌍 Intel Image Classification using CNN")
st.write("Upload an image of a natural scene and let the model classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: **{class_names[class_idx]}**")
    st.write(f"Confidence: {confidence:.2f}%")
