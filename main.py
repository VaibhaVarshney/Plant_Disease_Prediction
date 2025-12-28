import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_navigation_bar import st_navbar


# --------------------------------------------------
# Page configuration (recommended)
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition",
    layout="wide"
)


# --------------------------------------------------
# Load model once and cache it
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()


# --------------------------------------------------
# TensorFlow Model Prediction
# --------------------------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# --------------------------------------------------
# Navbar Styling
# --------------------------------------------------
styles = {
    "nav": {
        "background-color": "rgb(123, 209, 146)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}


# --------------------------------------------------
# Navigation Bar
# --------------------------------------------------
page = st_navbar(["Home", "Disease Recognition", "About"], styles=styles)


# --------------------------------------------------
# 🔴 SCROLL FIX — DO NOT REMOVE
# --------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            height: auto !important;
            overflow-y: auto !important;
        }

        [data-testid="stAppViewContainer"] {
            padding-bottom: 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Home Page
# --------------------------------------------------
if page == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")

    image_path = "home_page.jpg"
    st.image(image_path, use_container_width=True)

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to empower users to identify plant diseases quickly and accurately using deep learning.

    ### How It Works
    1. Upload a plant image
    2. The model analyzes the image
    3. Disease prediction is displayed instantly

    ### Why Choose This System?
    - High accuracy deep learning model
    - Simple and intuitive interface
    - Fast predictions

    Navigate to **Disease Recognition** to get started.
    """)


# --------------------------------------------------
# About Page
# --------------------------------------------------
elif page == "About":
    st.header("About")

    st.markdown("""
    #### Dataset Information

    The dataset is based on the **New Plant Diseases Dataset** from Kaggle and contains
    approximately **87,000 RGB images** across **38 classes**.

    #### Dataset Split
    - **Train:** 70,295 images  
    - **Validation:** 17,572 images  
    - **Test:** 33 images  

    The dataset uses offline augmentation to improve model generalization and robustness.
    """)


# --------------------------------------------------
# Disease Recognition Page
# --------------------------------------------------
elif page == "Disease Recognition":
    st.header("Disease Recognition")

    test_image = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

    if test_image and st.button("Show Image"):
        st.image(test_image, use_container_width=True)

    if test_image and st.button("Predict"):
        st.subheader("Prediction Result")

        result_index = model_prediction(test_image)

        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight',
            'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        st.success(f"🌱 The model predicts: **{class_name[result_index]}**")
