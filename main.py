import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_navigation_bar import st_navbar


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition",
    layout="wide"
)


# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = model.predict(img_arr)
    return np.argmax(preds)


# --------------------------------------------------
# Navbar styles
# --------------------------------------------------
styles = {
    "nav": {"background-color": "rgb(123, 209, 146)"},
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "padding": "0.5rem",
    },
    "active": {"background-color": "rgba(255,255,255,0.25)"},
    "hover": {"background-color": "rgba(255,255,255,0.35)"},
}


# --------------------------------------------------
# Navbar
# --------------------------------------------------
page = st_navbar(["Home", "Disease Recognition", "About"], styles=styles)


# --------------------------------------------------
# 🔴 HARD SCROLL FIX (REQUIRED)
# --------------------------------------------------
st.markdown(
    """
    <style>
    html, body {
        height: auto !important;
        overflow-y: auto !important;
    }

    main {
        height: auto !important;
        overflow-y: auto !important;
    }

    section[data-testid="stAppViewContainer"] {
        height: auto !important;
        overflow-y: auto !important;
    }

    div.block-container {
        height: auto !important;
        overflow-y: auto !important;
        padding-bottom: 6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    with st.container():
        st.header("PLANT DISEASE RECOGNITION SYSTEM")

        st.image("home_page.jpg", use_container_width=True)

        st.markdown("""
        Welcome to the **Plant Disease Recognition System**.

        This application uses a deep learning model to detect plant diseases
        from leaf images and helps farmers and researchers take timely action.

        ### How it works
        1. Upload an image of a plant leaf
        2. The model analyzes the image
        3. Disease prediction is displayed instantly

        Navigate to **Disease Recognition** to begin.
        """)


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About":
    with st.container():
        st.header("About the Project")

        st.markdown("""
        #### Dataset
        The dataset is derived from the *New Plant Diseases Dataset* on Kaggle.

        - **Total images:** ~87,000
        - **Classes:** 38 plant disease categories
        - **Train:** 70,295 images  
        - **Validation:** 17,572 images  
        - **Test:** 33 images  

        Offline data augmentation was used to improve robustness and accuracy.
        """)


# --------------------------------------------------
# DISEASE RECOGNITION PAGE
# --------------------------------------------------
elif page == "Disease Recognition":
    with st.container():
        st.header("Disease Recognition")

        test_image = st.file_uploader(
            "Upload a plant leaf image",
            type=["jpg", "jpeg", "png"]
        )

        if test_image and st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        if test_image and st.button("Predict"):
            st.subheader("Prediction")

            result_index = model_prediction(test_image)

            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot',
                'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot',
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                'Potato___Early_blight',
                'Potato___Late_blight',
                'Potato___healthy',
                'Raspberry___healthy',
                'Soybean___healthy',
                'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy',
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            st.success(f"Prediction: **{class_name[result_index]}**")
