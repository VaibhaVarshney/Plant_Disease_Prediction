import streamlit as st
import tensorflow as tf
import numpy as np


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition System",
    layout="wide"
)


# --------------------------------------------------
# Load model
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
# Sidebar navigation (SAFE)
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    ["Home", "Disease Recognition", "About"]
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")

    st.image("home_page.jpg", use_container_width=True)

    st.markdown("""
    Welcome to the **Plant Disease Recognition System**.

    This application uses a deep learning model to identify plant diseases
    from leaf images and provide fast, reliable predictions.

    ### How It Works
    1. Upload a plant leaf image  
    2. The model analyzes the image  
    3. The predicted disease is displayed  

    Navigate to **Disease Recognition** from the sidebar to begin.
    """)


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About":
    st.header("About")

    st.markdown("""
    #### Dataset Information

    - ~87,000 RGB images
    - 38 disease and healthy plant classes
    - Offline augmentation applied

    **Split**
    - Train: 70,295 images  
    - Validation: 17,572 images  
    - Test: 33 images  
    """)


# --------------------------------------------------
# DISEASE RECOGNITION PAGE
# --------------------------------------------------
elif page == "Disease Recognition":
    st.header("Disease Recognition")

    test_image = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if test_image:
        st.image(test_image, use_container_width=True)

        if st.button("Predict"):
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
