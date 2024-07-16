import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_navigation_bar import st_navbar


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element



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

#Sidebar
page = st_navbar(["Home", "Documentation", "Disease Recognition", "About",],styles=styles)


#Main Page
if(page=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to empower you with the tools needed to identify plant diseases swiftly and accurately. By simply uploading an image of a plant, our advanced system will analyze it to detect any signs of disease, helping to safeguard your crops and ensure a bountiful harvest.

    ### How It Works
    1. **Upload Image:** Navigate to the **Disease Recognition** page and upload an image of the plant you suspect may be diseased.
    2. **Analysis:** Leveraging cutting-edge algorithms, our system will process the image to identify any potential diseases.
    3. **Results:** Receive a detailed analysis and recommendations for appropriate actions to take.

    ### Why Choose Us?
    - **Accuracy:** Our system employs state-of-the-art machine learning techniques to deliver precise disease detection.
    - **User-Friendly:** Designed with simplicity and ease-of-use in mind, our interface ensures a seamless user experience.
    - **Fast and Efficient:** Obtain results in seconds, enabling you to make quick, informed decisions.
                
    ### Get Started
    To experience the full capabilities of our Plant Disease Recognition System, head over to the **Disease Recognition** page in the sidebar. Upload an image and let our technology work for you!

    ## About Us
    Discover more about our project, and our vision on the About page. We are committed to advancing agricultural health and productivity through innovative technology.
    """)

#About Project
elif(page=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset has been meticulously curated using offline augmentation techniques based on the original dataset, which can be accessed via this [GitHub repository](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data). It encompasses approximately 87,000 RGB images of both healthy and diseased crop leaves, categorized into 38 distinct classes.
                The dataset is thoughtfully divided to maintain the directory structure, with an 80/20 split between training and validation sets. Additionally, a new directory containing 33 test images has been created specifically for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                This comprehensive dataset is pivotal for developing robust machine learning models aimed at accurately identifying various plant diseases, thereby contributing to improved agricultural health and productivity.

                """)

#Prediction Page
elif(page=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
