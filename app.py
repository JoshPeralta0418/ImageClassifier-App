import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
    
# Add Lottiefiles to add modern look and feel and lightweight animation in the web app

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_tno6cg2w.json")
model_architecture = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_DbCYKfCXBZ.json")
satellite = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_7msclaec.json")
    
# Introduction
st.title("Deploying Deep Learning Model in Web App")
st.subheader("This app contains instructions on how to navigate and interact with a deep learning model using **Streamlit App**")
# st.write("I am passionate about data science")

st_lottie(lottie_coding, height=300, key="coding")



# --- Create Description ---
with st.container():
    st.write("---")
    st.header("Deep Learning Model for Image Classification")
    st.subheader(
        """
        The deep learning model can be used to upload an image in the front end and the pre-trained model will make an inference and send back the result as well as the accuracy of its prediction.
        """
    )
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("---")
        st_lottie(satellite, height=200, key="satellite")
        st.header("About the dataset")
        st.subheader("Satellite Image Classification")
        st.write("""
        Satellite Image Classification dataset has 4 different classes mixed from ggoogle map snapshots and sensors.
        - The dataset is typically labeled
        - Each image is associated with one or more predefined classes or categories. 
        
        The goal of satellite image classification is to develop deep learning models that can automatically analyze and classify these images into their respective classes based on their visual features.
        
        """)
        
    with right_column:
        st.write("---")
        st_lottie(model_architecture, height=200, key="CNN")
        st.header("Model Architecture")
        st.subheader("Convolutional Neural Network")
        st.write("""
        The deep learning used in this project is Convolutional Neural Network (CNN). The layers contains the following : 
        - Conv2D: These are convolution layers that are designed to deal with image data. Each layer will apply filters to the previous layer's output and produce a convolved feature map.
        - MaxPooling2D: These are pooling layers that will reduce the dimensionality of the feature maps, summarizing the presence of features in the previous layer's output. 
        - Flatten: This layer flattens the inputs, converting a multidimensional tensor into a one-dimensional tensor (i.e., a vector).
        - Dropout: These layers randomly set a fraction of input units to 0 at each update during training, which helps prevent overfitting.
        - Dense Layers : These are fully connected layers where each neuron in a layer is connected to every neuron in the previous layer.
        - Final Dense Layer : The final layer contains 4 neurons, and the model is designed and will be trained  to classify inputs into one of four classes.
        
        """)
        
st.write("---")
st.header("Image Recognition App")
st.write("###")
st.write("""
        The application will infer the one label out of 4 labels:
        - Cloudy
        - Dessert
        - Green_Area
        - Water
        
""")


### Block to upload image
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

st.subheader("Upload an image. ")

# Load the model
model = load_model("best_custom_model.h5")

# Define the class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']


# Define the function
st.cache_resource

# Load an image and perform prediction
def predict_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))  # Resize the image
    
    # Convert the image to an array
    img_array = np.array(img)
    
    # Reshape the image array to match the expected input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get the model predictions
    predictions = model.predict(img_array)
    
    # Get the class index with the highest predicted probability
    class_index = np.argmax(predictions[0])
    
    # Get the predicted class label
    predicted_label = class_names[class_index]
    
    return predicted_label

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpeg', 'jpg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=250)
    # st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform prediction
    predicted_label = predict_image(uploaded_file)
    st.subheader("The image is predicted to be '{}'.".format(predicted_label))