import base64
import os

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

def set_background(image_file):
    with open(image_file, "rb") as f:
        img = f.read()
    b64_encoded = base64.b64encode(img).decode()
    style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_encoded}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):

    # resize image
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = img_array.astype('float32')
    img_array = 1./255 * img_array
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    print(img_array.shape)
    predictions = model.predict(img_array)
    print(predictions)
    index = np.argmax(predictions)
    class_name = class_names[index]
    cf_healthy_score, cf_unhealthy_score = predictions[0]
    
    return class_name, cf_healthy_score, cf_unhealthy_score

    
    
    
