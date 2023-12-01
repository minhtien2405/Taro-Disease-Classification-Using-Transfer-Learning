import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from util import set_background, classify
import pandas as pd


# set page config
st.set_page_config(
    page_title="Disease Taro Classification",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# set page title
st.title("Disease Taro Classification")

# set page header
st.header("Upload an image of a Taro leaf to classify the disease")

# set sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This is a simple web app to classify disease Taro tuber. 
    It is built using Tensorflow, Keras and Streamlit.
    """
)
st.sidebar.title("Contact")
st.sidebar.info(
    """
    This web app is created by [Pham Minh Tien](https://www.linkedin.com/in/minhtien2405/), Tran Nhu Hieu and Nhat Linh.
    """
)

st.sidebar.title("Source Code")
st.sidebar.info(
    """
    The source code is available in this [Github repo](https://github.com/minhtien2405/Taro-Disease-Classification).
    """
)

# set background
# set_background('background.png')

# upload file
file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg"])


# load model
model = load_model('transfer_learning1.h5', compile = False)

# load class names
with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split()[1] for a in f.readlines()]

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width = True)


# classify image
# create a button
if st.button("Classify"):
    if file is None:
        st.write("Please upload an image file")
    else:
        class_name, cf_healthy_score, cf_unhealthy_score = classify(image, model, class_names)
        # show results
        st.write("Class name: ",class_name)
        # show confidence score
        st.write("Confidence score of healthy taro: ",round(cf_healthy_score*100, 2) ,"%")
        st.write("Confidence score of unhealthy taro: ",round(cf_unhealthy_score*100, 2),"%")
        # show bar chart
        bar_df = pd.DataFrame({cf_healthy_score, cf_unhealthy_score}, index = ['Healthy', 'Unhealthy'])
        st.bar_chart(bar_df)
