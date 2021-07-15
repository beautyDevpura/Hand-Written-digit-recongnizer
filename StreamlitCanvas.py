import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import fetch_openml
#mnist = fetch_openml('mnist_784')

mnist = pd.read_csv('/Users/bdevpura/Documents/Dojo_DS/DataSet/digit_rec_project2/DigitRec_train.csv')

X = mnist.drop(columns='label')
y = mnist.label
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49)

st.title("Hand Written Digit Recognizer ")
st.markdown("""
Draw on the canvas to predict the Digit !
""")

#Create a canvas component
canvas_result = st_canvas(stroke_color="white",stroke_width=70,height=700,
                          width=700,
                          background_color='black', key='canvas')

#Resize and predict
if canvas_result.image_data is not None:
    draw_digit = cv2.resize(canvas_result.image_data.astype(np.uint8), (28, 28))

    if st.button("Predict"):
        model = pickle.load(open('model.pkl', 'rb'))
        input_gray = cv2.cvtColor(draw_digit, cv2.COLOR_BGR2GRAY)
        input_image = input_gray.flatten()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        input_image = scaler.transform([input_image])
        digit_pred = model.predict(input_image)
        st.title(digit_pred[0])

        
 #resources: https://discuss.streamlit.io/t/drawable-canvas/3671/8 
#https://stackoverflow.com/questions/66372402/conversion-of-dimension-of-streamlit-canvas-image
#https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array/384926#384926
#https://stackoverflow.com/questions/45554008/error-in-python-script-expected-2d-array-got-1d-array-instead


