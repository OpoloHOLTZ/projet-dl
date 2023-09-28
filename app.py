import streamlit as st
import tensorflow as tf
import numpy as np
from  tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.ops.gen_array_ops import Size
from  tensorflow.keras.preprocessing.image import  load_img
from  tensorflow.keras.preprocessing.image import  img_to_array
from  tensorflow.keras.applications.vgg16  import  preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

@st.cache_resource
def load_model(path):
  return tf.keras.models.load_model(path)


def predict(path, model_trained, size):
  image = load_img(path, target_size=(size, size, 3))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)
  predictions = model_trained.predict(image)
  class_index = np.argmax(predictions)
  return class_index


st.title("Modèle CNN")
model_path_cnn = "C:\\Users\\opolo\\Documents\\MIA\\DL\\projet-dl\\Models\\cnn.h5"
model_cnn = load_model(model_path_cnn)

uploaded_file_cnn = st.file_uploader("Téléchargez votre fichier pour le modèle CNN ici...", type=['png', 'jpeg', 'jpg'])

if uploaded_file_cnn is not None:
    st.image(uploaded_file_cnn, caption='Image Téléchargée', use_column_width=True)
    cnn_predictions = predict( uploaded_file_cnn, model_cnn, 224)
    st.write("Indice de classe prédit par le modèle CNN :", cnn_predictions)

st.title("Modèle Manual")
model_path_manual = "C:\\Users\\opolo\\Documents\\MIA\\DL\\projet-dl\\Models\\manual.h5"
model_manual = load_model(model_path_manual)

uploaded_file_manual = st.file_uploader("Téléchargez votre fichier pour le modèle Manual ici...", type=['png', 'jpeg', 'jpg'])

if uploaded_file_manual is not None:
    st.image(uploaded_file_manual, caption='Image Téléchargée', use_column_width=True)
    manual_predictions = predict(uploaded_file_manual, model_manual, 150)
    st.write("Indice de classe prédit par le modèle Manual :", manual_predictions)
