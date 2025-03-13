import streamlit as st
import kagglehub
import tensorflow as tf
print(tf.__version__)
import numpy as np
from PIL import Image
# import keras

from tensorflow import keras
print(keras.__version__)
import os
from tensorflow.keras.layers import DepthwiseConv2D
# from keras.layers import DepthwiseConv2D
import h5py

st.title("Sports Image Prediction")
st.selectbox("Which model do you want to use?",
             ("sports_classifier_model_trained", "EfficientNetB0"),
)

# Download latest version
path = kagglehub.dataset_download("gpiosenka/sports-classification")
path = os.path.join(path, "EfficientNetB0-100-(224 X 224)- 98.40.h5") 

# f = h5py.File(path, mode="r+")
# model_config_string = f.attrs.get("model_config")
# if model_config_string.find('"groups": 1,') != -1:
#     model_config_string = model_config_string.replace('"groups": 1,', '')
#     f.attrs.modify('model_config', model_config_string)
#     f.flush()
#     model_config_string = f.attrs.get("model_config")
#     assert model_config_string.find('"groups": 1,') == -1

# f.close()

with h5py.File(path, mode="r+") as f:
    model_config_string = f.attrs.get("model_config")
    if model_config_string and '"groups": 1,' in model_config_string:
        model_config_string = model_config_string.replace('"groups": 1,', '')
        f.attrs.modify('model_config', model_config_string)
        f.flush()


# Define F1 score function
def F1_score(y_true, y_pred):
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Cache the model loading
# @st.cache_resource()
# def load_model():
#     return keras.models.load_model(path, custom_objects={"DepthwiseConv2D": DepthwiseConv2D}, compile=False)

# model = load_model()
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model = keras.models.load_model(path, compile=False)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.summary()

# Class labels
class_names = [
    "air hockey", "amputee football", "archery", "arm wrestling", "axe throwing", 
    "balance beam", "barrel racing", "baseball", "basketball", "baton twirling", 
    "bike polo", "billiards", "bmx", "bobsled", "bowling", "boxing", "bull riding", 
    "bungee jumping", "canoe slalom", "cheerleading", "chuckwagon racing", "cricket", 
    "croquet", "curling", "disc golf", "fencing", "field hockey", "figure skating men", 
    "figure skating pairs", "figure skating women", "fly fishing", "football", "formula 1 racing", 
    "frisbee", "gaga", "giant slalom", "golf", "hammer throw", "hang gliding", "harness racing", 
    "high jump", "hockey", "horse jumping", "horse racing", "horseshoe pitching", "hurdles", 
    "hydroplane racing", "ice climbing", "ice yachting", "jai alai", "javelin", "jousting", "judo", 
    "lacrosse", "log rolling", "luge", "motorcycle racing", "mushing", "nascar racing", 
    "olympic wrestling", "parallel bar", "pole climbing", "pole dancing", "pole vault", "polo", 
    "pommel horse", "rings", "rock climbing", "roller derby", "rollerblade racing", "rowing", "rugby", 
    "sailboat racing", "shot put", "shuffleboard", "sidecar racing", "ski jumping", "sky surfing", 
    "skydiving", "snowboarding", "snowmobile racing", "speed skating", "steer wrestling", 
    "sumo wrestling", "surfing", "swimming", "table tennis", "tennis", "track bicycle", "trapeze", 
    "tug of war", "ultimate", "uneven bars", "volleyball", "water cycling", "water polo", 
    "weightlifting", "wheelchair basketball", "wheelchair racing", "wingsuit flying"
]

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file).resize((224, 224))
    # img_array = np.array(image) / 255.0  # Normalize
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    print(prediction)
    predicted_class = np.argmax(prediction)

    # Display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Sport:** {class_names[predicted_class]}")
