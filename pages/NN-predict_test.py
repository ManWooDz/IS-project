import streamlit as st
import kagglehub
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import h5py
from tensorflow import keras
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

st.title("Sports Image Prediction")

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

# Model selection dropdown
model_option = st.selectbox(
    "Which model do you want to use?",
    ("EfficientNetB0", "ResNet50"),
)

# Paths to the models
effnet_path = kagglehub.dataset_download("gpiosenka/sports-classification")
effnet_path = os.path.join(effnet_path, "EfficientNetB0-100-(224 X 224)- 98.40.h5")
# local_model_path = "../models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5" # Replace with actual local path
# base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
# local_model_path = os.path.join(base_dir, "../models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Normalize path for different OS
# local_model_path = os.path.normpath(local_model_path)

# Modify EfficientNetB0 file if needed
with h5py.File(effnet_path, mode="r+") as f:
    model_config_string = f.attrs.get("model_config")
    if model_config_string and '"groups": 1,' in model_config_string:
        model_config_string = model_config_string.replace('"groups": 1,', '')
        f.attrs.modify('model_config', model_config_string)
        f.flush()

# Load the selected model
@st.cache_resource()  # Cache the model so it doesn't reload every time
def load_model(model_choice):
    if model_choice == "EfficientNetB0":
        return keras.models.load_model(effnet_path, compile=False)
    else:  # Load sports_classifier_model_trained
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(len(class_names), activation="softmax")(x)  # Adjust output to match class count
        return Model(inputs=base_model.input, outputs=x)

# Load model dynamically
model = load_model(model_option)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])



# File uploader
st.header("Upload an Image")
uploaded_file = st.file_uploader("", type=["jpg", "png"])

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
    st.image(image, caption="Uploaded Image", use_container_width = True)
    st.write(f"**Predicted Sport:** {class_names[predicted_class]}")
    
    accuracy = model.evaluate(img_array, np.array([predicted_class]))  # Using the predicted class as true label
    st.write(f"**Model Accuracy (on uploaded image):** {accuracy[1] * 100:.2f}%")

    

st.divider()

@st.dialog("Class Labels Used by the Model")
def opened_dialog():
    st.write("The model uses the following class labels:")
    for class_name in class_names:
        st.write(class_name)

if st.button("Show all sports classes"):
    opened_dialog()