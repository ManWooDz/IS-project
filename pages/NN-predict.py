import streamlit as st
import kagglehub
import tensorflow as tf
# print(tf.__version__)
import numpy as np
from PIL import Image
# import keras

from tensorflow import keras
# print(keras.__version__)
import os
from tensorflow.keras.layers import DepthwiseConv2D
# from keras.layers import DepthwiseConv2D
import h5py

st.title("Sports Image Prediction")
st.subheader("Using EfficientNetB0 Model")

# Download latest version
path = kagglehub.dataset_download("gpiosenka/sports-classification")
path = os.path.join(path, "EfficientNetB0-100-(224 X 224)- 98.40.h5") 


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


model = keras.models.load_model(path, compile=False)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# st.write(model.summary())

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
    st.image(image, caption="Uploaded Image", use_container_width = True)
    st.write(f"**Predicted Sport:** {class_names[predicted_class]}")


# # Load dataset for evaluation (replace with actual dataset)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# # Resize test images to match model input size (224x224)
# x_test_resized = tf.image.resize(x_test, (224, 224)).numpy()
# x_test_resized = x_test_resized / 255.0  # Normalize

# # Evaluate model accuracy
# loss, accuracy = model.evaluate(x_test_resized, y_test, verbose=0)

# # Display accuracy in Streamlit
# st.write(f"**Model Accuracy:** {accuracy:.2%}")


# from sklearn.metrics import classification_report

# # Make predictions on test set
# y_pred_probs = model.predict(x_test_resized)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # Generate classification report
# report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

# # Display in Streamlit
# st.subheader("Classification Report")
# st.text(report)



# import matplotlib.pyplot as plt

# # Load model training history (modify if needed)
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)

# # Plot Accuracy & Loss Graphs
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Accuracy Plot
# axes[0].plot(history.history['accuracy'], label='Train Accuracy')
# axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
# axes[0].set_title('Model Accuracy')
# axes[0].set_xlabel('Epochs')
# axes[0].set_ylabel('Accuracy')
# axes[0].legend()

# # Loss Plot
# axes[1].plot(history.history['loss'], label='Train Loss')
# axes[1].plot(history.history['val_loss'], label='Validation Loss')
# axes[1].set_title('Model Loss')
# axes[1].set_xlabel('Epochs')
# axes[1].set_ylabel('Loss')
# axes[1].legend()

# # Display in Streamlit
# st.pyplot(fig)
