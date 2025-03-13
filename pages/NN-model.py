import streamlit as st
import kagglehub
import pandas as pd
import numpy as np
import tensorflow as tf
import os


# Download latest version
path = kagglehub.dataset_download("gpiosenka/sports-classification")
path += "\\"

st.title("Full TMDB Movies Dataset")
# df = pd.read_csv(path)
st.write(path)
# st.write(os.listdir(path))

# Define dataset paths (Update these paths as necessary)
# data_dir = "path_to_dataset"
data_dir = path
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

# Set parameters
batch_size = 32
img_height = 224
img_width = 224

# Load training and validation datasets
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load test dataset
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_data.class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_data)
st.write(f"Test Accuracy: {test_acc:.2f}")

# Make predictions
predictions = model.predict(test_data)
st.write("Predictions generated.")