import streamlit as st
import kagglehub
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
import os
import h5py


st.title("Sports Image Prediction")
st.markdown(f'''#### This prediction's using ResNet50 Model ([Link to Model Training on Kaggle](https://www.kaggle.com/code/prosper0v0/resnet50-model-sports-classification)) ''')
# st.subheader("This prediction's using ResNet50 Model")

# Download latest version
# path = kagglehub.dataset_download("gpiosenka/sports-classification")
# path = os.path.join(path, "EfficientNetB0-100-(224 X 224)- 98.40.h5")
path = "/models/ResNet50_trained_model.keras"
# st.write("path: ", path)


model = keras.models.load_model(path, compile=False)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# st.write(model.summary())

# Class labels
class_names = [
    'Air Hockey', 'Amputee Football', 'Archery', 'Arm Wrestling', 'Axe Throwing',
    'Balance Beam', 'Barrel Racing', 'Baseball', 'Basketball', 'Baton Twirling',
    'Bike Polo', 'Billiards', 'Bmx', 'Bobsled', 'Bowling', 'Boxing', 'Bull Riding',
    'Bungee Jumping', 'Canoe Slalom', 'Cheerleading', 'Chuckwagon Racing', 'Cricket',
    'Croquet', 'Curling', 'Disc Golf', 'Fencing', 'Field Hockey', 'Figure Skating Men',
    'Figure Skating Pairs', 'Figure Skating Women', 'Fly Fishing', 'Football', 'Formula 1 Racing',
    'Frisbee', 'Gaga', 'Giant Slalom', 'Golf', 'Hammer Throw', 'Hang Gliding', 'Harness Racing',
    'High Jump', 'Hockey', 'Horse Jumping', 'Horse Racing', 'Horseshoe Pitching', 'Hurdles',
    'Hydroplane Racing', 'Ice Climbing', 'Ice Yachting', 'Jai Alai', 'Javelin', 'Jousting', 'Judo',
    'Lacrosse', 'Log Rolling', 'Luge', 'Motorcycle Racing', 'Mushing', 'Nascar Racing',
    'Olympic Wrestling', 'Parallel Bar', 'Pole Climbing', 'Pole Dancing', 'Pole Vault', 'Polo',
    'Pommel Horse', 'Rings', 'Rock Climbing', 'Roller Derby', 'Rollerblade Racing', 'Rowing', 'Rugby',
    'Sailboat Racing', 'Shot Put', 'Shuffleboard', 'Sidecar Racing', 'Ski Jumping', 'Sky Surfing',
    'Skydiving', 'Snowboarding', 'Snowmobile Racing', 'Speed Skating', 'Steer Wrestling',
    'Sumo Wrestling', 'Surfing', 'Swimming', 'Table Tennis', 'Tennis', 'Track Bicycle', 'Trapeze',
    'Tug Of War', 'Ultimate', 'Uneven Bars', 'Volleyball', 'Water Cycling', 'Water Polo',
    'Weightlifting', 'Wheelchair Basketball', 'Wheelchair Racing', 'Wingsuit Flying'
]

# File uploader
st.header("Upload an Image")
uploaded_file = st.file_uploader("", type=["jpg", "png"])


if uploaded_file is not None:
    # Process image
    # image = Image.open(uploaded_file).resize((224, 224))
    image = Image.open(uploaded_file).resize((299, 299))

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




