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
path = "models/ResNet50_trained_model.keras"
# st.write("path: ", path)



# with h5py.File(path, mode="r+") as f:
#     model_config_string = f.attrs.get("model_config")
#     if model_config_string and '"groups": 1,' in model_config_string:
#         model_config_string = model_config_string.replace('"groups": 1,', '')
#         f.attrs.modify('model_config', model_config_string)
#         f.flush()



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
