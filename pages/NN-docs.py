import streamlit as st

st.title("Model Development Process(Neural Network)")
# st.header("1.Data Collection")
# st.html(
#     "<p>The dataset used in this project was sourced from Kaggle, specifically the Sleep Health and Lifestyle Dataset. (<a href='https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset'>Link to Dataset</a>) This dataset provides valuable insights into individuals' sleep patterns, health conditions, and lifestyle habits.</p>"
# )

st.markdown(f'''## 1. Data Preparation
The model was trained using the Sports Classification Dataset from Kaggle, which consists of images labeled according to different sports. The key steps in data preparation included:  
#### **Dataset Collection**: 
The dataset used for this Neural Network model was sourced from Kaggle, specifically the Sports Classification Dataset. ([Link to Dataset](https://www.kaggle.com/code/alnourabdalrahman9/sports-classification)) This dataset contains a diverse collection of sports images, categorized into 100 different sports classes. The dataset was chosen for its high-quality labeled images, making it suitable for training a deep learning model for sports image classification.  
#### **Data Cleaning & Preprocessing**:  
* Images were resized to 224×224 pixels to match the input shape required by deep learning models.
* Pixel values were normalized (scaled between 0 and 1) for efficient model training.  
✅ Data Splitting: The dataset was divided into training, validation, and testing sets.
## 2. Theory Behind the Algorithm
The EfficientNetB0 deep learning model was used for image classification. This is a Convolutional Neural Network (CNN) that efficiently balances depth, width, and resolution for improved accuracy and computational efficiency.

#### CNN Architecture
* Convolutional Layers extract features (edges, textures, patterns) from images.
* Pooling Layers reduce dimensionality, making computation efficient.
* Fully Connected Layers map extracted features to different sports categories.
* Softmax Activation assigns probabilities to each class, enabling classification.
            
## 3. Model Development Steps
#### Step 1: Model Selection
We chose EfficientNetB0 due to its high accuracy (98.4%) and low computational cost compared to other CNN architectures like ResNet and VGG.

#### Step 2: Model Compilation
The model was loaded and compiled with:  
🔹 Optimizer: Adam (adaptive learning rate optimization).  
🔹 Loss Function: Sparse Categorical Crossentropy (for multi-class classification).  
🔹 Metrics: Accuracy (to measure prediction correctness).  
```python
    model = keras.models.load_model(path, compile=False)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

```
#### Step 3: Model Evaluation
To assess performance, the model was evaluated on a test dataset using:  
✅ Accuracy Calculation:
```python
    loss, accuracy = model.evaluate(x_test_resized, y_test, verbose=0)


```
✅ Classification Report: Precision, Recall, and F1-score were computed using Scikit-learn.
```python
    from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)


```
#### Step 4: Training & Performance Visualization
The model was trained for 10 epochs, and accuracy & loss graphs were plotted to analyze learning trends:
```python
    # Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)

# Plot Accuracy & Loss Graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['loss'], label='Train Loss')


```
## 4. Web App Development
The web app was built using Streamlit, which provides an interactive interface for users to upload images and receive sports predictions.

✅ File Uploader: Allows users to upload images for classification.  
✅ Image Preprocessing: Uploaded images are resized to 224×224 pixels before passing to the model.  
✅ Prediction Display: The app shows the predicted sport with the corresponding probability.  
```python
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    prediction = model.predict(np.expand_dims(np.array(image), axis=0))
    predicted_class = np.argmax(prediction)
    st.write(f"**Predicted Sport:** {{class_names[predicted_class]}}")
```
''')