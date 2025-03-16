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
* Images were resized to 299×299 pixels to match the input shape required by deep learning models.
* Pixel values were normalized (scaled between 0 and 1) for efficient model training.  
✅ Data Splitting: The dataset was divided into training, validation, and testing sets.
            
---

## 2. ResNet-50 Algorithm
#### What is ResNet-50?
ResNet-50 (Residual Network with 50 layers) is a deep convolutional neural network (CNN) designed to address the vanishing gradient problem in deep networks.

#### Key Features of ResNet-50:
* **Residual Learning:** Uses skip connections to pass information directly, improving gradient flow during training.
* **Pre-trained Weights:** We use a model pre-trained on ImageNet, allowing the model to transfer learned features and adapt to sports classification.
* **Global Average Pooling:** Instead of fully connected layers, we use GlobalAveragePooling2D, reducing overfitting.
            
## 3. Model Development Steps
##### Step 1: Loading Pre-Trained ResNet-50 Model
* Instead of training from scratch, we load the ResNet-50 model pre-trained on ImageNet.
* The model is initialized without the top classification layer (```include_top=False```) to allow customization for our dataset.
* Input shape is set to (299, 299, 3), ensuring compatibility with our dataset.

#### Step 2: Freezing Base Model Layers
* Initially, all layers of ResNet-50 are frozen (i.e., they won't be updated during training).
* This allows the model to use existing learned features from ImageNet and prevents catastrophic forgetting.
#### Step 3: Adding Custom Layers for Classification
* A Global Average Pooling layer is added to convert feature maps into a smaller feature vector.
* A Dropout layer (0.25) is added to reduce overfitting.
* A fully connected Dense layer with 100 output units and a softmax activation function is used to classify the images into 100 different sports categories.
#### Step 4: Compiling the Model
* **Optimizer:** Adam optimizer is used with a learning rate of 0.005.
* **Loss Function:** Categorical Crossentropy is used as it is a multi-class classification task.
* **Metrics:** Accuracy is used to evaluate model performance.
#### Step 5: Model Training
The model is trained for 30 epochs using the training dataset, and performance is validated using the validation dataset.  
* **Early Stopping:** Stops training if validation loss does not improve for 3 consecutive epochs.
* **ReduceLROnPlateau:** Reduces learning rate when the validation loss stops improving, allowing better convergence.
            
## 4. Model Performance Evaluation
#### 4.1 Training and Validation Metrics
After training, we extract:
* Training Accuracy & Loss
* Validation Accuracy & Loss
* These metrics are plotted to visualize the training progress.  
![alt text](https://www.kaggleusercontent.com/kf/227948822/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..jMK2COoxT34LrzlGa6I22A.vZeHhD6EO805aaKvOEe6_MbXZnbAEMJf9f1FqU4meQc9fOuBRyOXqzPgbcEg_tbZMDwDGMr2oEip3yS_q-72LLLLTbS3HQ8juCUhP1kx7aPq-7QlObtxBwGP2GWNj6U_GKPXYuqWNevKgzi8vW9ilxuBfcNMi5EsHNUe2KpLp7-4l9nK78IgvT6yuLFYTSdnJrOdykBcTo8NV81txvL-DCQzEaY-a6ahbId0hSGWiRAXEc17qrGPg95uw8w2mG94Q5JsFUSBMJO1DAfl6tFhQpdXSOkm-08BN4thVnWXrIM6DcV1slUM-3OOpO086kW5NDdBU70t2a3Jl02OYsr9ONyRpa-xifrRELLSMMC_rWZ9xamE4HBg5ILeCJIhOARangpr3hk2MzJAcjeFOpz3H-2dtT8zJxopb0kKadXVmGY0MrYxnb7B2SwGMxhZ5fG3MprPTWjJi6kieDU1SPQbgJdaYa3JKcDWBrwp4sMfrjr5TAdB1MGJScxwaPjUX8YRipHohuP6gS8Yc9lQFazGB9qHrxETqtrYQyyd17McTugb38ptUrnBFdSdw6kVIwRXzbLV0bEtxNgEIWYjFsOPimDzOgSYVGP7-PfUhdMJ8BoNhhpb5Vi-mqEYjTdn_toak3AjM7sGkwXWM04POtF7UbkngoTKh2nTR0lVYxAm1eU.e9_xcQcQMSAIrMroU7BpxQ/__results___files/__results___18_0.png "ResNet-50 Training and Validation Metrics")
#### 4.2 Model Testing on Unseen Data
* The model is evaluated on the test dataset using the evaluate() function.
* The test accuracy provides a measure of real-world performance.
#### 4.3 Generating Classification Report
* The model predictions are compared with actual labels.
* A classification report is generated to analyze performance across different sports categories.
* Confusion matrix can be used to identify misclassifications.
## 5. Model Saving and Deployment
* The trained model is saved in TensorFlow's .keras format (```model.save("ResNet50_trained_model.keras")```).
* This allows for easy loading and deployment in a web application for real-time predictions.

---
            
## 6. Web App Development
The web app was built using Streamlit, which provides an interactive interface for users to upload images and receive sports predictions.

✅ File Uploader: Allows users to upload images for classification.  
✅ Image Preprocessing: Uploaded images are resized to 299×299 pixels before passing to the model.  
✅ Prediction Display: The app shows the predicted sport with the corresponding probability.  
```python
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((299, 299))
    prediction = model.predict(np.expand_dims(np.array(image), axis=0))
    predicted_class = np.argmax(prediction)
    st.write(f"**Predicted Sport:** {{class_names[predicted_class]}}")
```
            
___

```          
Classification Report:
                        precision    recall  f1-score   support

           air hockey       1.00      1.00      1.00         5
      ampute football       1.00      1.00      1.00         5
              archery       1.00      1.00      1.00         5
        arm wrestling       1.00      1.00      1.00         5
         axe throwing       1.00      1.00      1.00         5
         balance beam       0.83      1.00      0.91         5
        barell racing       1.00      1.00      1.00         5
             baseball       0.60      0.60      0.60         5
           basketball       0.80      0.80      0.80         5
       baton twirling       1.00      0.80      0.89         5
            bike polo       1.00      1.00      1.00         5
            billiards       1.00      1.00      1.00         5
                  bmx       1.00      1.00      1.00         5
              bobsled       1.00      0.80      0.89         5
              bowling       1.00      0.80      0.89         5
               boxing       1.00      1.00      1.00         5
          bull riding       1.00      1.00      1.00         5
       bungee jumping       1.00      1.00      1.00         5
         canoe slamon       0.71      1.00      0.83         5
         cheerleading       1.00      0.40      0.57         5
    chuckwagon racing       1.00      1.00      1.00         5
              cricket       0.83      1.00      0.91         5
              croquet       1.00      1.00      1.00         5
              curling       1.00      1.00      1.00         5
            disc golf       1.00      1.00      1.00         5
              fencing       0.83      1.00      0.91         5
         field hockey       1.00      1.00      1.00         5
   figure skating men       1.00      1.00      1.00         5
 figure skating pairs       1.00      0.60      0.75         5
 figure skating women       0.62      1.00      0.77         5
          fly fishing       1.00      1.00      1.00         5
             football       0.50      0.60      0.55         5
     formula 1 racing       1.00      1.00      1.00         5
              frisbee       1.00      0.80      0.89         5
                 gaga       1.00      1.00      1.00         5
         giant slalom       1.00      1.00      1.00         5
                 golf       1.00      0.80      0.89         5
         hammer throw       1.00      1.00      1.00         5
         hang gliding       1.00      1.00      1.00         5
       harness racing       1.00      1.00      1.00         5
            high jump       1.00      1.00      1.00         5
               hockey       0.83      1.00      0.91         5
        horse jumping       1.00      1.00      1.00         5
         horse racing       0.83      1.00      0.91         5
   horseshoe pitching       1.00      0.80      0.89         5
              hurdles       1.00      1.00      1.00         5
    hydroplane racing       1.00      0.80      0.89         5
         ice climbing       1.00      0.80      0.89         5
         ice yachting       1.00      1.00      1.00         5
             jai alai       1.00      1.00      1.00         5
              javelin       0.71      1.00      0.83         5
             jousting       1.00      1.00      1.00         5
                 judo       0.80      0.80      0.80         5
             lacrosse       0.83      1.00      0.91         5
          log rolling       1.00      1.00      1.00         5
                 luge       0.83      1.00      0.91         5
    motorcycle racing       0.83      1.00      0.91         5
              mushing       1.00      1.00      1.00         5
        nascar racing       1.00      1.00      1.00         5
    olympic wrestling       1.00      1.00      1.00         5
         parallel bar       1.00      1.00      1.00         5
        pole climbing       0.83      1.00      0.91         5
         pole dancing       1.00      0.80      0.89         5
           pole vault       0.83      1.00      0.91         5
                 polo       0.83      1.00      0.91         5
         pommel horse       1.00      1.00      1.00         5
                rings       1.00      1.00      1.00         5
        rock climbing       1.00      1.00      1.00         5
         roller derby       1.00      0.60      0.75         5
   rollerblade racing       1.00      1.00      1.00         5
               rowing       1.00      0.80      0.89         5
                rugby       0.56      1.00      0.71         5
      sailboat racing       0.71      1.00      0.83         5
             shot put       1.00      0.60      0.75         5
         shuffleboard       1.00      0.80      0.89         5
       sidecar racing       1.00      0.80      0.89         5
          ski jumping       1.00      1.00      1.00         5
          sky surfing       1.00      0.80      0.89         5
            skydiving       0.83      1.00      0.91         5
        snow boarding       0.62      1.00      0.77         5
    snowmobile racing       1.00      0.80      0.89         5
        speed skating       1.00      1.00      1.00         5
      steer wrestling       1.00      1.00      1.00         5
       sumo wrestling       1.00      1.00      1.00         5
              surfing       1.00      1.00      1.00         5
             swimming       1.00      1.00      1.00         5
         table tennis       1.00      0.80      0.89         5
               tennis       1.00      1.00      1.00         5
        track bicycle       0.80      0.80      0.80         5
              trapeze       1.00      0.40      0.57         5
           tug of war       1.00      1.00      1.00         5
             ultimate       1.00      0.60      0.75         5
          uneven bars       0.83      1.00      0.91         5
           volleyball       1.00      1.00      1.00         5
        water cycling       1.00      1.00      1.00         5
           water polo       1.00      1.00      1.00         5
        weightlifting       1.00      1.00      1.00         5
wheelchair basketball       1.00      1.00      1.00         5
    wheelchair racing       1.00      0.80      0.89         5
      wingsuit flying       1.00      1.00      1.00         5

             accuracy                           0.93       500
            macro avg       0.94      0.93      0.92       500
         weighted avg       0.94      0.93      0.92       500
```
''')

