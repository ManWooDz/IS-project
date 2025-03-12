import streamlit as st

st.title("Model Development Process(Machine Learning)")
# st.header("1.Data Collection")
# st.html(
#     "<p>The dataset used in this project was sourced from Kaggle, specifically the Sleep Health and Lifestyle Dataset. (<a href='https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset'>Link to Dataset</a>) This dataset provides valuable insights into individuals' sleep patterns, health conditions, and lifestyle habits.</p>"
# )

st.markdown(f'''## 1.Data Collection
The dataset used in this project was sourced from Kaggle, specifically the Sleep Health and Lifestyle Dataset. ([Link to Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)) This dataset provides valuable insights into individuals' sleep patterns, health conditions, and lifestyle habits.
#### About the Dataset
##### Dataset Overview:
The Sleep Health and Lifestyle Dataset comprises 400 rows and 13 columns, covering a wide range of variables related to sleep and daily habits. It includes details such as gender, age, occupation, sleep duration, quality of sleep, physical activity level, stress levels, BMI category, blood pressure, heart rate, daily steps, and the presence or absence of sleep disorders.
##### Key Features of the Dataset:
* **Comprehensive Sleep Metrics:** Explore sleep duration, quality, and factors influencing sleep patterns.
* **Lifestyle Factors:** Analyze physical activity levels, stress levels, and BMI categories.
* **Cardiovascular Health:** Examine blood pressure and heart rate measurements.
* **Sleep Disorder Analysis:** Identify the occurrence of sleep disorders such as Insomnia and Sleep Apnea.
##### Dataset Columns:
* **Person ID:** An identifier for each individual.
* **Gender:** The gender of the person (Male/Female).
* **Age:** The age of the person in years.
* **Occupation:** The occupation or profession of the person.
* **Sleep Duration (hours):** The number of hours the person sleeps per day.
* **Quality of Sleep (scale: 1-10):** A subjective rating of the quality of sleep, ranging from 1 to 10.
* **Physical Activity Level (minutes/day):** The number of minutes the person engages in physical activity daily.
* **Stress Level (scale: 1-10):** A subjective rating of the stress level experienced by the person, ranging from 1 to 10.
* **BMI Category:** The BMI category of the person (e.g., Underweight, Normal, Overweight).
* **Blood Pressure (systolic/diastolic):** The blood pressure measurement of the person, indicated as systolic pressure over diastolic pressure.
* **Heart Rate (bpm):** The resting heart rate of the person in beats per minute.
* **Daily Steps:** The number of steps the person takes per day.
* **Sleep Disorder:** The presence or absence of a sleep disorder in the person (None, Insomnia, Sleep Apnea).

##### Details about Sleep Disorder Column:
* **None:** The individual does not exhibit any specific sleep disorder.
* **Insomnia:** The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.
* **Sleep Apnea:** The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.
## 2. Data Preparation
The dataset used in this project is the Sleep Health and Lifestyle Dataset, which contains various features related to sleep quality, health, and lifestyle habits.

#### Steps in Data Preparation:
* Handling Missing Values:
    * The dataset contained missing values in categorical columns (e.g., "Sleep Disorder"). Missing values were replaced with "No" to ensure consistency.
    ```
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("No")
    df2["Sleep Disorder"] = df2["Sleep Disorder"].fillna("No")
    ```
* Feature Selection:
    * Features such as Age, Sleep Duration, Physical Activity Level, Stress Level, Heart Rate, BMI Category, and Daily Steps were selected based on their relevance to sleep quality prediction.
    ```
        features = ["Age", "Sleep Duration", "Physical Activity Level", "Stress Level", "Heart Rate", "BMI Category", "Daily Steps"]
    ```
* Encoding Categorical Variables:
    * Columns like Gender, Occupation, BMI Category, and Blood Pressure were converted into numerical values using Label Encoding for machine learning compatibility.
    ```
        categorical_cols = ["Gender", "Occupation", "BMI Category", "Blood Pressure"]
    ```
* Train-Test Split:
    * The dataset was split into training (80%) and testing (20%) to evaluate model performance.
    ```
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)
    ```
## 3. Algorithm Theory
#### Random Forest Classifier 🌲
Random Forest is an ensemble learning algorithm based on decision trees. It operates using the bagging (Bootstrap Aggregation) technique, where multiple decision trees are trained on different subsets of the data. The final prediction is made based on the majority vote (classification) or average (regression).

##### Key Advantages of Random Forest:
✅ Reduces overfitting by averaging multiple trees.  
✅ Works well with high-dimensional data.  
✅ Provides feature importance ranking.

#### XGBoost Classifier 🚀
XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting algorithm that builds trees sequentially. Each tree corrects the errors of the previous one by focusing more on misclassified samples.

##### Key Advantages of XGBoost:
✅ Faster and more efficient than traditional boosting methods.  
✅ Handles missing values automatically.  
✅ Optimized for high accuracy and performance.

## 4. Model Development Steps
#### Step 1: Load and Preprocess Data
* Read the dataset.
* Handle missing values.
* Encode categorical columns.
* Select relevant features and define the target variable.
#### Step 2: Train-Test Split
* Split the dataset into training and testing sets (e.g., 80% training, 20% testing).
#### Step 3: Train Models
###### Random Forest Training
* Initialize a RandomForestClassifier with parameters:
    ```
    RandomForestClassifier(n_estimators=100, random_state=42)
    ```
* Train the model on the training data.
###### XGBoost Training
* Initialize an XGBClassifier with parameters:
    ```
    XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    ```
* Train the model using the same training data.
#### Step 4: Model Evaluation
* Make predictions using the test data.
* Compute accuracy score and classification report (precision, recall, F1-score).
* Extract feature importance from each model.
#### Step 5: Web App Integration
* The trained models were integrated into a Streamlit Web App.
* Users can input sleep-related factors, and the app will predict their sleep quality using either Random Forest or XGBoost.
* Additional suggestions are provided for users based on their predicted sleep disorder.
''')