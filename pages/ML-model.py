import streamlit as st
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
# import sklearn
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# st.write(sklearn.__version__)

# Download latest version of DataFrame
path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
path += "\Sleep_health_and_lifestyle_dataset.csv"


st.title("Sleep Health and Lifestyle Dataset")
# st.write(path)

# uploaded_file = st.file_uploader("Choose a file", type="csv")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write(df)
#     st.write(df.describe())

df = pd.read_csv(path)
df2 = pd.read_csv(path)

st.write("Original Dataset")
st.write(df)


# Preprocessed Data-----------------------------------------------
# Replace NaN (missing values) with "No" for "Sleep Disorder"
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("No")
df2["Sleep Disorder"] = df2["Sleep Disorder"].fillna("No")

# Now we keep all rows and only drop actual missing (NaN) values
# df.dropna(inplace=True)

# Fill missing numbers with median
# df.fillna(df.median(numeric_only=True), inplace=True)  

# Fill missing categorical values with mode
# df.fillna(df.mode().iloc[0], inplace=True)  
#---------------------------------------------------------------

# st.write("Result DataFrame")
# st.write(df)

# 1-Random Forest Classifier ------------------------------------------------------------
st.header("Random Forest Classifier")

# Encode categorical variables
categorical_cols = ["Gender", "Occupation", "BMI Category", "Blood Pressure"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Define features and target variable
# features = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps", "BMI Category", "Blood Pressure", "Occupation", "Gender"]
features = ["Age", "Sleep Duration", "Physical Activity Level", "Stress Level", "Heart Rate", "BMI Category", "Daily Steps"]
target = "Sleep Disorder"

X = df[features]
y = LabelEncoder().fit_transform(df[target])  # Encode "Yes" and "No" as 1 and 0

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)


# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.8f}")
st.write("Classification Report:\n")

# Generate classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame & round values
report_df = pd.DataFrame(report_dict).transpose()
# report_df = report_df.round(2)  # Round to 2 decimal places

# Convert index to string
report_df.index = report_df.index.map(str)

# Remove precision/recall/f1-score columns from "accuracy" row
if "accuracy" in report_df.index:
    report_df.loc["accuracy", ["precision", "recall", "f1-score"]] = None  # Hide irrelevant metrics


# Display in Streamlit
st.table(report_df)

# Feature importance
# feature_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
# st.write("\nFeature Importance:")
# st.write(feature_importance)

# st.write(X_train.columns)


# Ensure input matches all features used in training
st.subheader("Sleep Quality Prediction(RFC)")

#"Age", "Sleep Duration", "Physical Activity Level", "Stress Level", "Heart Rate", "BMI Category", "Daily Steps"

# Collect user input for all required features
age = st.number_input("Age", min_value=1, max_value=100, step=1)
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=1.0, max_value=12.0, step=0.5)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=10000, step=500)
stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)
physical_activity = st.slider("Physical Activity (1-5)", 1, 5, 3)


# Construct full input feature list (must match `X_train.columns`)
new_data = np.array([[sleep_duration, heart_rate, bmi, stress_level, physical_activity, age, daily_steps]])


# Define sleep disorder labels & recommendations based on dataset
sleep_quality_labels = {
    0: ("No Sleep Disorder 😃", 
        "✅ **Great job! Keep maintaining your healthy sleep habits:**\n"
        "- Stick to a consistent sleep schedule ⏰\n"
        "- Exercise regularly 🏃‍♂️\n"
        "- Maintain a balanced diet 🍏"),
    
    1: ("Insomnia 😴", 
        "⚠️ **You might have Insomnia. Here are some tips to improve sleep:**\n"
        "- Stick to a fixed bedtime & wake-up time 🕰️\n"
        "- Reduce caffeine and screen time before bed ☕📵\n"
        "- Try relaxation techniques (meditation, deep breathing) 🧘‍♀️"),
    
    2: ("Sleep Apnea 💤", 
        "🚨 **You might have Sleep Apnea. Consider these steps:**\n"
        "- Consult a doctor for a sleep study 🏥\n"
        "- Avoid sleeping on your back 🚫🛏️\n"
        "- If diagnosed, consider CPAP therapy 😷")
}

# Predict Sleep Disorder
if st.button("Predict Sleep Disorder"):
    prediction = rf_model.predict(new_data)[0]  # Get single prediction
    readable_prediction, suggestion = sleep_quality_labels.get(prediction, ("Unknown", "No recommendation available."))

    # Display result with personalized suggestions
    st.write(f"### 🛌 Predicted Condition: **{readable_prediction}**")
    st.write(f"#### 💡 Recommendation:\n{suggestion}")

st.divider()
#------------------------------------------------------------------------------------------


# 2-XGBoost  ---------------------------------------------------------------------
st.header("XGBoost Classifier")

# Preprocessing
df2["Sleep Disorder"] = df2["Sleep Disorder"].fillna("No")

# Encode categorical variables
categorical_cols = ["Gender", "Occupation", "BMI Category", "Blood Pressure"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col])
    label_encoders[col] = le

# Features & target
features = ["Age", "Sleep Duration", "Physical Activity Level", "Stress Level", "Heart Rate", "BMI Category", "Daily Steps"]
target = "Sleep Disorder"

X = df2[features]
y = LabelEncoder().fit_transform(df2[target])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)

# Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.8f}")

# Classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc["accuracy", ["precision", "recall", "f1-score"]] = None
st.table(report_df)

# Feature importance
# feature_importance = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
# st.write("Feature Importance:")
# st.write(feature_importance)

# Ensure input matches all features used in training
st.subheader("Sleep Quality Prediction(XGBoost)")

#"Age", "Sleep Duration", "Physical Activity Level", "Stress Level", "Heart Rate", "BMI Category", "Daily Steps"

# Collect user input for all required features
age2 = st.number_input("Age", min_value=1, max_value=100, step=1, key="age2")
sleep_duration2 = st.number_input("Sleep Duration (hours)", min_value=1.0, max_value=12.0, step=0.5, key="sleep_duration2")
heart_rate2 = st.number_input("Heart Rate", min_value=40, max_value=200, step=1, key="heart_rate2")
bmi2 = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, key="bmi2")
daily_steps2 = st.number_input("Daily Steps", min_value=0, max_value=10000, step=500, key="daily_steps2")
stress_level2 = st.slider("Stress Level (1-5)", 1, 5, 3, key="stress_level2")
physical_activity2 = st.slider("Physical Activity (1-5)", 1, 5, 3, key="physical_activity2")


# Construct full input feature list (must match `X_train.columns`)
new_data2 = np.array([[sleep_duration2, heart_rate2, bmi2, stress_level2, physical_activity2, age2, daily_steps2]])


# Define sleep disorder labels & recommendations based on dataset
sleep_quality_labels = {
    0: ("No Sleep Disorder 😃", 
        "✅ **Great job! Keep maintaining your healthy sleep habits:**\n"
        "- Stick to a consistent sleep schedule ⏰\n"
        "- Exercise regularly 🏃‍♂️\n"
        "- Maintain a balanced diet 🍏"),
    
    1: ("Insomnia 😴", 
        "⚠️ **You might have Insomnia. Here are some tips to improve sleep:**\n"
        "- Stick to a fixed bedtime & wake-up time 🕰️\n"
        "- Reduce caffeine and screen time before bed ☕📵\n"
        "- Try relaxation techniques (meditation, deep breathing) 🧘‍♀️"),
    
    2: ("Sleep Apnea 💤", 
        "🚨 **You might have Sleep Apnea. Consider these steps:**\n"
        "- Consult a doctor for a sleep study 🏥\n"
        "- Avoid sleeping on your back 🚫🛏️\n"
        "- If diagnosed, consider CPAP therapy 😷")
}

# Predict Sleep Disorder
if st.button("Predict Sleep Disorder", key="predict_button"):
    prediction = rf_model.predict(new_data2)[0]  # Get single prediction
    readable_prediction, suggestion = sleep_quality_labels.get(prediction, ("Unknown", "No recommendation available."))

    # Display result with personalized suggestions
    st.write(f"### 🛌 Predicted Condition: **{readable_prediction}**")
    st.write(f"#### 💡 Recommendation:\n{suggestion}")

#-------------------------------------------------------------------------------------------

