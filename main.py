import streamlit as st
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt

# Download latest version
# path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")


st.write("Sleep Health and Lifestyle Dataset")

uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    st.write(df.describe())

