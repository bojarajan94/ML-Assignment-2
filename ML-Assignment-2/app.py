
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

st.title('Wine Quality Prediction Dashboard')

# Load the data and models from saved files
try:
    X_test = pd.read_csv('model/X_test.csv').values
    y_test = pd.read_csv('model/y_test.csv')['label']
    results_df = pd.read_csv('model/results_df.csv')

    # Load models
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "kNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }

except FileNotFoundError:
    st.error("Required model or data files not found. Please ensure the Colab notebook was run to generate these files.")
    st.stop()

# Display overall performance
st.header("Overall Model Performance")
st.dataframe(results_df)

# Model selection
selected_model_name = st.selectbox('Select a model to view detailed analysis:', list(models.keys()))

if selected_model_name:
    selected_model = models[selected_model_name]

    st.subheader(f'Detailed Analysis for {selected_model_name}')

    # Make predictions
    y_pred = selected_model.predict(X_test)

    # Classification Report
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix for {selected_model_name}')
    st.pyplot(fig)

