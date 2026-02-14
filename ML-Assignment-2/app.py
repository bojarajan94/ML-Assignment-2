
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Wine Quality Prediction Dashboard')

# Load the data and models from the Colab environment
# In a real-world scenario, these would be loaded from saved files or a database
# For this demonstration, we assume X_test, y_test, models, and results_df are available from the Colab notebook's global scope

# To make this app runnable independently, we'd need to save/load these objects.
# For now, we'll assume the script is run within the Colab environment after the training cell.

# Placeholder for demonstration if running directly outside Colab after saving state
# In a complete deployment, you'd load models and data from persistent storage.

# Example of how to get data if this were run standalone (requires saving X_test, y_test, etc.):
# try:
#     X_test = pd.read_csv('X_test.csv')
#     y_test = pd.read_csv('y_test.csv')['label']
#     # Load models from .pkl files if they were saved
#     import joblib
#     models = {
#         "Logistic Regression": joblib.load("model/logreg.pkl"),
#         "Decision Tree": joblib.load("model/dtree.pkl"),
#         # ... load other models
#     }
#     results_df = pd.read_csv('results_df.csv')
# except FileNotFoundError:
#     st.error("Please run the model training cell in Colab first to generate data and models, or ensure necessary files are saved.")
#     st.stop()

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

