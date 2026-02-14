
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib # Import joblib for loading models
import os

st.title('Wine Quality Prediction Dashboard')

# Load the data and models from the Colab environment
# In a real-world scenario, these would be loaded from saved files or a database
# For this demonstration, we assume X_test, y_test, models, and results_df are available from the Colab notebook's global scope

# To make this app runnable independently, we'd need to save/load these objects.
# For now, we'll assume the script is run within the Colab environment after the training cell.

# Example of how to get data if this were run standalone (requires saving X_test, y_test, etc.):
try:
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Load models from .pkl files
    model_dir = os.path.join(script_dir, "model")
    loaded_models = {}
    model_names = ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
    for name in model_names:
        model_filename = os.path.join(model_dir, f"{name.lower()}.pkl")
        if os.path.exists(model_filename):
            loaded_models[name.replace('_', ' ')] = joblib.load(model_filename)
        else:
            st.error(f"Model file not found: {model_filename}. Please ensure models are trained and saved.")
            st.stop()
    models = loaded_models

    # Load X_test, y_test, results_df from saved files
    data_dir = os.path.join(script_dir, "data")
    X_test_path = os.path.join(data_dir, 'X_test.pkl')
    y_test_path = os.path.join(data_dir, 'y_test.pkl')
    results_df_path = os.path.join(data_dir, 'results_df.csv')

    if os.path.exists(X_test_path) and os.path.exists(y_test_path) and os.path.exists(results_df_path):
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)
        results_df = pd.read_csv(results_df_path)
    else:
        st.error(f"Data files not found in '{data_dir}'. Please ensure X_test, y_test, and results_df are saved in the Colab notebook.")
        st.stop()

except Exception as e:
    st.error(f"An error occurred while loading data/models: {e}. Please ensure the training cell in Colab was run and objects were saved if running standalone.")
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

