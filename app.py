import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, f1_score, matthews_corrcoef
import os

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Wine Quality Prediction App",
    page_icon="🍷",
    layout="wide"
)

# Define paths for data and model directories
DATA_DIR = "data"
MODEL_DIR = "model"

# Load results_df, X_test, and y_test
@st.cache_data
def load_data():
    results_df = pd.read_csv(os.path.join(DATA_DIR, 'results_df.csv'))
    X_test = joblib.load(os.path.join(DATA_DIR, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(DATA_DIR, 'y_test.pkl'))
    return results_df, X_test, y_test

results_df, X_test, y_test = load_data()

# Function to load all models
@st.cache_resource
def load_models():
    models = {}
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pkl"):
            model_name = filename.replace(".pkl", "").replace("_", " ").title()
            model_path = os.path.join(MODEL_DIR, filename)
            models[model_name] = joblib.load(model_path)
    return models

loaded_models = load_models()

# Streamlit Title and Introduction
st.title("🍷 Wine Quality Prediction Model Performance")
st.markdown(
    "This application evaluates the performance of various machine learning models "
    "trained to predict wine quality (binary classification: quality >= 6 or < 6). "
    "Below you can see the overall performance of all models and then select a specific "
    "model for a detailed analysis."
)

st.subheader("Overall Model Performance")
st.dataframe(results_df, width=1000) # Changed use_container_width to width

# Sidebar for model selection
st.sidebar.header("Select a Model for Detailed Analysis")
selected_model_name = st.sidebar.selectbox(
    "Choose a model:", list(loaded_models.keys())
)

if selected_model_name:
    st.subheader(f"Detailed Analysis for {selected_model_name}")
    model = loaded_models[selected_model_name]

    # Make predictions and probability predictions
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred # For models without predict_proba (e.g., some SVMs)

    # Display performance metrics
    st.markdown("#### Performance Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    col5.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
    if hasattr(model, "predict_proba"):
        col6.metric("AUC", f"{auc(roc_curve(y_test, y_prob)[0], roc_curve(y_test, y_prob)[1]):.4f}")
    else:
        col6.metric("AUC", "N/A")

    # Classification Report
    st.markdown("#### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title(f'Confusion Matrix for {selected_model_name}')
    ax_cm.set_ylabel('True label')
    ax_cm.set_xlabel('Predicted label')
    tick_marks = np.arange(2)
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.text(0, 0, cm[0, 0], ha="center", va="center", color="red")
    ax_cm.text(0, 1, cm[0, 1], ha="center", va="center", color="red")
    ax_cm.text(1, 0, cm[1, 0], ha="center", va="center", color="red")
    ax_cm.text(1, 1, cm[1, 1], ha="center", va="center", color="red")
    plt.tight_layout()
    st.pyplot(fig_cm)

    # ROC Curve
    if hasattr(model, "predict_proba"):
        st.markdown("#### ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'Receiver Operating Characteristic (ROC) for {selected_model_name}')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.markdown("#### ROC Curve")
        st.info("ROC Curve cannot be plotted for this model as it does not have 'predict_proba' method.")


st.sidebar.markdown("----")
st.sidebar.info("**How to run:**\n1. Save this code as `app.py`\n2. Make sure you have `model` and `data` directories in the same location as `app.py`\n3. Run `streamlit run app.py` in your terminal.")
