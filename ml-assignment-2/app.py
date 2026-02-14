import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.title("Wine Quality Prediction Dashboard")

# -----------------------------
# Load Models
# -----------------------------
try:
    models = {
        "Logistic Regression": joblib.load("model/logisticregression.pkl"),
        "Decision Tree": joblib.load("model/decisiontree.pkl"),
        "kNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naivebayes.pkl"),
        "Random Forest": joblib.load("model/randomforest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -----------------------------
# Select Mode
# -----------------------------
st.header("Select Evaluation Mode")

mode = st.radio(
    "Choose Data Source:",
    ("Use Existing Test Files", "Upload New Test Dataset")
)

# ======================================================
# MODE 1: EXISTING SAVED TEST FILES
# ======================================================
if mode == "Use Existing Test Files":

    try:
        X_test = pd.read_csv("X_test.csv").values
        y_test = pd.read_csv("y_test.csv").values.ravel()
        results_df = pd.read_csv("results_df.csv")

        st.subheader("Overall Model Performance")
        st.dataframe(results_df)

        selected_model_name = st.selectbox(
            "Select Model for Detailed Analysis",
            list(models.keys())
        )

        if st.button("Run Detailed Analysis"):
            selected_model = models[selected_model_name]
            y_pred = selected_model.predict(X_test)

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.json(report)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix - {selected_model_name}")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Saved test files not found: {e}")
        st.info("Please upload test dataset instead.")

# ======================================================
# MODE 2: UPLOAD NEW TEST DATASET
# ======================================================
elif mode == "Upload New Test Dataset":

    uploaded_file = st.file_uploader("Upload CSV file (Test Data Only)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Preview of Uploaded Data:")
        st.dataframe(data.head())

        target_column = st.selectbox("Select Target Column", data.columns)

        if target_column:
            X_test = data.drop(columns=[target_column])
            y_test = data[target_column]

            selected_model_name = st.selectbox(
                "Select Model for Prediction",
                list(models.keys())
            )

            if st.button("Run Prediction"):
                selected_model = models[selected_model_name]
                y_pred = selected_model.predict(X_test)

                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.json(report)

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix - {selected_model_name}")

                st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to proceed.")
