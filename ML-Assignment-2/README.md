\# Machine Learning Assignment 2 – Classification Model Deployment



\## Problem Statement



The objective of this project is to build and compare multiple machine learning classification models to predict wine quality using the selected dataset. The goal is to evaluate model performance using various evaluation metrics and deploy the best performing model using Streamlit.



\## Dataset Description



\- \*\*Dataset Name\*\*: Wine Quality Red Dataset

\- \*\*Source\*\*: UCI Machine Learning Repository

\- \*\*Number of Instances\*\*: 1599

\- \*\*Number of Features\*\*: 11

\- \*\*Target Variable\*\*: label (derived from 'quality', indicating if quality >= 6)

\- \*\*Problem Type\*\*: Binary Classification





\## Machine Learning Models Implemented



1\. Logistic Regression

2\. Decision Tree Classifier

3\. K-Nearest Neighbors (KNN)

4\. Naive Bayes (Gaussian / Multinomial)

5\. Random Forest (Ensemble)

6\. XGBoost (Ensemble)



\## Evaluation Metrics



\- Accuracy

\- AUC Score

\- Precision

\- Recall

\- F1 Score

\- Matthews Correlation Coefficient (MCC)



&nbsp;             Model  Accuracy       AUC  Precision    Recall        F1         MCC

Logistic Regression  0.740625  0.819050   0.785714  0.737430  0.760807    0.479299

&nbsp;     Decision Tree  0.715625  0.710131   0.765060  0.709497  0.736232    0.430141

&nbsp;               kNN  0.706250  0.773743   0.720207  0.776536  0.747312    0.399359

&nbsp;       Naive Bayes  0.734375  0.792702   0.758242  0.770950  0.764543    0.460015

&nbsp;     Random Forest  0.784375  0.891874   0.805556  0.810056  0.807799    0.562263

&nbsp;           XGBoost  0.812500  0.878719   0.836158  0.826816  0.831461    0.620258





\### Observations on Model Performance



Based on the performance metrics, here are some observations:



XGBoost appears to be the best-performing model overall, with the highest Accuracy (0.8125), AUC (0.8787), Precision (0.8362), Recall (0.8268), F1-score (0.8315), and MCC (0.6203).



Random Forest is also a strong performer, coming in second across most metrics, especially for AUC (0.8919) which is slightly higher than XGBoost, suggesting good discrimination capability.



Logistic Regression and Naive Bayes show moderate performance, with accuracy around 0.73-0.74.



kNN and Decision Tree are the lower-performing models in this comparison, particularly the Decision Tree having the lowest AUC and MCC, indicating it might be struggling with the dataset or could benefit from more tuning.





Random Forest showed the best overall performance with highest accuracy and MCC,

indicating strong generalization capability and reduced overfitting.





\## Streamlit Application Features



\- CSV test dataset upload

\- Model selection dropdown

\- Display of evaluation metrics

\- Confusion matrix visualization







\## How to Run Locally



1\. Clone the repository

2\. Install dependencies:

   pip install -r requirements.txt

3\. Run:

   streamlit run app.py

