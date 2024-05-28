# Thyroid_Disease_Detection
Thyroid disease is a very common problem in India, more than one crore people are suffering with the disease every year. Thyroid disorder can speed up or slow down the metabolism of the body.

The main objective of this project is to predict if a person is having compensated hypothyroid, primary hypothyroid, secondary hypothyroid or negative (no thyroid) with the help of Machine Learning. Classification algorithms such as Random Forest, XGBoost and KNN Model have been trained on the thyroid dataset, UCI Machine Learning repository. After hyperparameter tuning XGBoost model has performed well with better accuracy, precision and recall. Application has deployed on Heroku with the help of flask framework.
# Demo
https://github.com/pavitra147/Thyroid-Disease-Detection/assets/130755029/1d78ea71-96e3-4e11-a746-2a8606b23c90

# Technical Aspects
Python 3.7 and more
Important Libraries: sklearn, pandas, numpy, matplotlib & seaborn
Front-end: HTML, CSS
Back-end: Flask framework
IDE: Jupyter Notebook & VSCode
# Workflow
Thyroid Disease Data Set from UCI Machine Learning Repository.

Link:https://archive.ics.uci.edu/ml/datasets/thyroid+disease
# Data Pre-processing
Missing values handling by Simple imputation (KNN Imputer)
Outliers detection and removal by boxplot and percentile methods
Categorical features handling by ordinal encoding and label encoding
Feature scaling done by Standard Scalar method
Imbalanced dataset handled by SMOTE
Drop unnecessary columns
# Model Creation and Evaluation
Various classification algorithms like Random Forest, XGBoost, KNN etc tested.
Random Forest, XGBoost and KNN were all performed well. XGBoost was chosen for the final model training and testing.
Hyper parameter tuning was performed using RandomizedSearchCV
Model performance evaluated based on accuracy, confusion matrix, classification report.
# Author
Pavitra Mishra:https://www.linkedin.com/in/pavitra-mishra-483183251/



# Help Me Improve
Hello Reader if you find any bug please consider raising issue I will address them asap.
