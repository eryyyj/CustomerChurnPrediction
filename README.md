# Customer Churn Prediction

A machine learning project that predicts customer churn using
XGBoost, LightGBM, and CatBoost classifiers.

## Project Structure
churn-demo/
├── app.py                  # Streamlit demo app
├── models/                 # Saved trained models
├── CustomerChurnPrediction.ipynb  # Full project notebook
└── README.md

## Dataset
Kaggle — Customer Churn Prediction Dataset

## Models Used
- XGBoost
- LightGBM
- CatBoost

## Key Features Used
- Support Calls
- Contract Length
- Total Spend
- Payment Delay
- Age

## How to Run
# Install dependencies
uv add streamlit joblib numpy pandas scikit-learn lightgbm catboost xgboost

# Run the app
uv run streamlit run app.py

## Results
All 3 models achieved high accuracy after hyperparameter
tuning with Optuna and 5-Fold Stratified Cross Validation.
