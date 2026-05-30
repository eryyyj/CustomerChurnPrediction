import streamlit as st
import joblib
import numpy as np
import pandas as pd

xgb_model  = joblib.load('models/xgboost_model.pkl')
lgbm_model = joblib.load('models/lightgbm_model.pkl')
cat_model  = joblib.load('models/catboost_model.pkl')

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon=":)",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.markdown("Enter customer details below to predict churn probability.")
st.divider()

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age            = st.number_input("Age",            min_value=18,  max_value=100, value=35)
    support_calls  = st.number_input("Support Calls",  min_value=0,   max_value=20,  value=2)
    payment_delay  = st.number_input("Payment Delay (days)", min_value=0, max_value=60, value=5)

with col2:
    total_spend    = st.number_input("Total Spend ($)", min_value=0.0, max_value=5000.0, value=500.0)
    contract_length = st.selectbox("Contract Length",
                                   options=["Monthly", "Quarterly", "Annual"])

st.divider()

st.subheader("Select Model")
model_choice = st.radio(
    "Choose which model to use for prediction:",
    options=["XGBoost", "LightGBM", "CatBoost", "Compare All 3"],
    horizontal=True
)

def prepare_input(age, support_calls, payment_delay,
                  total_spend, contract_length):

    contract_map   = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
    contract_enc   = contract_map[contract_length]

    risk_score       = support_calls * payment_delay
    high_caller_flag = 1 if support_calls >= 4 else 0
    low_spender_flag = 1 if total_spend < 500 else 0

    features = pd.DataFrame([[
        age, support_calls, payment_delay,
        contract_enc, total_spend,
        risk_score, high_caller_flag, low_spender_flag
    ]], columns=[
        'Age', 'Support Calls', 'Payment Delay',
        'Contract Length', 'Total Spend',
        'risk_score', 'high_caller_flag', 'low_spender_flag'
    ])

    return features

def predict(model, features):
    pred    = model.predict(features)[0]
    prob    = model.predict_proba(features)[0]
    return pred, prob

if st.button("Predict Churn", use_container_width=True, type="primary"):

    features = prepare_input(
        age, support_calls, payment_delay,
        total_spend, contract_length
    )

    models = {
        "XGBoost" : xgb_model,
        "LightGBM": lgbm_model,
        "CatBoost": cat_model
    }

    st.divider()
    st.subheader("Prediction Results")

    if model_choice != "Compare All 3":
        model        = models[model_choice]
        pred, prob   = predict(model, features)
        churn_prob   = prob[1] * 100
        no_churn_prob = prob[0] * 100

        if pred == 1:
            st.error(f"**Customer is likely to CHURN**")
        else:
            st.success(f"**Customer is likely to STAY**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Probability",    f"{churn_prob:.1f}%")
        with col2:
            st.metric("Retention Probability", f"{no_churn_prob:.1f}%")

        st.progress(int(churn_prob))

    else:
        results = []
        for name, model in models.items():
            pred, prob = predict(model, features)
            results.append({
                "Model"               : name,
                "Prediction"          : "Churn" if pred == 1 else "Stay",
                "Churn Probability"   : f"{prob[1]*100:.1f}%",
                "Retention Probability": f"{prob[0]*100:.1f}%"
            })

        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )

        preds = [predict(m, features)[0] for m in models.values()]
        if all(p == 1 for p in preds):
            st.error("All 3 models agree — Customer will CHURN")
        elif all(p == 0 for p in preds):
            st.success("All 3 models agree — Customer will STAY")
        else:
            st.warning("Models disagree — Manual review recommended")

    st.divider()
    with st.expander("View Input Summary"):
        st.dataframe(features, use_container_width=True, hide_index=True)