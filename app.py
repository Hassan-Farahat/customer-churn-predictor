import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Page setup
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="wide")

# Train ML model on cached synthetic dataset
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000

    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n)
    contract_one_year = np.random.choice([0, 1], n, p=[0.7, 0.3])
    tech_support = np.random.choice([0, 1], n, p=[0.6, 0.4])

    # Rule: Short tenure (< 12 mos), high bill (> $70), and NO annual contract = Churn (1)
    churn = np.where((tenure < 12) & (monthly_charges > 70) & (contract_one_year == 0), 1, 0)

    X = pd.DataFrame({
        'Tenure (Months)': tenure,
        'Monthly Charges ($)': monthly_charges,
        'One-Year Contract': contract_one_year,
        'Tech Support': tech_support
    })


    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, churn)
    return model

model = train_model()

# Header
st.title("🔮 Customer Churn Risk Predictor")
st.markdown("Predict customer retention risk in real-time using Machine Learning.")

# Sidebar - Customer Input
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One-Year / Two-Year"])
tech_support = st.sidebar.selectbox("Tech Support Included", ["No", "Yes"])


# Format inputs
contract_val = 1 if contract == "One-Year / Two-Year" else 0
tech_val = 1 if tech_support == "Yes" else 0

input_data = pd.DataFrame({
    'Tenure (Months)': [tenure],
    'Monthly Charges ($)': [monthly_charges],
    'One-Year Contract': [contract_val],
    'Tech Support': [tech_val]
})

# Run prediction
churn_proba = model.predict_proba(input_data)[0][1] * 100


# Results view
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Churn Probability")
    st.metric("Risk Score", f"{churn_proba:.1f}%")
    
    if churn_proba >= 60:
        st.error("🚨 High Churn Risk! Consider offering a loyalty discount.")
    elif churn_proba >= 35:
        st.warning("⚠️ Moderate Churn Risk. Monitor account activity.")
    else:
        st.success("✅ Low Churn Risk. Customer is stable.")

with col2:
    st.subheader("Model Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', template='plotly_white')
    st.plotly_chart(fig_imp, use_container_width=True)