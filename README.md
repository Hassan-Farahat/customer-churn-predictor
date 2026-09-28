# 🔮 Customer Churn Risk Predictor

An interactive Machine Learning web app built with Python, Streamlit, and Scikit-Learn to predict customer churn probability and highlight key risk factors in real time.

🚀 **[Live Demo](https://hassan-farahat-customer-churn-predictor-app-pynbef.streamlit.app/)**

---

## ✨ Features

* **Real-time Churn Assessment:** Calculates risk scores dynamically based on tenure, monthly charges, contract type, and tech support.
* **Random Forest ML Model:** On-the-fly model training with cached results for fast predictions.
* **Feature Importance Insights:** Interactive Plotly bar chart showing which factors influence churn risk the most.
* **Automated Business Recommendations:** Provides clear action items based on risk severity levels (High, Moderate, Low).

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (Web App Framework)
* **Scikit-Learn** (Machine Learning Model)
* **Plotly Express** (Interactive Visualizations)
* **Pandas & NumPy** (Data Processing)

---

## 🚀 Quick Start (Local Run)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hassan-Farahat/customer-churn-predictor.git
   cd customer-churn-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
