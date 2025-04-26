# Churn-Prediction


📝 Project Description
This project focuses on Customer Churn Prediction using an open banking dataset.
We performed Exploratory Data Analysis (EDA) to understand customer behavior and then built a Logistic Regression model to predict whether a customer will stay or leave (churn).

The main goal is to analyze important features influencing churn and build a predictive model that helps businesses retain customers.

📁 Project Structure
bash
Copy
Edit

├── churn_prediction_(EDA)_.ipynb   # Full EDA and Model Training Notebook

├── churn_model.pkl                 # Saved Logistic Regression Model

├── scaler.pkl                      # Saved Scaler (MinMaxScaler)

├── requirements.txt                # Required Python Libraries

├── README.md                       # Project Documentation

├── streamlit_app.py                 # Streamlit Web App Code

🚀 Features
* 📈 Exploratory Data Analysis (EDA)

  * Feature distributions

  * Correlation analysis

  * Insights on churned vs non-churned customers

* 🛠️ Preprocessing

  * Categorical encoding (Label Encoding for Geography and Gender)

  * Feature scaling using MinMaxScaler

* 🤖 Modeling

  * Logistic Regression classifier

  * Model evaluation: Accuracy, Confusion Matrix, Classification Report

* 🎯 Key Features Used for Prediction

  * Credit Score

  * Age

  * Tenure

  * Balance

  * Number of Products

  * Has Credit Card

  * Is Active Member

  * Estimated Salary

  * Geography

  * Gender

* 🌐 Deployment (Streamlit App)

  * Upload customer information

  * Predict churn probability

  * Real-time results

🛠️ Tech Stack

  * Python 🐍

  * Pandas

  * NumPy

  * Matplotlib

  * Seaborn

  * Scikit-Learn

  * Streamlit

📦 Setup Instructions

1. Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Install Requirements

bash
Copy
Edit
pip install -r requirements.txt

3. Run Streamlit App

bash
Copy
Edit
streamlit run streamlit_app.py

📊 Results

* Model Accuracy: ~80% (based on dataset and preprocessing)

* Key Insights:

  * Higher age customers churn more.

  * Non-active members have higher churn rate.

  * Customers with low number of products are more likely to churn.
