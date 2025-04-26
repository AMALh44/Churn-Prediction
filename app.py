import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Title
st.title("Customer Churn Prediction App 🚀")
st.write("Upload customer data and predict whether they are likely to churn.")

# Sidebar inputs
st.sidebar.header('Input Customer Features')

def user_input_features():
    CreditScore = st.sidebar.slider('Credit Score', 300, 900, 600)
    Age = st.sidebar.slider('Age', 18, 100, 35)
    Tenure = st.sidebar.slider('Tenure (Years with bank)', 0, 10, 3)
    Balance = st.sidebar.slider('Balance', 0, 300000, 50000)
    NumOfProducts = st.sidebar.slider('Number of Products', 1, 4, 1)
    HasCrCard = st.sidebar.selectbox('Has Credit Card?', (0, 1))
    IsActiveMember = st.sidebar.selectbox('Is Active Member?', (0, 1))
    EstimatedSalary = st.sidebar.slider('Estimated Salary', 0, 200000, 50000)
    Geography = st.sidebar.selectbox('Geography', ('France', 'Germany', 'Spain'))
    Gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))

    # Manual mapping
    geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_map = {'Female': 0, 'Male': 1}

    data = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Geography': geography_map[Geography],
        'Gender': gender_map[Gender]
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input features:')
st.write(input_df)

# ---- MODEL LOADING PART ----
# Load the trained model and scaler
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)


model = joblib.load('churn_model.pkl')
scaler = joblib.load('churn_scaler.pkl')


# Preprocess the input data
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# ---- PREDICTION OUTPUT ----
st.subheader('Prediction Result:')
churn_result = np.where(prediction == 1, 'Customer will churn ❌', 'Customer will stay ✅')
st.write(f'**{churn_result[0]}**')

st.subheader('Prediction Probability:')
st.write(f"Churn Probability: {prediction_proba[0][1]:.2f}")
st.write(f"Stay Probability: {prediction_proba[0][0]:.2f}")
