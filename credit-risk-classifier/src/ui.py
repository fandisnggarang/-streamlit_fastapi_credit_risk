from fastapi import FastAPI
import streamlit as st 
import requests

# Tentukan URL API
API_URL = "http://localhost:8000/predict"

st.title('Credit Risk Classifier App')
st.markdown('**Created By Fandis Nggarang | Batch I Juli 2023**')
st.divider()
st.markdown('Just type the value. The model will define your status')

# Buat kolom input untuk pengguna memasukkan data mereka
person_age = st.number_input("Person Age (years)", min_value=0, max_value=100)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Employment Length", min_value=0.0)
person_income = st.number_input("Annual Income (USD)", min_value=0.0)
loan_amnt = st.number_input("Loan Amount (USD)", min_value=0.0)

# Hitung loan_percent_income secara otomatis
loan_percent_income = 0.0
if person_income > 0:
    loan_percent_income = loan_amnt / person_income
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=loan_percent_income, step=0.01, format="%.2f")

loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0)
cb_person_default_on_file = st.selectbox("Ever been defaulted?", ["Y", "N"])

# Simpan data input dalam dictionary
input_data = {
    "person_age": person_age,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": person_emp_length,
    "person_income": person_income,
    "loan_amnt": loan_amnt,
    "loan_percent_income": loan_percent_income,
    "loan_int_rate": loan_int_rate,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "cb_person_default_on_file": cb_person_default_on_file
}

# Tahapan yang terjadi setelah tombol "Predict" diklik
if st.button("predict"):
    with st.spinner('Wait for it...'):
        # Kirim permintaan ke endpoint FastAPI
        response = requests.post(API_URL, json=input_data)
        
        # Dapatkan hasil prediksi
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['Prediction']}, Probability: {result['Probability']:.2f}")
        else:
            st.error("Error in prediction request")