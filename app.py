import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_model.pkl")

st.title("Employee Salary Prediction")
st.write("Predict employee salary using Machine Learning")

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=30,
    value=2
)

education = st.selectbox(
    "Education",
    ["Intermediate", "BTech", "MTech"]
)

job_role = st.selectbox(
    "Job Role",
    [
        "Data Analyst",
        "ML Engineer",
        "Software Developer",
        "Data Scientist",
        "Web Developer",
        "Business Analyst"
    ]
)

location = st.selectbox(
    "Location",
    [
        "Hyderabad",
        "Bangalore",
        "Chennai",
        "Pune"
    ]
)

if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        "experience": [experience],
        "education": [education],
        "job_role": [job_role],
        "location": [location]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ₹{prediction:.2f} LPA")