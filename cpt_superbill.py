import streamlit as st
import mysql.connector
import datetime

DB_CONFIG = {
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "host": st.secrets["DB_HOST"],
    "database": st.secrets["DB_NAME"],
    "autocommit": False
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

st.title("CPT Data Entry App")

form_type = st.sidebar.selectbox("Select form", ["Provider", "Patient", "Patient Insurance"])

# 1. Provider Form
if form_type == "Provider":
    st.header("Enter Provider Info")
    with st.form("provider_form"):
        name = st.text_input("Name *")
        npi = st.text_input("NPI")
        license_number = st.text_input("License Number")
        dea_number = st.text_input("DEA Number")
        medicare_number = st.text_input("Medicare Number")
        medicaid_number = st.text_input("Medicaid Number")
        etin = st.text_input("ETIN")
        ein_tin = st.text_input("EIN/TIN")
        upin_number = st.text_input("UPIN")
        clia_number = st.text_input("CLIA")

        submitted = st.form_submit_button("Submit")
        if submitted and name:
            try:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO cpt_providers (
                        name, npi, license_number, dea_number,
                        medicare_number, medicaid_number, etin,
