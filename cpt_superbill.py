import streamlit as st
import mysql.connector
from mysql.connector import Error
import datetime as dt

# -----------------------------
# DB CONFIG (from st.secrets)
# -----------------------------
DB_CONFIG = {
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "host": st.secrets["DB_HOST"],
    "database": st.secrets["DB_NAME"],
    "autocommit": False,
}

# -----------------------------
# Helpers
# -----------------------------
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def exec_write(sql, params):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            cur.close()
            conn.close()

def exec_read(sql, params=None, dict_cursor=True):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=dict_cursor)
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        return rows
    finally:
        if conn:
            cur.close()
            conn.close()

def to_none_if_blank(v: str):
    return v.strip() if isinstance(v, str) and v.strip() != "" else None

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="CPT Data Entry", page_icon="📝", layout="wide")
st.title("📝 CPT Data Entry App")

menu = st.sidebar.selectbox(
    "Select form / view",
    ["Provider", "Patient", "Patient Insurance", "Browse / Verify Data"]
)

# -----------------------------
# 1) Provider
# -----------------------------
if menu == "Provider":
    st.header("Enter Provider Info")

    with st.form("provider_form", clear_on_submit=True):
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

        submitted = st.form_submit_button("Save Provider")
        if submitted:
            if not name.strip():
                st.error("Name is required.")
            else:
                try:
                    exec_write(
                        """
                        INSERT INTO cpt_providers (
                            name, npi, license_number, dea_number,
                            medicare_number, medicaid_number, etin,
                            ein_tin, upin_number, clia_number
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            name.strip(),
                            to_none_if_blank(npi),
                            to_none_if_blank(license_number),
                            to_none_if_blank(dea_number),
                            to_none_if_blank(medicare_number),
                            to_none_if_blank(medicaid_number),
                            to_none_if_blank(etin),
                            to_none_if_blank(ein_tin),
                            to_none_if_blank(upin_number),
                            to_none_if_blank(c_
