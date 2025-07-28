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
                            to_none_if_blank(clia_number),
                        )
                    )
                    st.success("Provider added successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Insert failed: {e}")

    st.subheader("Latest 100 Providers")
    try:
        providers = exec_read(
            "SELECT id, name, npi, updated_at FROM cpt_providers ORDER BY id DESC LIMIT 100"
        )
        if providers:
            st.dataframe(providers, use_container_width=True)
        else:
            st.info("No providers yet.")
    except Exception as e:
        st.error(f"Read failed: {e}")

# -----------------------------
# 2) Patient
# -----------------------------
elif menu == "Patient":
    st.header("Enter Patient Info")

    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("Name *")
        dob = st.date_input("DOB", value=dt.date(1990, 1, 1))
        address = st.text_area("Address")
        phone = st.text_input("Phone")
        gender_at_birth = st.text_input("Gender at Birth")
        gender_identity = st.text_input("Gender Identity")
        pronouns = st.text_input("Pronouns")
        marital_status = st.text_input("Marital Status")
        race = st.text_input("Race")
        language = st.text_input("Language")
        religion = st.text_input("Religion")
        medicare_number = st.text_input("Medicare Number")
        medicaid_number = st.text_input("Medicaid Number")
        ssn = st.text_input("SSN")
        employer_name = st.text_input("Employer Name")
        employer_address = st.text_area("Employer Address")

        submitted = st.form_submit_button("Save Patient")
        if submitted:
            if not name.strip():
                st.error("Name is required.")
            else:
                try:
                    exec_write(
                        """
                        INSERT INTO cpt_patients (
                            name, dob, address, phone, gender_at_birth,
                            gender_identity, pronouns, marital_status, race, language,
                            religion, medicare_number, medicaid_number, ssn,
                            employer_name, employer_address
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                  %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            name.strip(), dob,
                            to_none_if_blank(address),
                            to_none_if_blank(phone),
                            to_none_if_blank(gender_at_birth),
                            to_none_if_blank(gender_identity),
                            to_none_if_blank(pronouns),
                            to_none_if_blank(marital_status),
                            to_none_if_blank(race),
                            to_none_if_blank(language),
                            to_none_if_blank(religion),
                            to_none_if_blank(medicare_number),
                            to_none_if_blank(medicaid_number),
                            to_none_if_blank(ssn),
                            to_none_if_blank(employer_name),
                            to_none_if_blank(employer_address),
                        )
                    )
                    st.success("Patient added successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Insert failed: {e}")

    st.subheader("Latest 100 Patients")
    try:
        patients = exec_read(
            """
            SELECT id, name, dob, phone, updated_at
            FROM cpt_patients
            ORDER BY id DESC LIMIT 100
            """
        )
        if patients:
            st.dataframe(patients, use_container_width=True)
        else:
            st.info("No patients yet.")
    except Exception as e:
        st.error(f"Read failed: {e}")

# -----------------------------
# 3) Patient Insurance
# -----------------------------
elif menu == "Patient Insurance":
    st.header("Enter Patient Insurance Info")

    # Fetch patients for the dropdown
    try:
        patients = exec_read("SELECT id, name FROM cpt_patients ORDER BY name")
    except Exception as e:
        patients = []
        st.error(f"Could not fetch patients: {e}")

    if not patients:
        st.warning("No patients found. Please add a patient first.")
    else:
        patient_map = {f"{p['name']} (#{p['id']})": p["id"] for p in patients}

        with st.form("insurance_form", clear_on_submit=True):
            patient_display = st.selectbox("Patient *", list(patient_map.keys()))
            plan_name = st.text_input("Plan Name")
            policy_id = st.text_input("Policy ID")
            coverage_type = st.selectbox("Coverage Type", ["Medical", "Other"])
            expiration_date = st.date_input("Expiration Date", value=dt.date.today())

            submitted = st.form_submit_button("Save Insurance")
            if submitted:
                try:
                    exec_write(
                        """
                        INSERT INTO cpt_patient_insurances (
                            patient_id, plan_name, policy_id, coverage_type, expiration_date
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            patient_map[patient_display],
                            to_none_if_blank(plan_name),
                            to_none_if_blank(policy_id),
                            coverage_type,
                            expiration_date
                        )
                    )
                    st.success("Insurance added successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Insert failed: {e}")

    st.subheader("Latest 100 Patient Insurances")
    try:
        ins = exec_read(
            """
            SELECT i.id, p.name AS patient, i.plan_name, i.policy_id,
                   i.coverage_type, i.expiration_date, i.updated_at
            FROM cpt_patient_insurances i
            JOIN cpt_patients p ON p.id = i.patient_id
            ORDER BY i.id DESC
            LIMIT 100
            """
        )
        if ins:
            st.dataframe(ins, use_container_width=True)
        else:
            st.info("No patient insurances yet.")
    except Exception as e:
        st.error(f"Read failed: {e}")

# -----------------------------
# 4) Browse / Verify Data
# -----------------------------
elif menu == "Browse / Verify Data":
    st.header("Quick Browse")
    which = st.selectbox("Choose a table", ["Providers", "Patients", "Patient Insurances"])

    try:
        if which == "Providers":
            rows = exec_read(
                """
                SELECT id, name, npi, medicare_number, medicaid_number, updated_at
                FROM cpt_providers
                ORDER BY id DESC LIMIT 1000
                """
            )
        elif which == "Patients":
            rows = exec_read(
                """
                SELECT id, name, dob, phone, ssn, medicare_number, medicaid_number, updated_at
                FROM cpt_patients
                ORDER BY id DESC LIMIT 1000
                """
            )
        else:
            rows = exec_read(
                """
                SELECT i.id, p.name AS patient, i.plan_name, i.policy_id,
                       i.coverage_type, i.expiration_date, i.updated_at
                FROM cpt_patient_insurances i
                JOIN cpt_patients p ON p.id = i.patient_id
                ORDER BY i.id DESC LIMIT 1000
                """
            )
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No rows yet.")
    except Exception as e:
        st.error(f"Read failed: {e}")
