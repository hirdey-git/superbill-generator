# app.py
import streamlit as st
import mysql.connector
from mysql.connector import Error
import datetime as dt
import re
import time
from typing import Optional, Tuple, List, Dict, Any
import pdfkit
import tempfile
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO


from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# =============================
# Streamlit & Page Settings
# =============================
st.set_page_config(page_title="CPT Data Entry + MDM/CPT Generator", page_icon="📝", layout="wide")
st.title("📝 CPT Data Entry App")

# =============================
# OpenAI
# =============================
OPENAI_MODEL = "gpt-4o-mini"  # fast/cheap. Swap to 'gpt-4o' or 'gpt-4.1' if needed.
_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def openai_chat(prompt: str, *, system: Optional[str] = None, temperature: float = 0.0, model: str = OPENAI_MODEL) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# =============================
# DB CONFIG
# =============================
DB_CONFIG = {
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "host": st.secrets["DB_HOST"],
    "database": st.secrets["DB_NAME"],
    "autocommit": False,
}

# =============================
# Helpers
# =============================
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def exec_write(sql, params):
    conn = None
    cur = None
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
        if cur:
            cur.close()
        if conn:
            conn.close()

def exec_read(sql, params=None, dict_cursor=True):
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=dict_cursor)
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        return rows
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def to_none_if_blank(v: str):
    return v.strip() if isinstance(v, str) and v.strip() != "" else None

def create_encounters_table_if_needed():
    """Create cpt_encounters if it doesn't exist."""
    sql = """
    CREATE TABLE IF NOT EXISTS cpt_encounters (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        patient_id BIGINT NOT NULL,
        provider_id BIGINT,
        site_of_origination VARCHAR(255),
        mdm_text LONGTEXT,
        mdm_level VARCHAR(50),
        time_minutes INT,
        matched_cpt_code VARCHAR(20),
        matched_by_time BOOLEAN,
        rationale LONGTEXT,
        improved_mdm_text LONGTEXT,
        improved_cpt_code VARCHAR(20),
        ai_model VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES cpt_patients(id),
        FOREIGN KEY (provider_id) REFERENCES cpt_providers(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def create_required_indexes():
    """Optional: indexes for performance."""
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_encounters_patient ON cpt_encounters(patient_id)",
        "CREATE INDEX IF NOT EXISTS idx_encounters_provider ON cpt_encounters(provider_id)",
        "CREATE INDEX IF NOT EXISTS idx_mapping_site ON cpt_mdm_mapping(site_of_origination)",
        "CREATE INDEX IF NOT EXISTS idx_mapping_level ON cpt_mdm_mapping(mdm_level)",
    ]
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        for s in stmts:
            try:
                cur.execute(s)
            except Exception:
                # Older MySQL doesn't support "IF NOT EXISTS" for indexes, ignore errors
                pass
        conn.commit()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# run table creation once at startup
try:
    create_encounters_table_if_needed()
    create_required_indexes()
except Exception as e:
    st.error(f"Failed to ensure schema is ready: {e}")

# =============================
# MDM / CPT Engine
# =============================
def load_site_options() -> List[str]:
    rows = exec_read(
        "SELECT DISTINCT site_of_origination FROM cpt_mdm_mapping ORDER BY site_of_origination",
        dict_cursor=False
    )
    return [r[0] for r in rows]

def load_mapping_for_site(site: str) -> List[Dict[str, Any]]:
    return exec_read(
        "SELECT * FROM cpt_mdm_mapping WHERE site_of_origination = %s",
        (site,),
        dict_cursor=True
    )

def determine_mdm_level_with_ai(mdm_text: str, mapping_rows: List[Dict[str, Any]]) -> str:
    level_summary = {}
    for row in mapping_rows:
        level = row['mdm_level']
        if level not in level_summary:
            level_summary[level] = {
                "Problems Addressed": row['problems_addressed'],
                "Data Complexity": row['data_complexity'],
                "Risk": row['risk']
            }

    prompt = (
        "You are a medical coding assistant. Based on the doctor's Medical Decision Making (MDM) text below, "
        "choose the most appropriate MDM Level: Straightforward, Low, Moderate, or High.\n\n"
    )
    for level, criteria in level_summary.items():
        prompt += f"MDM Level: {level}\n"
        prompt += f"  - Problems Addressed: {criteria['Problems Addressed']}\n"
        prompt += f"  - Data Complexity: {criteria['Data Complexity']}\n"
        prompt += f"  - Risk: {criteria['Risk']}\n\n"
    prompt += f'MDM Text:\n"""\n{mdm_text}\n"""\n\nRespond ONLY with the MDM level.'

    return openai_chat(prompt, temperature=0.0)

def extract_or_estimate_time(mdm_text: str) -> Optional[int]:
    """Extract the minutes from mdm_text if present, else estimate with AI."""
    match = re.search(r'(\d{1,3})\s*(minutes|min)\b', mdm_text.lower())
    if match:
        return int(match.group(1))

    prompt = (
        "Estimate how many minutes were spent with the patient based on the following MDM text. "
        "Respond with only the integer number of minutes.\n"
        f'"""\n{mdm_text}\n"""'
    )
    try:
        content = openai_chat(prompt, temperature=0.0)
        n = re.search(r'(\d+)', content)
        return int(n.group(1)) if n else None
    except Exception:
        return None

def match_cpt(mapping_rows: List[Dict[str, Any]], mdm_level: str, time_minutes: Optional[int]) -> Tuple[Optional[Dict[str, Any]], bool]:
    # 1) try match by time range
    if time_minutes is not None:
        for r in mapping_rows:
            if r['mdm_level'].lower() == mdm_level.lower():
                if r['time_min'] is not None and r['time_max'] is not None:
                    try:
                        if r['time_min'] <= time_minutes <= r['time_max']:
                            return r, True
                    except TypeError:
                        pass
    # 2) fallback: match purely by level
    for r in mapping_rows:
        if r['mdm_level'].lower() == mdm_level.lower():
            return r, False
    return None, False

def generate_mdm_criteria_and_cpt_rationale(matched_row: Dict[str, Any], mdm_level: str, time_minutes: Optional[int], matched_by_time: bool) -> str:
    rationale = []
    rationale.append(f"The selected CPT code ({matched_row['cpt_code']}) is based on the following:")
    rationale.append("\n**MDM Criteria Met:**")
    rationale.append(f"- **Problems Addressed:** {matched_row['problems_addressed']}")
    rationale.append(f"- **Data Complexity:** {matched_row['data_complexity']}")
    rationale.append(f"- **Risk:** {matched_row['risk']}")
    rationale.append(f"\n**MDM Level:** {mdm_level}")

    if time_minutes is not None:
        if matched_by_time:
            rationale.append(f"**Time Spent:** {time_minutes} minutes (within range {matched_row['time_min']}–{matched_row['time_max']} minutes)")
        else:
            rationale.append(f"**Time Spent:** {time_minutes} minutes (Time was not used in CPT code selection)")
    else:
        rationale.append("**Time Spent:** Not documented; CPT determined using MDM level only.")

    if 'patient_type' in matched_row:
        rationale.append(f"**Patient Type:** {matched_row['patient_type']}")
    return "\n".join(rationale)

def suggest_higher_billing_mdm_and_cpt(mdm_text: str, mapping_rows: List[Dict[str, Any]]) -> str:
    # We are *not* to add new diagnoses—only reorganize/clarify based on existing info.
    higher_levels = [row for row in mapping_rows if row['mdm_level'].lower() == 'high']
    prompt = (
        "You are a medical coding assistant. Here is the doctor's current MDM text:\n"
        f"{mdm_text}\n\n"
        "Without introducing any new diagnoses or conditions, reword/restructure ONLY using content already in the text to support a higher CPT level (if reasonably supportable). "
        "If it cannot reasonably be improved, say so clearly. Then, suggest the new CPT code based on the improved MDM and explain which MDM criteria are met.\n\n"
    )
    for row in higher_levels:
        prompt += f"MDM Level: {row['mdm_level']}\n"
        prompt += f"- Problems Addressed: {row['problems_addressed']}\n"
        prompt += f"- Data Complexity: {row['data_complexity']}\n"
        prompt += f"- Risk: {row['risk']}\n\n"

    return openai_chat(prompt, temperature=0.0)

def parse_cpt_from_text(text: str) -> Optional[str]:
    m = re.search(r'\b(9\d{4})\b', text)  # typical E/M CPT pattern 99xxx
    return m.group(1) if m else None

def save_encounter(patient_id: int,
                   provider_id: Optional[int],
                   site: str,
                   mdm_text: str,
                   mdm_level: str,
                   time_minutes: Optional[int],
                   matched_cpt: Optional[Dict[str, Any]],
                   matched_by_time: bool,
                   rationale: str,
                   improved_mdm_text: Optional[str],
                   improved_cpt_code: Optional[str],
                   ai_model: str = OPENAI_MODEL) -> int:
    exec_write(
        """
        INSERT INTO cpt_encounters
        (patient_id, provider_id, site_of_origination, mdm_text, mdm_level, time_minutes,
         matched_cpt_code, matched_by_time, rationale, improved_mdm_text, improved_cpt_code, ai_model)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            patient_id,
            provider_id,
            site,
            mdm_text,
            mdm_level,
            time_minutes,
            matched_cpt['cpt_code'] if matched_cpt else None,
            1 if matched_by_time else 0,
            rationale,
            improved_mdm_text,
            improved_cpt_code,
            ai_model
        )
    )
    # mysql-connector-python doesn't return lastrowid through our helper, so do a quick read:
    row = exec_read("SELECT LAST_INSERT_ID() AS id", dict_cursor=True)[0]
    return int(row["id"])

#-----------------------
def generate_superbill_pdf_bytes(patient_name: str,
                                 provider_name: str,
                                 site: str,
                                 cpt_code: str,
                                 mdm_level: str,
                                 time_minutes,
                                 encounter_id: int) -> bytes:
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    lh = 18

    def line(text, bold=False, size=11):
        nonlocal y
        p.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        p.drawString(50, y, str(text))
        y -= lh

    line("SUPERBILL", bold=True, size=16)
    line(f"Encounter ID: {encounter_id}")
    line("")

    line("Patient", bold=True)
    line(f"Name: {patient_name}")
    line("")

    line("Provider", bold=True)
    line(f"Name: {provider_name}")
    line("")

    line("Encounter / Coding", bold=True)
    line(f"Site of Origination: {site}")
    line(f"CPT Code: {cpt_code or 'N/A'}")
    line(f"MDM Level: {mdm_level}")
    line(f"Time Spent: {time_minutes if time_minutes is not None else 'Not documented'} minutes")

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer.getvalue()

# =============================
# UI Menu
# =============================
menu = st.sidebar.selectbox(
    "Select form / view",
    [
        "Provider",
        "Patient",
        "Patient Insurance",
        "MDM / CPT Generator",
        "Browse / Verify Data"
    ]
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
                    st.rerun()
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
                    st.rerun()
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
                    st.rerun()
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
# 4) MDM / CPT Generator
# -----------------------------
elif menu == "MDM / CPT Generator":
    st.header("🧠 Generate CPT Code from MDM Text + Superbill PDF")

    # Load dropdowns
    try:
        patients = exec_read("SELECT id, name, dob FROM cpt_patients ORDER BY name")
        providers = exec_read("SELECT id, name FROM cpt_providers ORDER BY name")
        sites = load_site_options()
    except Exception as e:
        st.error(f"Error loading dropdowns: {e}")
        st.stop()

    if not patients:
        st.warning("No patients found. Please add a patient first.")
        st.stop()
    if not providers:
        st.warning("No providers found. Please add a provider first.")
        st.stop()
    if not sites:
        st.warning("No sites found in cpt_mdm_mapping. Please populate that table first.")
        st.stop()

    patient_map = {f"{p['name']} (#{p['id']}, DOB: {p['dob']})": p['id'] for p in patients}
    provider_map = {f"{p['name']} (#{p['id']})": p['id'] for p in providers}

    with st.form("mdm_form", clear_on_submit=False):
        patient_display = st.selectbox("Select Patient *", list(patient_map.keys()))
        provider_display = st.selectbox("Select Provider *", list(provider_map.keys()))
        site = st.selectbox("Site of Origination *", sites)
        mdm_text = st.text_area("Enter MDM Text *", height=240)
        submit_generate = st.form_submit_button("Generate CPT & Save Encounter")

    if submit_generate:
        if not mdm_text.strip():
            st.error("MDM Text is required.")
            st.stop()

        try:
            mapping_rows = load_mapping_for_site(site)
            if not mapping_rows:
                st.error("No mapping rows found for this site.")
                st.stop()

            mdm_level = determine_mdm_level_with_ai(mdm_text, mapping_rows)
            time_minutes = extract_or_estimate_time(mdm_text)
            matched_cpt, matched_by_time = match_cpt(mapping_rows, mdm_level, time_minutes)

            if matched_cpt:
                st.success(f"Matched CPT: {matched_cpt['cpt_code']}")
                rationale = generate_mdm_criteria_and_cpt_rationale(matched_cpt, mdm_level, time_minutes, matched_by_time)
                st.markdown("### Rationale")
                st.markdown(rationale)
            else:
                st.warning("❌ No CPT code found matching the MDM level/time.")
                rationale = "No CPT found."

            improved_suggestion = suggest_higher_billing_mdm_and_cpt(mdm_text, mapping_rows)
            st.markdown("### Suggested Improvement for Higher CPT (if possible)")
            st.write(improved_suggestion)
            improved_cpt_code = parse_cpt_from_text(improved_suggestion)

            # Save encounter
            encounter_id = save_encounter(
                patient_id=patient_map[patient_display],
                provider_id=provider_map[provider_display],
                site=site,
                mdm_text=mdm_text,
                mdm_level=mdm_level,
                time_minutes=time_minutes,
                matched_cpt=matched_cpt,
                matched_by_time=matched_by_time,
                rationale=rationale,
                improved_mdm_text=improved_suggestion,
                improved_cpt_code=improved_cpt_code,
                ai_model=OPENAI_MODEL  # keep if your table has this column
            )
            st.success(f"✅ Encounter saved (ID: {encounter_id}).")

            # -------------------
            # 📄 Superbill PDF (ReportLab)
            # -------------------
            st.subheader("📄 Download Superbill PDF")
            patient_name = patient_display.split(" (")[0]
            provider_name = provider_display.split(" (")[0]
            cpt_code = matched_cpt['cpt_code'] if matched_cpt else "N/A"

            pdf_bytes = generate_superbill_pdf_bytes(
                patient_name=patient_name,
                provider_name=provider_name,
                site=site,
                cpt_code=cpt_code,
                mdm_level=mdm_level,
                time_minutes=time_minutes,
                encounter_id=encounter_id
            )

            st.download_button(
                label="⬇️ Download Superbill PDF",
                data=pdf_bytes,
                file_name=f"superbill_encounter_{encounter_id}.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error generating/saving encounter: {e}")

    st.subheader("Latest 100 Encounters")
    try:
        rows = exec_read(
            """
            SELECT e.id, p.name AS patient, pr.name AS provider,
                   e.site_of_origination, e.mdm_level, e.time_minutes,
                   e.matched_cpt_code, e.improved_cpt_code, e.created_at
            FROM cpt_encounters e
            JOIN cpt_patients p ON p.id = e.patient_id
            LEFT JOIN cpt_providers pr ON pr.id = e.provider_id
            ORDER BY e.id DESC
            LIMIT 100
            """
        )
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No encounters yet.")
    except Exception as e:
        st.error(f"Read failed: {e}")


# -----------------------------
# 5) Browse / Verify Data
# -----------------------------
elif menu == "Browse / Verify Data":
    st.header("Quick Browse")
    which = st.selectbox("Choose a table", ["Providers", "Patients", "Patient Insurances", "Encounters"])

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
        elif which == "Patient Insurances":
            rows = exec_read(
                """
                SELECT i.id, p.name AS patient, i.plan_name, i.policy_id,
                       i.coverage_type, i.expiration_date, i.updated_at
                FROM cpt_patient_insurances i
                JOIN cpt_patients p ON p.id = i.patient_id
                ORDER BY i.id DESC LIMIT 1000
                """
            )
        else:
            rows = exec_read(
                """
                SELECT e.id, p.name AS patient, pr.name AS provider,
                       e.site_of_origination, e.mdm_level, e.time_minutes,
                       e.matched_cpt_code, e.improved_cpt_code, e.created_at
                FROM cpt_encounters e
                JOIN cpt_patients p ON p.id = e.patient_id
                LEFT JOIN cpt_providers pr ON pr.id = e.provider_id
                ORDER BY e.id DESC LIMIT 1000
                """
            )
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No rows yet.")
    except Exception as e:
        st.error(f"Read failed: {e}")
