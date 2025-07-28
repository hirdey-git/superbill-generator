# app.py
import streamlit as st
import mysql.connector
from mysql.connector import Error
from datetime import date
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# CONFIG / INIT
# ----------------------------
load_dotenv()
st.set_page_config(page_title="CPT + Superbill Generator", page_icon="🩺", layout="wide")

# Secrets expected: OPENAI_API_KEY, DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DB_CONFIG = {
    "user": st.secrets["DB_USER"],
    "password": st.secrets["DB_PASSWORD"],
    "host": st.secrets["DB_HOST"],
    "database": st.secrets["DB_NAME"],
    "autocommit": False
}

# ----------------------------
# DB UTILITIES
# ----------------------------
def get_conn():
    return mysql.connector.connect(**DB_CONFIG)

def run_query(conn, sql, params=None, fetchone=False, fetchall=False, return_lastrowid=False):
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, params or ())
    result = None
    if fetchone:
        result = cur.fetchone()
    elif fetchall:
        result = cur.fetchall()
    if return_lastrowid:
        result = cur.lastrowid
    return result, cur

def commit_and_close(conn, cur=None):
    if cur:
        cur.close()
    conn.commit()
    conn.close()

def rollback_and_close(conn, cur=None):
    if cur:
        cur.close()
    conn.rollback()
    conn.close()

# ----------------------------
# MASTER DATA LOADERS
# ----------------------------
def load_site_options() -> List[str]:
    conn = get_conn()
    try:
        rows, cur = run_query(
            conn,
            "SELECT DISTINCT site_of_origination FROM cpt_mdm_mapping ORDER BY site_of_origination",
            fetchall=True
        )
        commit_and_close(conn, cur)
        return [r["site_of_origination"] for r in rows]
    except Exception:
        conn.close()
        return []

def load_providers() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows, cur = run_query(conn, "SELECT * FROM providers ORDER BY name", fetchall=True)
    commit_and_close(conn, cur)
    return rows

def load_patients() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows, cur = run_query(conn, "SELECT * FROM patients ORDER BY name", fetchall=True)
    commit_and_close(conn, cur)
    return rows

def load_mapping_for_site(site: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows, cur = run_query(
        conn,
        "SELECT * FROM cpt_mdm_mapping WHERE site_of_origination = %s",
        (site,),
        fetchall=True
    )
    commit_and_close(conn, cur)
    return rows

# ----------------------------
# UPSERT HELPERS
# ----------------------------
def upsert_provider(payload: Dict[str, Any]) -> int:
    """
    Upsert by NPI if present, else by name.
    Returns provider_id.
    """
    conn = get_conn()
    try:
        if payload.get("npi"):
            existing, cur = run_query(
                conn,
                "SELECT id FROM providers WHERE npi = %s",
                (payload["npi"],),
                fetchone=True
            )
        else:
            existing, cur = run_query(
                conn,
                "SELECT id FROM providers WHERE name = %s",
                (payload["name"],),
                fetchone=True
            )

        if existing:
            prov_id = existing["id"]
            sql = """
                UPDATE providers
                SET name=%s, npi=%s, license_number=%s, dea_number=%s, medicare_number=%s,
                    medicaid_number=%s, etin=%s, ein_tin=%s, upin_number=%s, clia_number=%s
                WHERE id=%s
            """
            params = (
                payload.get("name"), payload.get("npi"), payload.get("license_number"),
                payload.get("dea_number"), payload.get("medicare_number"),
                payload.get("medicaid_number"), payload.get("etin"), payload.get("ein_tin"),
                payload.get("upin_number"), payload.get("clia_number"), prov_id
            )
            cur.execute(sql, params)
        else:
            sql = """
                INSERT INTO providers
                (name, npi, license_number, dea_number, medicare_number, medicaid_number,
                 etin, ein_tin, upin_number, clia_number)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            params = (
                payload.get("name"), payload.get("npi"), payload.get("license_number"),
                payload.get("dea_number"), payload.get("medicare_number"),
                payload.get("medicaid_number"), payload.get("etin"), payload.get("ein_tin"),
                payload.get("upin_number"), payload.get("clia_number")
            )
            cur.execute(sql, params)
            prov_id = cur.lastrowid

        commit_and_close(conn, cur)
        return prov_id
    except Error as e:
        rollback_and_close(conn)
        raise e

def upsert_patient(payload: Dict[str, Any]) -> int:
    """
    Simple upsert by (name, dob). Adjust your dedupe strategy if needed.
    """
    conn = get_conn()
    try:
        existing, cur = run_query(
            conn,
            "SELECT id FROM patients WHERE name = %s AND dob = %s",
            (payload["name"], payload["dob"]),
            fetchone=True
        )

        if existing:
            pid = existing["id"]
            sql = """
                UPDATE patients
                SET address=%s, phone=%s, gender_at_birth=%s, gender_identity=%s, pronouns=%s,
                    marital_status=%s, race=%s, language=%s, religion=%s, medicare_number=%s,
                    medicaid_number=%s, ssn=%s, employer_name=%s, employer_address=%s
                WHERE id=%s
            """
            params = (
                payload.get("address"), payload.get("phone"), payload.get("gender_at_birth"),
                payload.get("gender_identity"), payload.get("pronouns"), payload.get("marital_status"),
                payload.get("race"), payload.get("language"), payload.get("religion"),
                payload.get("medicare_number"), payload.get("medicaid_number"),
                payload.get("ssn"), payload.get("employer_name"), payload.get("employer_address"), pid
            )
            cur.execute(sql, params)
        else:
            sql = """
                INSERT INTO patients
                (name, dob, address, phone, gender_at_birth, gender_identity, pronouns,
                 marital_status, race, language, religion, medicare_number, medicaid_number,
                 ssn, employer_name, employer_address)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            params = (
                payload.get("name"), payload.get("dob"), payload.get("address"), payload.get("phone"),
                payload.get("gender_at_birth"), payload.get("gender_identity"), payload.get("pronouns"),
                payload.get("marital_status"), payload.get("race"), payload.get("language"),
                payload.get("religion"), payload.get("medicare_number"), payload.get("medicaid_number"),
                payload.get("ssn"), payload.get("employer_name"), payload.get("employer_address")
            )
            cur.execute(sql, params)
            pid = cur.lastrowid

        commit_and_close(conn, cur)
        return pid
    except Error as e:
        rollback_and_close(conn)
        raise e

def replace_patient_insurances(patient_id: int, ins_list: List[Dict[str, Any]]):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM patient_insurances WHERE patient_id = %s", (patient_id,))
        if ins_list:
            sql = """
                INSERT INTO patient_insurances
                (patient_id, plan_name, policy_id, coverage_type, expiration_date)
                VALUES (%s,%s,%s,%s,%s)
            """
            for ins in ins_list:
                cur.execute(sql, (
                    patient_id,
                    ins.get("plan_name"),
                    ins.get("policy_id"),
                    ins.get("coverage_type") or "Medical",
                    ins.get("expiration_date")
                ))
        commit_and_close(conn, cur)
    except Error as e:
        rollback_and_close(conn)
        raise e

# ----------------------------
# MDM + AI HELPERS
# ----------------------------
def parse_icd_list(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r'[,;\n]+', raw)
    return [p.strip() for p in parts if p.strip()]

def determine_mdm_level_with_ai(
    mdm_text: str,
    mapping_rows: List[Dict[str, Any]]
) -> Tuple[str, Dict[str, int]]:
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
        "You are a medical coding assistant. Based on the doctor's Medical Decision Making (MDM) "
        "description below, identify the most appropriate MDM Level: Straightforward, Low, Moderate, or High.\n\n"
    )
    for level, c in level_summary.items():
        prompt += f"MDM Level: {level}\n"
        prompt += f"  - Problems Addressed: {c['Problems Addressed']}\n"
        prompt += f"  - Data Complexity: {c['Data Complexity']}\n"
        prompt += f"  - Risk: {c['Risk']}\n\n"
    prompt += f'MDM Text:\n"""\n{mdm_text}\n"""\n\nRespond ONLY with the MDM level.'

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    mdm_level = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if usage:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }
    return mdm_level, usage_dict

def extract_or_estimate_time(mdm_text: str) -> Optional[int]:
    match = re.search(r'(\d{1,3})\s*(minutes|min)', mdm_text.lower())
    if match:
        return int(match.group(1))

    prompt = (
        'Estimate how many minutes were spent with the patient based on the following MDM text:\n"""\n'
        f'{mdm_text}\n"""Respond with only a number of minutes'
    )

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    try:
        return int(re.search(r'(\d+)', content).group(1))
    except Exception:
        return None

def match_cpt(mapping_rows, mdm_level, time_minutes):
    # Try time match first
    if time_minutes is not None:
        for row in mapping_rows:
            if row['mdm_level'].lower() == mdm_level.lower():
                if row['time_min'] is not None and row['time_max'] is not None:
                    if row['time_min'] <= time_minutes <= row['time_max']:
                        return row, True
    # Fallback to MDM
    for row in mapping_rows:
        if row['mdm_level'].lower() == mdm_level.lower():
            return row, False
    return None, False

def generate_mdm_criteria_and_cpt_rationale(matched_row, mdm_level, time_minutes, matched_by_time):
    if not matched_row:
        return "No CPT code matched."
    rationale = f"The selected CPT code ({matched_row['cpt_code']}) is based on the following:\n"
    rationale += f"\n**MDM Criteria Met:**"
    rationale += f"\n- **Problems Addressed:** {matched_row['problems_addressed']}"
    rationale += f"\n- **Data Complexity:** {matched_row['data_complexity']}"
    rationale += f"\n- **Risk:** {matched_row['risk']}"
    rationale += f"\n\n**MDM Level:** {mdm_level}"

    if time_minutes:
        rationale += f"\n**Time Spent:** {time_minutes} minutes"
        if matched_by_time:
            rationale += f" (within range {matched_row['time_min']}–{matched_row['time_max']} minutes)"
        else:
            rationale += " (Time was not used in CPT code selection)"
    else:
        rationale += "\n**Time Spent:** Not documented; CPT determined using MDM level only."

    rationale += f"\n**Patient Type:** {matched_row['patient_type']}"
    return rationale

def suggest_higher_billing_mdm_and_cpt(mdm_text, mapping_rows):
    higher_levels = [row for row in mapping_rows if row['mdm_level'].lower() == 'high']
    prompt = (
        f"You are a medical coding assistant. Here is the doctor's current MDM text:\n{mdm_text}\n\n"
        "Use ONLY the following elements (Problems Addressed, Data Complexity, Risk) from higher MDM levels to suggest "
        "how this text could be reworded to support higher CPT billing (without adding new diagnoses or conditions). "
        "Then, suggest the new CPT code based on the improved MDM and provide the MDM criteria met:\n\n"
    )

    for row in higher_levels:
        prompt += f"MDM Level: {row['mdm_level']}\n"
        prompt += f"- Problems Addressed: {row['problems_addressed']}\n"
        prompt += f"- Data Complexity: {row['data_complexity']}\n"
        prompt += f"- Risk: {row['risk']}\n\n"

    prompt += "If it cannot reasonably be improved, state that clearly."

    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# ----------------------------
# WRITE: MDM REQUEST/RESULT
# ----------------------------
def insert_mdm_request(patient_id, provider_id, site_of_origination, mdm_text) -> int:
    conn = get_conn()
    try:
        sql = """
            INSERT INTO mdm_requests (patient_id, provider_id, site_of_origination, mdm_text)
            VALUES (%s,%s,%s,%s)
        """
        cur = conn.cursor()
        cur.execute(sql, (patient_id, provider_id, site_of_origination, mdm_text))
        mdm_request_id = cur.lastrowid
        commit_and_close(conn, cur)
        return mdm_request_id
    except Error as e:
        rollback_and_close(conn)
        raise e

def insert_mdm_result(mdm_request_id, mdm_level, time_minutes, selected_cpt, matched_by_time,
                      rationale, suggestion, model_used, usage_dict) -> int:
    conn = get_conn()
    try:
        sql = """
            INSERT INTO mdm_results
            (mdm_request_id, mdm_level, estimated_time_minutes, selected_cpt_code, matched_by_time,
             rationale, suggestion_higher_mdm, model_used, temperature,
             prompt_tokens, completion_tokens, total_tokens)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        cur = conn.cursor()
        cur.execute(sql, (
            mdm_request_id, mdm_level, time_minutes, selected_cpt, matched_by_time,
            rationale, suggestion, model_used, 0.0,
            usage_dict.get("prompt_tokens", 0),
            usage_dict.get("completion_tokens", 0),
            usage_dict.get("total_tokens", 0)
        ))
        mdm_result_id = cur.lastrowid
        commit_and_close(conn, cur)
        return mdm_result_id
    except Error as e:
        rollback_and_close(conn)
        raise e

# ----------------------------
# WRITE: SUPERBILL (+ child tables)
# ----------------------------
def insert_superbill(patient_id, provider_id, mdm_result_id, dos, site_of_visit,
                     prior_auth, custom_code) -> int:
    conn = get_conn()
    try:
        sql = """
            INSERT INTO superbills
            (patient_id, provider_id, mdm_result_id, dos, site_of_visit, prior_authorization, custom_code)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """
        cur = conn.cursor()
        cur.execute(sql, (
            patient_id, provider_id, mdm_result_id, dos, site_of_visit, prior_auth, custom_code
        ))
        superbill_id = cur.lastrowid
        commit_and_close(conn, cur)
        return superbill_id
    except Error as e:
        rollback_and_close(conn)
        raise e

def insert_superbill_dx_codes(superbill_id: int, icd_codes: List[str]):
    conn = get_conn()
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO superbill_icd10_codes (superbill_id, icd10_code, position_tiny)
            VALUES (%s, %s, %s)
        """
        for i, code in enumerate(icd_codes, start=1):
            cur.execute(sql, (superbill_id, code, i))
        commit_and_close(conn, cur)
    except Error as e:
        rollback_and_close(conn)
        raise e

def insert_superbill_lines(superbill_id: int, lines: List[Dict[str, Any]]):
    conn = get_conn()
    try:
        cur = conn.cursor()
        sql = """
            INSERT INTO superbill_lines
            (superbill_id, code_type, code, modifier1, modifier2, modifier3, modifier4,
             units_or_minutes, price, dx_pointer_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        for ln in lines:
            cur.execute(sql, (
                superbill_id,
                ln.get("code_type", "CPT"),
                ln.get("code"),
                ln.get("modifier1"),
                ln.get("modifier2"),
                ln.get("modifier3"),
                ln.get("modifier4"),
                ln.get("units_or_minutes", 1),
                ln.get("price"),
                json.dumps(ln.get("dx_pointer_json")) if ln.get("dx_pointer_json") else None
            ))
        commit_and_close(conn, cur)
    except Error as e:
        rollback_and_close(conn)
        raise e

# ----------------------------
# UI
# ----------------------------
st.title("🩺 Streamlit → MySQL: Intake, MDM→CPT, and Superbill")

tab_intake, tab_mdm = st.tabs([
    "👤 Intake (Providers / Patients / Insurance)",
    "🧠 MDM → CPT & Superbill"
])

# ----------------------------
# TAB 1: INTAKE
# ----------------------------
with tab_intake:
    st.subheader("Provider + Patient Intake")

    # IMPORTANT: submit button is INSIDE this form
    with st.form("intake_form"):
        st.markdown("### Provider")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            prov_name = st.text_input("Provider Name *")
            prov_npi = st.text_input("NPI")
            prov_license = st.text_input("License #")
            prov_dea = st.text_input("DEA #")
            prov_upin = st.text_input("UPIN #")
        with pcol2:
            prov_medicare = st.text_input("Medicare #")
            prov_medicaid = st.text_input("Medicaid #")
            prov_etin = st.text_input("ETIN #")
            prov_ein = st.text_input("EIN/TIN")
            prov_clia = st.text_input("CLIA #")
        with pcol3:
            st.write("")  # spacer

        st.markdown("---")
        st.markdown("### Patient")
        name = st.text_input("Patient Name *")
        dob = st.date_input("DOB *", value=date(1980, 1, 1))
        address = st.text_area("Address")
        phone = st.text_input("Phone")
        pcol4, pcol5, pcol6 = st.columns(3)
        with pcol4:
            gender_at_birth = st.text_input("Gender at Birth")
            marital = st.text_input("Marital Status")
            race = st.text_input("Race")
            ssn = st.text_input("SSN")
        with pcol5:
            gender_identity = st.text_input("Gender Identity")
            language = st.text_input("Language")
            religion = st.text_input("Religion")
            employer_name = st.text_input("Employer Name")
        with pcol6:
            pronouns = st.text_input("Pronouns")
            medicare_number = st.text_input("Medicare #")
            medicaid_number = st.text_input("Medicaid #")
            employer_address = st.text_area("Employer Address")

        st.markdown("---")
        st.markdown("### Insurance (1)")
        ins1_plan = st.text_input("Plan (1)")
        ins1_policy = st.text_input("Policy ID (1)")
        ins1_cov = st.selectbox("Coverage Type (1)", ["Medical", "Other"])
        ins1_exp = st.date_input("Expiration (1)", value=None, key="ins1_exp2")

        st.markdown("### Insurance (2)")
        ins2_plan = st.text_input("Plan (2)")
        ins2_policy = st.text_input("Policy ID (2)")
        ins2_cov = st.selectbox("Coverage Type (2)", ["Medical", "Other"])
        ins2_exp = st.date_input("Expiration (2)", value=None, key="ins2_exp3")

        submitted = st.form_submit_button("Save / Update")

    if submitted:
        if not prov_name:
            st.error("Provider name is required.")
            st.stop()
        if not name or not dob:
            st.error("Patient name and DOB are required.")
            st.stop()

        provider_payload = {
            "name": prov_name,
            "npi": prov_npi,
            "license_number": prov_license,
            "dea_number": prov_dea,
            "medicare_number": prov_medicare,
            "medicaid_number": prov_medicaid,
            "etin": prov_etin,
            "ein_tin": prov_ein,
            "upin_number": prov_upin,
            "clia_number": prov_clia
        }
        prov_id = upsert_provider(provider_payload)

        patient_payload = {
            "name": name,
            "dob": dob,
            "address": address,
            "phone": phone,
            "gender_at_birth": gender_at_birth,
            "gender_identity": gender_identity,
            "pronouns": pronouns,
            "marital_status": marital,
            "race": race,
            "language": language,
            "religion": religion,
            "medicare_number": medicare_number,
            "medicaid_number": medicaid_number,
            "ssn": ssn,
            "employer_name": employer_name,
            "employer_address": employer_address
        }
        pat_id = upsert_patient(patient_payload)

        ins_list = []
        if ins1_plan or ins1_policy:
            ins_list.append({
                "plan_name": ins1_plan,
                "policy_id": ins1_policy,
                "coverage_type": ins1_cov,
                "expiration_date": ins1_exp
            })
        if ins2_plan or ins2_policy:
            ins_list.append({
                "plan_name": ins2_plan,
                "policy_id": ins2_policy,
                "coverage_type": ins2_cov,
                "expiration_date": ins2_exp
            })
        replace_patient_insurances(pat_id, ins_list)

        st.success(f"Saved! Provider ID = {prov_id}, Patient ID = {pat_id}")

# ----------------------------
# TAB 2: MDM + CPT + SUPERBILL
# ----------------------------
with tab_mdm:
    st.subheader("MDM → CPT + Save to mdm_requests/mdm_results + Superbill")

    providers = load_providers()
    patients = load_patients()
    site_options = load_site_options()

    if not providers or not patients:
        st.warning("Please create at least one Provider and one Patient in the Intake tab.")
    else:
        # IMPORTANT: submit button is INSIDE this form
        with st.form("mdm_form"):
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                provider_choice = st.selectbox(
                    "Provider", providers, format_func=lambda x: f"{x['name']} (ID {x['id']})"
                )
            with pcol2:
                patient_choice = st.selectbox(
                    "Patient", patients, format_func=lambda x: f"{x['name']} (ID {x['id']})"
                )

            site = st.selectbox("Site of Origination (from mapping)", site_options)

            mdm_text = st.text_area("MDM Text", height=250)
            icd10_raw = st.text_area("ICD-10 Codes (comma or newline separated)")

            # Superbill info:
            st.markdown("#### Superbill Options")
            dos = st.date_input("DOS", value=date.today())
            site_of_visit = st.text_input("Site of Visit", value=site)
            prior_auth = st.text_input("Prior Authorization #")
            custom_code = st.text_input("Custom Code (self pay, no show, etc.)")
            modifier = st.text_input("Modifier")
            qty_minutes = st.number_input("Units/Minutes", min_value=0, value=0)
            price_override = st.number_input("CPT Price (optional)", min_value=0.00, value=0.00, step=0.01)
            create_bill = st.checkbox("Create Superbill now", value=True)

            submitted2 = st.form_submit_button("Run AI, Save, (optionally) Create Superbill")

        if submitted2:
            if not mdm_text.strip():
                st.error("MDM text is required.")
                st.stop()

            mapping_rows = load_mapping_for_site(site)
            if not mapping_rows:
                st.error("No mappings found for this site.")
                st.stop()

            patient_id = patient_choice["id"]
            provider_id = provider_choice["id"]

            # 1) Save raw request
            mdm_request_id = insert_mdm_request(patient_id, provider_id, site, mdm_text)

            # 2) AI → mdm_level, time
            mdm_level, usage = determine_mdm_level_with_ai(mdm_text, mapping_rows)
            time_minutes = extract_or_estimate_time(mdm_text)
            matched_cpt, matched_by_time = match_cpt(mapping_rows, mdm_level, time_minutes)
            selected_cpt = matched_cpt["cpt_code"] if matched_cpt else None
            rationale = generate_mdm_criteria_and_cpt_rationale(matched_cpt, mdm_level, time_minutes, matched_by_time)
            suggestion = suggest_higher_billing_mdm_and_cpt(mdm_text, mapping_rows)

            # 3) Save mdm_result
            mdm_result_id = insert_mdm_result(
                mdm_request_id=mdm_request_id,
                mdm_level=mdm_level,
                time_minutes=time_minutes,
                selected_cpt=selected_cpt,
                matched_by_time=matched_by_time,
                rationale=rationale,
                suggestion=suggestion,
                model_used="gpt-4",
                usage_dict=usage
            )

            st.success(f"MDM saved. mdm_request_id={mdm_request_id}, mdm_result_id={mdm_result_id}")
            st.write("**MDM Level:**", mdm_level)
            st.write("**Estimated Time:**", time_minutes if time_minutes else "N/A")
            st.write("**Selected CPT:**", selected_cpt or "N/A")
            st.markdown("**Rationale:**")
            st.markdown(rationale)
            st.markdown("**Suggestion (higher CPT if possible):**")
            st.markdown(suggestion)

            if create_bill:
                icd_list = parse_icd_list(icd10_raw)

                # 4) Create superbill
                superbill_id = insert_superbill(
                    patient_id=patient_id,
                    provider_id=provider_id,
                    mdm_result_id=mdm_result_id,
                    dos=dos,
                    site_of_visit=site_of_visit,
                    prior_auth=prior_auth,
                    custom_code=custom_code
                )
                # ICDs
                insert_superbill_dx_codes(superbill_id, icd_list)

                # CPT line
                lines = []
                if selected_cpt:
                    lines.append({
                        "code_type": "CPT",
                        "code": selected_cpt,
                        "modifier1": modifier or None,
                        "units_or_minutes": qty_minutes if qty_minutes else (time_minutes or 1),
                        "price": price_override if price_override else None,
                        "dx_pointer_json": [str(i) for i in range(1, len(icd_list) + 1)] if icd_list else None
                    })

                insert_superbill_lines(superbill_id, lines)

                st.success(f"Superbill created! ID = {superbill_id}")
                st.json({
                    "superbill_id": superbill_id,
                    "lines": lines,
                    "icd10": icd_list
                })
