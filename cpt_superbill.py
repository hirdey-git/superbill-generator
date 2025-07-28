import os
import datetime as dt
from typing import Optional, List

import streamlit as st
from sqlalchemy import (
    create_engine, Column, BigInteger, String, Text, Date, Enum, TIMESTAMP,
    ForeignKey, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, scoped_session

# ----------------------------
# SQLAlchemy setup
# ----------------------------
Base = declarative_base()

coverage_type_enum = Enum('Medical', 'Other', name='coverage_type')

class CPTProvider(Base):
    __tablename__ = "cpt_providers"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    npi = Column(String(50))
    license_number = Column(String(50))
    dea_number = Column(String(50))
    medicare_number = Column(String(50))
    medicaid_number = Column(String(50))
    etin = Column(String(50))
    ein_tin = Column(String(50))
    upin_number = Column(String(50))
    clia_number = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )


class CPTPatient(Base):
    __tablename__ = "cpt_patients"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    dob = Column(Date)
    address = Column(Text)
    phone = Column(String(50))
    gender_at_birth = Column(String(50))
    gender_identity = Column(String(50))
    pronouns = Column(String(50))
    marital_status = Column(String(50))
    race = Column(String(50))
    language = Column(String(50))
    religion = Column(String(50))
    medicare_number = Column(String(50))
    medicaid_number = Column(String(50))
    ssn = Column(String(20))
    employer_name = Column(String(255))
    employer_address = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )

    insurances = relationship("CPTPatientInsurance", back_populates="patient", cascade="all, delete-orphan")


class CPTPatientInsurance(Base):
    __tablename__ = "cpt_patient_insurances"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    patient_id = Column(BigInteger, ForeignKey("cpt_patients.id", ondelete="CASCADE"), nullable=False)
    plan_name = Column(String(255))
    policy_id = Column(String(100))
    coverage_type = Column(coverage_type_enum, default="Medical")
    expiration_date = Column(Date)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )

    patient = relationship("CPTPatient", back_populates="insurances")


@st.cache_resource
def get_engine_and_sessionmaker():
    db_url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        st.stop()  # hard stop with a message below
    engine = create_engine(db_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
    return engine, SessionLocal


def get_session(SessionLocal):
    return SessionLocal()


def to_date(opt_date) -> Optional[dt.date]:
    """Helper to accept either None or a datetime.date from date_input."""
    return opt_date if isinstance(opt_date, dt.date) else None


def success_and_clear(message: str):
    st.success(message)
    st.experimental_rerun()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="CPT Data Entry", page_icon="📝", layout="wide")
st.title("📝 CPT Data Entry")

engine_session = None
try:
    engine, SessionLocal = get_engine_and_sessionmaker()
    engine_session = True
except Exception as e:
    st.error(
        "Could not connect to the database. Please set `DATABASE_URL` in "
        ".streamlit/secrets.toml or as an environment variable."
    )
    st.exception(e)

if not engine_session:
    st.stop()

menu = st.sidebar.radio("Go to", ["Providers", "Patients", "Patient Insurances", "Browse / Verify Data"])

# ----------------------------
# Providers Form
# ----------------------------
if menu == "Providers":
    st.header("Add Provider")
    with st.form("provider_form", clear_on_submit=True):
        name = st.text_input("Name*", max_chars=255)
        npi = st.text_input("NPI", max_chars=50)
        license_number = st.text_input("License Number", max_chars=50)
        dea_number = st.text_input("DEA Number", max_chars=50)
        medicare_number = st.text_input("Medicare Number", max_chars=50)
        medicaid_number = st.text_input("Medicaid Number", max_chars=50)
        etin = st.text_input("ETIN", max_chars=50)
        ein_tin = st.text_input("EIN/TIN", max_chars=50)
        upin_number = st.text_input("UPIN Number", max_chars=50)
        clia_number = st.text_input("CLIA Number", max_chars=50)

        submitted = st.form_submit_button("Save Provider")
        if submitted:
            if not name.strip():
                st.error("Name is required.")
            else:
                session = get_session(SessionLocal)
                try:
                    rec = CPTProvider(
                        name=name.strip(),
                        npi=npi or None,
                        license_number=license_number or None,
                        dea_number=dea_number or None,
                        medicare_number=medicare_number or None,
                        medicaid_number=medicaid_number or None,
                        etin=etin or None,
                        ein_tin=ein_tin or None,
                        upin_number=upin_number or None,
                        clia_number=clia_number or None,
                    )
                    session.add(rec)
                    session.commit()
                    session.close()
                    success_and_clear("Provider added!")
                except Exception as ex:
                    session.rollback()
                    session.close()
                    st.error("Failed to insert provider.")
                    st.exception(ex)

    st.subheader("Providers (latest 100)")
    session = get_session(SessionLocal)
    providers = session.query(CPTProvider).order_by(CPTProvider.id.desc()).limit(100).all()
    session.close()
    if providers:
        st.dataframe(
            [
                {
                    "id": p.id,
                    "name": p.name,
                    "npi": p.npi,
                    "updated_at": p.updated_at
                }
                for p in providers
            ],
            use_container_width=True
        )
    else:
        st.info("No providers yet.")

# ----------------------------
# Patients Form
# ----------------------------
elif menu == "Patients":
    st.header("Add Patient")
    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("Name*", max_chars=255)
        dob = st.date_input("DOB", value=dt.date(1990, 1, 1))
        address = st.text_area("Address")
        phone = st.text_input("Phone", max_chars=50)
        gender_at_birth = st.text_input("Gender at Birth", max_chars=50)
        gender_identity = st.text_input("Gender Identity", max_chars=50)
        pronouns = st.text_input("Pronouns", max_chars=50)
        marital_status = st.text_input("Marital Status", max_chars=50)
        race = st.text_input("Race", max_chars=50)
        language = st.text_input("Language", max_chars=50)
        religion = st.text_input("Religion", max_chars=50)
        medicare_number = st.text_input("Medicare Number", max_chars=50)
        medicaid_number = st.text_input("Medicaid Number", max_chars=50)
        ssn = st.text_input("SSN", max_chars=20, help="Store securely per your compliance policies.")
        employer_name = st.text_input("Employer Name", max_chars=255)
        employer_address = st.text_area("Employer Address")

        submitted = st.form_submit_button("Save Patient")
        if submitted:
            if not name.strip():
                st.error("Name is required.")
            else:
                session = get_session(SessionLocal)
                try:
                    rec = CPTPatient(
                        name=name.strip(),
                        dob=to_date(dob),
                        address=address or None,
                        phone=phone or None,
                        gender_at_birth=gender_at_birth or None,
                        gender_identity=gender_identity or None,
                        pronouns=pronouns or None,
                        marital_status=marital_status or None,
                        race=race or None,
                        language=language or None,
                        religion=religion or None,
                        medicare_number=medicare_number or None,
                        medicaid_number=medicaid_number or None,
                        ssn=ssn or None,
                        employer_name=employer_name or None,
                        employer_address=employer_address or None,
                    )
                    session.add(rec)
                    session.commit()
                    session.close()
                    success_and_clear("Patient added!")
                except Exception as ex:
                    session.rollback()
                    session.close()
                    st.error("Failed to insert patient.")
                    st.exception(ex)

    st.subheader("Patients (latest 100)")
    session = get_session(SessionLocal)
    patients = session.query(CPTPatient).order_by(CPTPatient.id.desc()).limit(100).all()
    session.close()
    if patients:
        st.dataframe(
            [
                {
                    "id": p.id,
                    "name": p.name,
                    "dob": p.dob,
                    "phone": p.phone,
                    "updated_at": p.updated_at,
                }
                for p in patients
            ],
            use_container_width=True
        )
    else:
        st.info("No patients yet.")

# ----------------------------
# Patient Insurances Form
# ----------------------------
elif menu == "Patient Insurances":
    st.header("Add Patient Insurance")

    session = get_session(SessionLocal)
    pts = session.query(CPTPatient.id, CPTPatient.name).order_by(CPTPatient.name.asc()).all()
    session.close()

    if not pts:
        st.warning("No patients found. Please add a patient first.")
    else:
        patient_map = {f"{p.name} (#{p.id})": p.id for p in pts}
        with st.form("patient_insurance_form", clear_on_submit=True):
            patient_key = st.selectbox("Patient*", list(patient_map.keys()))
            plan_name = st.text_input("Plan Name", max_chars=255)
            policy_id = st.text_input("Policy ID", max_chars=100)
            coverage_type = st.selectbox("Coverage Type", ["Medical", "Other"])
            expiration_date = st.date_input("Expiration Date", value=dt.date.today())

            submitted = st.form_submit_button("Save Insurance")
            if submitted:
                session = get_session(SessionLocal)
                try:
                    rec = CPTPatientInsurance(
                        patient_id=patient_map[patient_key],
                        plan_name=plan_name or None,
                        policy_id=policy_id or None,
                        coverage_type=coverage_type or "Medical",
                        expiration_date=to_date(expiration_date),
                    )
                    session.add(rec)
                    session.commit()
                    session.close()
                    success_and_clear("Patient insurance added!")
                except Exception as ex:
                    session.rollback()
                    session.close()
                    st.error("Failed to insert insurance.")
                    st.exception(ex)

    st.subheader("Patient Insurances (latest 100)")
    session = get_session(SessionLocal)
    ins = (
        session.query(
            CPTPatientInsurance.id,
            CPTPatient.name,
            CPTPatientInsurance.plan_name,
            CPTPatientInsurance.policy_id,
            CPTPatientInsurance.coverage_type,
            CPTPatientInsurance.expiration_date,
            CPTPatientInsurance.updated_at
        )
        .join(CPTPatient, CPTPatientInsurance.patient_id == CPTPatient.id)
        .order_by(CPTPatientInsurance.id.desc())
        .limit(100)
        .all()
    )
    session.close()

    if ins:
        st.dataframe(
            [
                {
                    "id": r.id,
                    "patient": r.name,
                    "plan_name": r.plan_name,
                    "policy_id": r.policy_id,
                    "coverage_type": r.coverage_type,
                    "expiration_date": r.expiration_date,
                    "updated_at": r.updated_at,
                }
                for r in ins
            ],
            use_container_width=True
        )
    else:
        st.info("No patient insurances yet.")

# ----------------------------
# Browse / Verify
# ----------------------------
elif menu == "Browse / Verify Data":
    st.header("Quick Browse")
    which = st.selectbox("Choose a table", ["Providers", "Patients", "Patient Insurances"])

    session = get_session(SessionLocal)
    if which == "Providers":
        rows = session.query(CPTProvider).order_by(CPTProvider.id.desc()).limit(1000).all()
        data = [
            {
                "id": r.id, "name": r.name, "npi": r.npi, "medicare_number": r.medicare_number,
                "medicaid_number": r.medicaid_number, "updated_at": r.updated_at
            } for r in rows
        ]
    elif which == "Patients":
        rows = session.query(CPTPatient).order_by(CPTPatient.id.desc()).limit(1000).all()
        data = [
            {
                "id": r.id, "name": r.name, "dob": r.dob, "phone": r.phone, "ssn": r.ssn,
                "medicare_number": r.medicare_number, "medicaid_number": r.medicaid_number,
                "updated_at": r.updated_at
            } for r in rows
        ]
    else:
        rows = (
            session.query(
                CPTPatientInsurance.id,
                CPTPatient.name,
                CPTPatientInsurance.plan_name,
                CPTPatientInsurance.policy_id,
                CPTPatientInsurance.coverage_type,
                CPTPatientInsurance.expiration_date,
                CPTPatientInsurance.updated_at
            )
            .join(CPTPatient, CPTPatientInsurance.patient_id == CPTPatient.id)
            .order_by(CPTPatientInsurance.id.desc()).limit(1000).all()
        )
        data = [
            {
                "id": r.id, "patient": r.name, "plan_name": r.plan_name, "policy_id": r.policy_id,
                "coverage_type": r.coverage_type, "expiration_date": r.expiration_date,
                "updated_at": r.updated_at
            } for r in rows
        ]
    session.close()

    if data:
        st.dataframe(data, use_container_width=True)
    else:
        st.info("No rows yet.")
