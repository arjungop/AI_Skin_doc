from sqlalchemy.orm import Session
from passlib.context import CryptContext
from . import models, schemas
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

# -----------------------
# Patient
# -----------------------
def _unique_username(db: Session, base: str) -> str:
    base = base.strip().lower().replace(' ', '.') or 'user'
    username = base
    i = 1
    while db.query(models.User).filter(models.User.username == username).first():
        i += 1
        username = f"{base}{i}"
    return username


def create_patient(db: Session, patient: schemas.PatientCreate):
    hashed = hash_password(patient.password)
    # Auto-generate username if not provided
    uname = patient.username or f"{patient.first_name}.{patient.last_name}".lower()
    if not uname or db.query(models.User).filter(models.User.username == uname).first():
        uname = _unique_username(db, uname or patient.email.split('@')[0])
    new_user = models.User(
        username=uname,
        email=patient.email,
        hashed_password=hashed,
        role="PATIENT"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    new_patient = models.Patient(
        user_id=new_user.user_id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        age=patient.age,
        gender=patient.gender
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    # Attach user info for response
    new_patient.username = new_user.username
    new_patient.email = new_user.email
    new_patient.role = new_user.role
    return new_patient

def authenticate_patient(db: Session, email: str, password: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
    if patient:
        patient.username = user.username
        patient.email = user.email
        patient.role = user.role
    return patient

# -----------------------
# Appointments
# -----------------------
def create_appointment(db: Session, appointment: schemas.AppointmentCreate):
    new_app = models.Appointment(
        patient_id=appointment.patient_id,
        doctor_id=appointment.doctor_id,
        appointment_date=appointment.appointment_date,
        reason=appointment.reason,
        status="Scheduled"
    )
    db.add(new_app)
    db.commit()
    db.refresh(new_app)
    return new_app

# -----------------------
# Lesions
# -----------------------
def create_lesion(db: Session, lesion: schemas.LesionCreate):
    new_lesion = models.Lesion(
        patient_id=lesion.patient_id,
        image_path=lesion.image_path,
        prediction=lesion.prediction,
        created_at=datetime.utcnow(),
    )
    db.add(new_lesion)
    db.commit()
    db.refresh(new_lesion)
    return new_lesion
