from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from backend.database import get_db
from backend import models, crud, schemas
from backend.security import require_roles

router = APIRouter()

@router.post("/register", response_model=schemas.PatientOut)
def register_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    # Check duplicates
    existing = db.query(models.User).filter(
        or_(models.User.email == patient.email, models.User.username == patient.username)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    return crud.create_patient(db, patient)

@router.post("/login", response_model=schemas.PatientOut)
def login_patient(data: schemas.PatientLogin, db: Session = Depends(get_db)):
    patient = crud.authenticate_patient(db, data.email, data.password)
    if not patient:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return patient


@router.get("/", response_model=list[schemas.PatientOut])
def list_patients(q: str | None = None, db: Session = Depends(get_db), _=Depends(require_roles("ADMIN", "DOCTOR"))):
    # Admin and Doctor can list patients
    query = db.query(models.Patient).join(models.User, models.User.user_id == models.Patient.user_id)
    
    if q:
        q_like = f"%{q}%"
        if q.isdigit():
            query = query.filter(
                (models.Patient.patient_id == int(q)) |
                (models.User.user_id == int(q)) |
                (models.User.username.ilike(q_like)) |
                (models.User.email.ilike(q_like)) |
                (models.Patient.first_name.ilike(q_like)) |
                (models.Patient.last_name.ilike(q_like))
            )
        else:
            query = query.filter(
                (models.User.username.ilike(q_like)) |
                (models.User.email.ilike(q_like)) |
                (models.Patient.first_name.ilike(q_like)) |
                (models.Patient.last_name.ilike(q_like))
            )
    
    patients = query.limit(50).all()
    
    # Build response with proper user data
    result = []
    for patient in patients:
        patient_data = {
            "patient_id": patient.patient_id,
            "user_id": patient.user_id,
            "username": patient.user.username,
            "email": patient.user.email,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "age": patient.age,
            "gender": patient.gender,
            "role": patient.user.role or "PATIENT",
        }
        result.append(patient_data)
    
    return result
