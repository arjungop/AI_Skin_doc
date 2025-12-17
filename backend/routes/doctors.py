from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from backend.database import get_db
from backend import models, schemas, crud
from backend.security import require_roles

router = APIRouter()


@router.get("/", response_model=list[schemas.DoctorOut])
def list_doctors(q: str | None = None, db: Session = Depends(get_db), _=Depends(require_roles("ADMIN", "PATIENT", "DOCTOR"))):
    query = db.query(models.Doctor).join(models.User, models.User.user_id == models.Doctor.user_id)
    # Only approved doctors exist in doctors table
    if q:
        like = f"%{q}%"
        query = query.filter(or_(models.User.username.ilike(like), models.Doctor.specialization.ilike(like)))
    rows = query.all()
    
    # Build response with proper user data
    result = []
    for doctor in rows:
        doctor_data = {
            "doctor_id": doctor.doctor_id,
            "user_id": doctor.user_id,
            "username": doctor.user.username,
            "email": doctor.user.email,
            "specialization": doctor.specialization
        }
        result.append(doctor_data)
    
    return result


@router.post("/apply", response_model=schemas.DoctorApplicationOut)
def apply_doctor(data: schemas.DoctorApply, db: Session = Depends(get_db)):
    # Create user with role PENDING_DOCTOR
    existing = db.query(models.User).filter(models.User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use")
    # derive username from names or email local part
    base = f"{data.first_name}.{data.last_name}".lower()
    username = base
    if db.query(models.User).filter(models.User.username == username).first():
        username = (data.email.split('@')[0]).lower()
        if db.query(models.User).filter(models.User.username == username).first():
            username = crud._unique_username(db, base or username)
    user = models.User(username=username, email=data.email, hashed_password=crud.hash_password(data.password), role="PENDING_DOCTOR")
    db.add(user)
    db.commit()
    db.refresh(user)

    app = models.DoctorApplication(
        user_id=user.user_id,
        first_name=data.first_name,
        last_name=data.last_name,
        specialization=data.specialization,
        license_no=data.license_no,
        hospital=data.department,  # store department in 'hospital' column
        status="PENDING",
    )
    db.add(app)
    db.commit()
    db.refresh(app)

    # Shape response
    app.username = user.username  # type: ignore
    app.email = user.email  # type: ignore
    app.first_name = data.first_name  # type: ignore
    app.last_name = data.last_name  # type: ignore
    return app


@router.get("/{doctor_id}/availability", response_model=list[schemas.AvailabilityOut])
def list_availability(doctor_id: int, db: Session = Depends(get_db)):
    return db.query(models.DoctorAvailability).filter(models.DoctorAvailability.doctor_id == doctor_id).all()


@router.post("/{doctor_id}/availability", response_model=list[schemas.AvailabilityOut])
def set_availability(
    doctor_id: int,
    items: list[schemas.AvailabilityItem],
    db: Session = Depends(get_db),
    user = Depends(require_roles("DOCTOR", "ADMIN")),
):
    # Only the owning doctor or admin can set
    if (user.role or "").upper() == "DOCTOR":
        doc = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
        if not doc or doc.doctor_id != doctor_id:
            raise HTTPException(status_code=403, detail="Forbidden")

    # Replace existing availability
    db.query(models.DoctorAvailability).filter(models.DoctorAvailability.doctor_id == doctor_id).delete()
    out = []
    for it in items:
        row = models.DoctorAvailability(
            doctor_id=doctor_id,
            weekday=it.weekday,
            start_time=it.start_time,
            end_time=it.end_time,
            timezone=it.timezone or "local",
        )
        db.add(row)
        db.flush()
        out.append(row)
    db.commit()
    return out


@router.get("/me/profile")
def get_my_profile(db: Session = Depends(get_db), user = Depends(require_roles("DOCTOR", "ADMIN"))):
    if (user.role or '').upper() == 'DOCTOR':
        doc = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Doctor not found")
        prof = db.query(models.DoctorProfile).filter(models.DoctorProfile.doctor_id == doc.doctor_id).first()
        return {
            "doctor_id": doc.doctor_id,
            "bio": prof.bio if prof else None,
            "visibility": (prof.visibility if prof else 'true'),
            "specialization": doc.specialization,
        }
    # Admin: require doctor_id param
    raise HTTPException(status_code=400, detail="Admin must use /doctors/{doctor_id}/profile")


@router.post("/me/profile")
def set_my_profile(body: dict, db: Session = Depends(get_db), user = Depends(require_roles("DOCTOR",))):
    doc = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    prof = db.query(models.DoctorProfile).filter(models.DoctorProfile.doctor_id == doc.doctor_id).first()
    if not prof:
        prof = models.DoctorProfile(doctor_id=doc.doctor_id)
        db.add(prof)
    bio = (body.get('bio') or '').strip() if isinstance(body.get('bio'), str) else None
    vis = body.get('visibility')
    if bio is not None:
        prof.bio = bio
    if vis is not None:
        prof.visibility = 'true' if (str(vis).lower() in {'1','true','yes','on'}) else 'false'
    db.commit(); db.refresh(prof)
    return {"ok": True}
