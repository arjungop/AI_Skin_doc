from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from backend.database import get_db
from backend import models, schemas, crud
from backend.security import require_roles

router = APIRouter()


@router.get("/")
def list_doctors(
    q: str | None = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _=Depends(require_roles("ADMIN", "PATIENT", "DOCTOR")),
):
    page_size = min(page_size, 100)
    query = db.query(models.Doctor).join(models.User, models.User.user_id == models.Doctor.user_id)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(models.User.username.ilike(like), models.Doctor.specialization.ilike(like)))
    total = query.count()
    rows = query.offset(max(0, (page - 1) * page_size)).limit(page_size).all()
    
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
    
    return {"items": result, "page": page, "page_size": page_size, "total": total}


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

@router.get("/me/patients", response_model=list[schemas.DoctorPatientListOut])
def get_my_patients(q: str | None = None, db: Session = Depends(get_db), user = Depends(require_roles("DOCTOR"))):
    doc = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    query = db.query(
        models.Patient,
        models.User,
        func.count(models.Appointment.appointment_id).label("visits"),
        func.max(models.Appointment.appointment_date).label("lastVisit")
    ).join(
        models.DoctorPatient, models.DoctorPatient.patient_id == models.Patient.patient_id
    ).join(
        models.User, models.User.user_id == models.Patient.user_id
    ).outerjoin(
        models.Appointment, 
        (models.Appointment.patient_id == models.Patient.patient_id) & 
        (models.Appointment.doctor_id == doc.doctor_id) &
        (models.Appointment.status != 'Cancelled')
    ).filter(
        models.DoctorPatient.doctor_id == doc.doctor_id
    ).group_by(
        models.Patient.patient_id, models.User.user_id
    )

    if q:
        like = f"%{q}%"
        query = query.filter(or_(
            models.User.username.ilike(like),
            models.Patient.first_name.ilike(like),
            models.Patient.last_name.ilike(like)
        ))
        
    results = query.all()
    out = []
    for pt, u, visits, last_visit in results:
        name = f"{pt.first_name} {pt.last_name}".strip() or u.username
        out.append({
            "patient_id": pt.patient_id,
            "name": name,
            "visits": visits,
            "lastVisit": last_visit
        })
        
    out.sort(key=lambda x: x["lastVisit"].isoformat() if x["lastVisit"] else "", reverse=True)
    return out


@router.get("/patients/{patient_id}/overview")
def get_patient_overview(
    patient_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_roles("DOCTOR")),
):
    """Full patient context card for the doctor — demographics, skin profile, scans, AI routine, plans."""
    doc = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Access guard — only the patient's doctor can view
    link = db.query(models.DoctorPatient).filter(
        models.DoctorPatient.doctor_id == doc.doctor_id,
        models.DoctorPatient.patient_id == patient_id,
    ).first()
    if not link:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    pt = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not pt:
        raise HTTPException(status_code=404, detail="Patient not found")

    skin = db.query(models.UserProfile).filter(models.UserProfile.user_id == pt.user_id).first()

    lesions = (
        db.query(models.Lesion)
        .filter(models.Lesion.patient_id == patient_id)
        .order_by(models.Lesion.created_at.desc())
        .limit(6)
        .all()
    )

    routine = (
        db.query(models.RoutineItem)
        .filter(
            models.RoutineItem.user_id == pt.user_id,
            models.RoutineItem.is_active == True,
        )
        .order_by(models.RoutineItem.time_of_day, models.RoutineItem.step_order)
        .all()
    )

    plans = (
        db.query(models.TreatmentPlan)
        .filter(
            models.TreatmentPlan.patient_id == patient_id,
            models.TreatmentPlan.status == "active",
        )
        .order_by(models.TreatmentPlan.created_at.desc())
        .all()
    )

    return {
        "patient": {
            "patient_id": pt.patient_id,
            "user_id": pt.user_id,
            "name": f"{pt.first_name} {pt.last_name}".strip(),
            "age": pt.age,
            "gender": pt.gender,
            "email": pt.user.email if pt.user else None,
        },
        "skin_profile": {
            "skin_type": skin.skin_type,
            "sensitivity_level": skin.sensitivity_level,
            "acne_prone": skin.acne_prone,
            "fitzpatrick_type": skin.fitzpatrick_type,
            "allergies": skin.allergies,
            "goals": skin.goals,
            "location_city": skin.location_city,
        } if skin else None,
        "recent_scans": [
            {
                "lesion_id": l.lesion_id,
                "prediction": l.prediction,
                "created_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in lesions
        ],
        "ai_routine": [
            {
                "step": r.step_order,
                "time": r.time_of_day,
                "product": r.product_name,
                "brand": r.brand or "",
                "instructions": r.notes or "",
            }
            for r in routine
        ],
        "active_plans": [
            {
                "plan_id": p.plan_id,
                "diagnosis": p.diagnosis,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in plans
        ],
    }
