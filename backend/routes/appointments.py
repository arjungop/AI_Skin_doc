from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import crud, schemas, models
from backend.security import require_roles, get_current_user
from pydantic import BaseModel
from backend.notify import NotificationHub
import os
from datetime import datetime, timedelta

router = APIRouter()


@router.get("/booked-slots")
def get_booked_slots(
    doctor_id: int,
    date: str,  # YYYY-MM-DD
    db: Session = Depends(get_db),
    _user: models.User = Depends(get_current_user),
):
    """Return list of HH:MM strings for slots already booked on the given date."""
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    day_start = day
    day_end = day + timedelta(days=1)

    appointments = (
        db.query(models.Appointment)
        .filter(
            models.Appointment.doctor_id == doctor_id,
            models.Appointment.status.in_(["Scheduled", "Confirmed"]),
            models.Appointment.appointment_date >= day_start,
            models.Appointment.appointment_date < day_end,
        )
        .all()
    )

    duration_min = int(os.getenv("APPOINTMENT_DURATION_MINUTES", "30"))
    booked: list[str] = []
    for appt in appointments:
        t = appt.appointment_date
        end = t + timedelta(minutes=duration_min)
        cur = t
        while cur < end:
            booked.append(cur.strftime("%H:%M"))
            cur += timedelta(minutes=30)

    return booked

# Accept both with and without trailing slash
@router.post("/", response_model=schemas.AppointmentOut)
@router.post("", response_model=schemas.AppointmentOut)
def create_appointment(
    data: schemas.AppointmentCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(require_roles("PATIENT", "ADMIN")),
):
    # Patients may only create for themselves
    role = (user.role or "").upper()
    if role == "PATIENT":
        # find patient's id
        patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
        if not patient or patient.patient_id != data.patient_id:
            raise HTTPException(status_code=403, detail="Cannot create for another patient")
    # Conflict checks
    appt_dt = data.appointment_date
    weekday = appt_dt.weekday()
    avail = db.query(models.DoctorAvailability).filter(
        models.DoctorAvailability.doctor_id == data.doctor_id,
        models.DoctorAvailability.weekday == weekday,
    ).all()
    hhmm = appt_dt.strftime("%H:%M")
    within = any(a.start_time <= hhmm < a.end_time for a in avail)
    if not within:
        raise HTTPException(status_code=400, detail="Selected time is outside doctor's availability")

    duration_min = int(os.getenv("APPOINTMENT_DURATION_MINUTES", "30"))
    end_dt = appt_dt + timedelta(minutes=duration_min)

    # Doctor conflict check
    existing = db.query(models.Appointment).filter(
        models.Appointment.doctor_id == data.doctor_id,
        models.Appointment.status.in_(["Scheduled", "Confirmed"]),
        models.Appointment.appointment_date < end_dt,
    ).all()
    for e in existing:
        e_end = e.appointment_date + timedelta(minutes=duration_min)
        if not (e_end <= appt_dt or e.appointment_date >= end_dt):
            raise HTTPException(status_code=400, detail="Time slot already booked")

    # Patient conflict check — prevent overlapping appointments for the same patient
    patient_existing = db.query(models.Appointment).filter(
        models.Appointment.patient_id == data.patient_id,
        models.Appointment.status.in_(["Scheduled", "Confirmed"]),
        models.Appointment.appointment_date < end_dt,
    ).all()
    for e in patient_existing:
        e_end = e.appointment_date + timedelta(minutes=duration_min)
        if not (e_end <= appt_dt or e.appointment_date >= end_dt):
            raise HTTPException(status_code=400, detail="You already have an appointment at this time")

    return crud.create_appointment(db, data)

@router.get("/", response_model=list[schemas.AppointmentOut])
@router.get("", response_model=list[schemas.AppointmentOut])
def get_appointments(db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    role = (user.role or "").upper()
    q = db.query(models.Appointment)
    if role == "ADMIN":
        return q.all()
    if role == "DOCTOR":
        doctor = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
        if not doctor:
            return []
        return q.filter(models.Appointment.doctor_id == doctor.doctor_id).all()
    # default PATIENT
    patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
    if not patient:
        return []
    return q.filter(models.Appointment.patient_id == patient.patient_id).all()


class AppointmentStatusUpdate(BaseModel):
    status: str


@router.patch("/{appointment_id}/status", response_model=schemas.AppointmentOut)
def update_status(
    appointment_id: int,
    body: AppointmentStatusUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    app = db.query(models.Appointment).filter(models.Appointment.appointment_id == appointment_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Appointment not found")

    new_status = (body.status or "").capitalize()
    if new_status not in {"Scheduled", "Confirmed", "Completed", "Cancelled"}:
        raise HTTPException(status_code=400, detail="Invalid status")

    role = (user.role or "").upper()

    # Authorization and simple transition rules
    if role == "ADMIN":
        app.status = new_status
    elif role == "DOCTOR":
        doctor = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
        if not doctor or doctor.doctor_id != app.doctor_id:
            raise HTTPException(status_code=403, detail="Not your appointment")
        if new_status == "Confirmed" and app.status == "Scheduled":
            app.status = new_status
            # Auto-create chat room for doctor-patient pair if not exists
            exists = db.query(models.ChatRoom).filter(
                models.ChatRoom.patient_id == app.patient_id,
                models.ChatRoom.doctor_id == app.doctor_id,
            ).first()
            if not exists:
                room = models.ChatRoom(patient_id=app.patient_id, doctor_id=app.doctor_id)
                db.add(room)
        elif new_status == "Completed" and app.status in {"Scheduled", "Confirmed"}:
            app.status = new_status
        elif new_status == "Cancelled" and app.status in {"Scheduled", "Confirmed"}:
            app.status = new_status
        else:
            raise HTTPException(status_code=400, detail="Invalid transition for doctor")
    else:  # PATIENT
        patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
        if not patient or patient.patient_id != app.patient_id:
            raise HTTPException(status_code=403, detail="Not your appointment")
        if new_status == "Cancelled" and app.status in {"Scheduled", "Confirmed"}:
            app.status = new_status
        else:
            raise HTTPException(status_code=400, detail="Invalid transition for patient")

    db.add(app)
    db.commit()
    db.refresh(app)
    try:
        # Notify both doctor and patient users
        pat = db.query(models.Patient).filter(models.Patient.patient_id == app.patient_id).first()
        doc = db.query(models.Doctor).filter(models.Doctor.doctor_id == app.doctor_id).first()
        user_ids = set()
        if pat:
            user_ids.add(pat.user_id)
        if doc:
            user_ids.add(doc.user_id)
        NotificationHub.send_many(user_ids, 'appointment_status', {
            'appointment_id': app.appointment_id,
            'status': app.status,
        })
    except Exception:
        pass
    return app
