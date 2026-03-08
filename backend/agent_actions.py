"""
Agent action executor – runs approved actions against the database.

Each action_type maps to a concrete DB operation (create appointment,
add routine items, log skin notes, etc.).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from backend import models

_log = logging.getLogger(__name__)


def execute_action(action: models.AgentAction, db: Session) -> str:
    """Execute a single approved AgentAction. Returns a result message."""
    payload = json.loads(action.payload or "{}")

    # Resolve patient → user_id for models that need it
    session = db.query(models.AgentSession).filter(
        models.AgentSession.session_id == action.session_id
    ).first()
    if not session:
        return "Session not found."
    patient = db.query(models.Patient).filter(
        models.Patient.patient_id == session.patient_id
    ).first()
    if not patient:
        return "Patient not found."

    executors = {
        "schedule_appointment": _exec_schedule_appointment,
        "create_routine": _exec_create_routine,
        "add_skin_log": _exec_add_skin_log,
        "set_reminder": _exec_set_reminder,
    }

    fn = executors.get(action.action_type)
    if not fn:
        return f"Unknown action type: {action.action_type}"

    return fn(patient, db, payload)


# ── Individual executors ──────────────────────────────────────────────────

def _exec_schedule_appointment(
    patient: models.Patient, db: Session, payload: dict
) -> str:
    reason = payload.get("reason", "Dermatology consultation")
    urgency = payload.get("urgency", "routine")

    # Find any doctor linked to patient, or first available doctor
    doctor = (
        db.query(models.Doctor)
        .join(models.DoctorPatient)
        .filter(models.DoctorPatient.patient_id == patient.patient_id)
        .first()
    )
    if not doctor:
        doctor = db.query(models.Doctor).first()
    if not doctor:
        return "No doctors available in the system. Please contact the clinic directly."

    # Schedule 3 days out for routine, tomorrow for urgent
    days_ahead = 1 if urgency == "urgent" else 3
    appt_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)

    appt = models.Appointment(
        patient_id=patient.patient_id,
        doctor_id=doctor.doctor_id,
        appointment_date=appt_date,
        reason=reason,
        status="Scheduled",
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)

    return (
        f"Appointment #{appt.appointment_id} scheduled with Dr. {doctor.user.username if doctor.user else 'TBD'} "
        f"on {appt_date.strftime('%b %d, %Y')} for: {reason}"
    )


def _exec_create_routine(
    patient: models.Patient, db: Session, payload: dict
) -> str:
    steps = payload.get("steps", [])
    if not steps:
        return "No routine steps provided."

    created = []
    for s in steps:
        item = models.RoutineItem(
            user_id=patient.user_id,
            product_name=s.get("product", "Unknown product"),
            time_of_day=s.get("time", "AM"),
            step_order=s.get("order", 1),
            is_active=True,
        )
        db.add(item)
        created.append(f"{item.time_of_day}: {item.product_name}")

    db.commit()
    return f"Created {len(created)} routine items: " + ", ".join(created)


def _exec_add_skin_log(
    patient: models.Patient, db: Session, payload: dict
) -> str:
    notes = payload.get("notes", "")
    tags = payload.get("tags", "")

    log = models.SkinLog(
        user_id=patient.user_id,
        notes=notes,
        tags=tags,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return f"Skin log #{log.log_id} created: {notes[:100]}"


def _exec_set_reminder(
    patient: models.Patient, db: Session, payload: dict
) -> str:
    reminder = payload.get("reminder", "Health reminder")
    frequency = payload.get("frequency", "once")

    # Store as a skin log with a "reminder" tag for now
    tag_list = json.dumps(["reminder", frequency])
    log = models.SkinLog(
        user_id=patient.user_id,
        notes=f"[Reminder - {frequency}] {reminder}",
        tags=tag_list,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return f"Reminder set (log #{log.log_id}): {reminder} ({frequency})"
