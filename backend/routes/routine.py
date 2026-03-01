"""
Treatment Plan Adherence System.

Replaces the generic skincare routine tracker with a medical-grade system
where doctors prescribe treatment plans and patients track adherence.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user
from datetime import datetime
from typing import List

router = APIRouter()


# ── View plans — role-aware ───────────────────────────────────────────

@router.get("/", response_model=List[schemas.TreatmentPlanOut])
def get_my_plans(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """
    Get treatment plans for the current user.
    - Patients see their own plans.
    - Doctors see plans they created.
    - Admins see all plans.
    """
    role = (user.role or "").upper()

    if role == "ADMIN":
        plans = (
            db.query(models.TreatmentPlan)
            .options(joinedload(models.TreatmentPlan.steps))
            .order_by(models.TreatmentPlan.created_at.desc())
            .all()
        )
        return plans

    # Doctor — return plans they created
    doctor = db.query(models.Doctor).filter(
        models.Doctor.user_id == user.user_id
    ).first()
    if doctor:
        plans = (
            db.query(models.TreatmentPlan)
            .options(joinedload(models.TreatmentPlan.steps))
            .filter(models.TreatmentPlan.doctor_id == doctor.doctor_id)
            .order_by(models.TreatmentPlan.created_at.desc())
            .all()
        )
        return plans

    # Patient — return their own plans
    patient = db.query(models.Patient).filter(
        models.Patient.user_id == user.user_id
    ).first()
    if not patient:
        return []

    plans = (
        db.query(models.TreatmentPlan)
        .options(joinedload(models.TreatmentPlan.steps))
        .filter(models.TreatmentPlan.patient_id == patient.patient_id)
        .order_by(models.TreatmentPlan.created_at.desc())
        .all()
    )
    return plans


# ── Doctor: create a plan ────────────────────────────────────────────────

@router.post("/plans", response_model=schemas.TreatmentPlanOut)
def create_plan(
    data: schemas.TreatmentPlanCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Doctor creates a treatment plan for a patient."""
    role = (user.role or "").upper()
    if role not in ("DOCTOR", "ADMIN"):
        raise HTTPException(status_code=403, detail="Only doctors can create treatment plans")

    # Resolve doctor_id from user
    doctor = db.query(models.Doctor).filter(
        models.Doctor.user_id == user.user_id
    ).first()
    if not doctor and role != "ADMIN":
        raise HTTPException(status_code=403, detail="Doctor profile not found")

    doctor_id = doctor.doctor_id if doctor else data.doctor_id

    # Verify patient exists
    patient = db.query(models.Patient).filter(
        models.Patient.patient_id == data.patient_id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    plan = models.TreatmentPlan(
        patient_id=data.patient_id,
        doctor_id=doctor_id,
        diagnosis=data.diagnosis,
        status="active",
        notes=data.notes,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


# ── Doctor: add steps to a plan ──────────────────────────────────────────

@router.post("/plans/{plan_id}/steps", response_model=schemas.TreatmentStepOut)
def add_step(
    plan_id: int,
    data: schemas.TreatmentStepCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Doctor adds a medication step to a treatment plan."""
    role = (user.role or "").upper()
    if role not in ("DOCTOR", "ADMIN"):
        raise HTTPException(status_code=403, detail="Only doctors can modify treatment plans")

    plan = db.query(models.TreatmentPlan).filter(
        models.TreatmentPlan.plan_id == plan_id
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Treatment plan not found")

    step = models.TreatmentStep(
        plan_id=plan_id,
        medication_name=data.medication_name,
        dosage=data.dosage,
        frequency=data.frequency or "daily",
        time_of_day=data.time_of_day or "PM",
        instructions=data.instructions,
        step_order=data.step_order or 1,
    )
    db.add(step)
    db.commit()
    db.refresh(step)
    return step


# ── Get steps for a plan ─────────────────────────────────────────────────

@router.get("/plans/{plan_id}/steps", response_model=List[schemas.TreatmentStepOut])
def get_steps(
    plan_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Get all steps in a treatment plan."""
    plan = db.query(models.TreatmentPlan).filter(
        models.TreatmentPlan.plan_id == plan_id
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Verify access: patient or their doctor or admin
    role = (user.role or "").upper()
    if role != "ADMIN":
        patient = db.query(models.Patient).filter(
            models.Patient.user_id == user.user_id
        ).first()
        doctor = db.query(models.Doctor).filter(
            models.Doctor.user_id == user.user_id
        ).first()
        if not (
            (patient and patient.patient_id == plan.patient_id)
            or (doctor and doctor.doctor_id == plan.doctor_id)
        ):
            raise HTTPException(status_code=403, detail="Access denied")

    steps = (
        db.query(models.TreatmentStep)
        .filter(models.TreatmentStep.plan_id == plan_id, models.TreatmentStep.is_active == True)
        .order_by(models.TreatmentStep.step_order)
        .all()
    )
    return steps


# ── Patient: record adherence ────────────────────────────────────────────

@router.post("/plans/{plan_id}/adherence", response_model=schemas.TreatmentAdherenceOut)
def record_adherence(
    plan_id: int,
    data: schemas.TreatmentAdherenceCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Patient records taking/skipping a medication step."""
    # Verify plan belongs to patient
    patient = db.query(models.Patient).filter(
        models.Patient.user_id == user.user_id
    ).first()
    if not patient:
        raise HTTPException(status_code=403, detail="Patient profile not found")

    plan = db.query(models.TreatmentPlan).filter(
        models.TreatmentPlan.plan_id == plan_id,
        models.TreatmentPlan.patient_id == patient.patient_id,
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Verify step belongs to this plan
    step = db.query(models.TreatmentStep).filter(
        models.TreatmentStep.step_id == data.step_id,
        models.TreatmentStep.plan_id == plan_id,
    ).first()
    if not step:
        raise HTTPException(status_code=404, detail="Treatment step not found in this plan")

    record = models.TreatmentAdherence(
        step_id=data.step_id,
        date=data.date or datetime.utcnow(),
        taken=data.taken,
        side_effects=data.side_effects,
        notes=data.notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ── Get adherence for a plan on a date ───────────────────────────────────

@router.get("/plans/{plan_id}/adherence", response_model=List[schemas.TreatmentAdherenceOut])
def get_adherence(
    plan_id: int,
    date: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Get adherence records for a plan, optionally filtered by date."""
    query = (
        db.query(models.TreatmentAdherence)
        .join(models.TreatmentStep)
        .filter(models.TreatmentStep.plan_id == plan_id)
    )

    if date:
        try:
            target = datetime.strptime(date, "%Y-%m-%d").date()
            records = query.all()
            return [r for r in records if r.date.date() == target]
        except ValueError:
            raise HTTPException(status_code=400, detail="Use YYYY-MM-DD date format")

    return query.order_by(models.TreatmentAdherence.date.desc()).limit(100).all()


# ── Patient: report side effect ──────────────────────────────────────────

@router.post("/plans/{plan_id}/report-side-effect")
def report_side_effect(
    plan_id: int,
    data: schemas.SideEffectReport,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Patient reports a side effect for a medication step."""
    patient = db.query(models.Patient).filter(
        models.Patient.user_id == user.user_id
    ).first()
    if not patient:
        raise HTTPException(status_code=403, detail="Patient profile not found")

    plan = db.query(models.TreatmentPlan).filter(
        models.TreatmentPlan.plan_id == plan_id,
        models.TreatmentPlan.patient_id == patient.patient_id,
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    step = db.query(models.TreatmentStep).filter(
        models.TreatmentStep.step_id == data.step_id,
        models.TreatmentStep.plan_id == plan_id,
    ).first()
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")

    record = models.TreatmentAdherence(
        step_id=data.step_id,
        date=datetime.utcnow(),
        taken=True,
        side_effects=data.description,
        notes=f"SEVERITY: {data.severity}",
    )
    db.add(record)
    db.commit()

    return {"status": "reported", "step": step.medication_name, "severity": data.severity}
