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
from backend.notify import NotificationHub
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

    # Auto-inject SYSTEM message into chat
    try:
        room = db.query(models.ChatRoom).filter(
            models.ChatRoom.patient_id == data.patient_id,
            models.ChatRoom.doctor_id == doctor_id
        ).first()
        if room:
            doc_user = db.query(models.User).filter(models.User.user_id == user.user_id).first()
            doc_name = doc_user.username if doc_user else "The doctor"
            msg = models.Message(
                room_id=room.room_id,
                sender_user_id=user.user_id,
                message_type=models.MessageType.SYSTEM,
                content=f"{doc_name} has prescribed a new Treatment Plan for: {data.diagnosis}. Please review it in your plans tab."
            )
            db.add(msg)
            db.commit()

            pat_user = db.query(models.User).filter(models.User.user_id == patient.user_id).first()
            if pat_user:
                NotificationHub.send_many([pat_user.user_id], "new_message", {
                    "room_id": room.room_id,
                    "message_id": msg.message_id,
                    "sender": "System",
                    "content": msg.content
                })
    except Exception as e:
        print("Failed to auto-inject treatment plan creation message:", e)

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

    # Auto-inject URGENT SYSTEM message into chat
    try:
        room = db.query(models.ChatRoom).filter(
            models.ChatRoom.patient_id == patient.patient_id,
            models.ChatRoom.doctor_id == plan.doctor_id
        ).first()
        if room:
            content = f"⚠️ URGENT: The patient reported a {data.severity.upper()} side effect for medication: {step.medication_name}.\nDescription: {data.description}"
            msg = models.Message(
                room_id=room.room_id,
                sender_user_id=user.user_id,
                message_type=models.MessageType.SYSTEM,
                content=content,
                is_urgent=True
            )
            db.add(msg)
            db.commit()
            
            doc = db.query(models.Doctor).filter(models.Doctor.doctor_id == plan.doctor_id).first()
            if doc:
                NotificationHub.send_many([doc.user_id], "new_message", {
                    "room_id": room.room_id,
                    "message_id": msg.message_id,
                    "sender": "System",
                    "content": content
                })
    except Exception as e:
        print("Failed to auto-inject side-effect alert message:", e)

    return {"status": "reported", "step": step.medication_name, "severity": data.severity}


# ── AI: generate personalised skincare routine ───────────────────────────

import os as _os
import re as _re
import requests as _requests
import json as _json
import logging as _logging
from pydantic import BaseModel as _BaseModel
from typing import List as _List, Optional as _Optional

_routine_logger = _logging.getLogger(__name__)

_OPENROUTER_KEY = _os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL = _os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-8b-instruct:free")

_ROUTINE_SYSTEM_PROMPT = """You are a board-certified dermatologist and skincare expert.
Given a patient's skin profile, generate a highly personalised morning (AM) and evening (PM) skincare routine.
Tailor your product and ingredient choices to the patient's age, gender, location/climate, skin type, Fitzpatrick type, allergies, and stated goals.
CRITICAL: Return ONLY a valid JSON array. No markdown, no code fences, no explanation — just the raw JSON array starting with [ and ending with ].
Each element must have exactly these fields:
  "step": integer starting at 1 per time group
  "time": "AM" or "PM"
  "product": specific real product name (e.g. "CeraVe Hydrating Facial Cleanser")
  "brand": brand name
  "instructions": one concise usage sentence
Include 4-5 AM steps and 4-5 PM steps. For acne-prone/sensitive skin avoid stacking strong actives."""


class AiRoutineRequest(_BaseModel):
    skin_type: _Optional[str] = None
    concerns: _List[str] = []
    sensitivity: _Optional[str] = None
    goals: _Optional[str] = None


def _extract_json_array(text: str) -> list:
    """Robustly extract a JSON array from LLM output that may contain markdown."""
    # Strip code fences
    cleaned = _re.sub(r"```[a-zA-Z]*\n?", "", text).strip()
    cleaned = _re.sub(r"```", "", cleaned).strip()
    # Find the outermost [ ... ]
    match = _re.search(r"\[.*\]", cleaned, _re.DOTALL)
    if match:
        return _json.loads(match.group(0))
    return _json.loads(cleaned)


@router.post("/ai-generate")
def ai_generate_routine(
    body: AiRoutineRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Generate a personalised AI skincare routine using the patient's real profile."""
    if not _OPENROUTER_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI routine service not configured — set OPENROUTER_API_KEY in .env",
        )

    # ── Pull real data from DB ──────────────────────────────────────────
    db_profile = db.query(models.UserProfile).filter(
        models.UserProfile.user_id == user.user_id
    ).first()
    patient = db.query(models.Patient).filter(
        models.Patient.user_id == user.user_id
    ).first()

    # Merge: DB profile wins, request body fills gaps
    skin_type = (db_profile.skin_type if db_profile and db_profile.skin_type else body.skin_type) or "combination"
    sensitivity = (
        (db_profile.sensitivity_level.lower() if db_profile and db_profile.sensitivity_level else None)
        or body.sensitivity or "medium"
    )
    acne_prone = (db_profile.acne_prone if db_profile is not None else False) or ("acne" in body.concerns)
    fitzpatrick = db_profile.fitzpatrick_type if db_profile else None
    allergies = db_profile.allergies if db_profile else None
    location = db_profile.location_city if db_profile else None
    goals = (db_profile.goals if db_profile and db_profile.goals else None) or body.goals or "healthy, balanced skin"
    concerns_list = list(body.concerns)
    if acne_prone and "acne" not in concerns_list:
        concerns_list.append("acne")

    first_name = patient.first_name if patient else "Patient"
    age = patient.age if patient else None
    gender = patient.gender if patient else None

    # ── Build rich personalized prompt ─────────────────────────────────
    parts = [f"Patient: {first_name}"]
    if age:
        parts.append(f"Age: {age}")
    if gender:
        parts.append(f"Gender: {gender}")
    parts.append(f"Skin type: {skin_type}")
    parts.append(f"Sensitivity: {sensitivity}")
    if acne_prone:
        parts.append("Acne-prone: yes")
    if fitzpatrick:
        parts.append(f"Fitzpatrick type: {fitzpatrick}/6")
    if location:
        parts.append(f"Location/climate: {location}")
    if allergies:
        parts.append(f"Allergies/ingredients to avoid: {allergies}")
    if concerns_list:
        parts.append(f"Skin concerns: {', '.join(concerns_list)}")
    parts.append(f"Goals: {goals}")

    user_msg = ". ".join(parts) + "."

    content = "[]"
    try:
        resp = _requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": _ROUTINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.5,
                "max_tokens": 2000,
            },
            timeout=40,
        )

        if resp.status_code != 200:
            _routine_logger.warning("OpenRouter %s: %s", resp.status_code, resp.text[:300])
            raise HTTPException(status_code=502, detail=f"AI service temporarily unavailable ({resp.status_code})")

        content = (
            resp.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "[]")
            .strip()
        )

        steps = _extract_json_array(content)

    except HTTPException:
        raise
    except (_json.JSONDecodeError, ValueError):
        _routine_logger.warning("Routine LLM returned non-JSON: %s", content[:300])
        raise HTTPException(status_code=502, detail="AI returned an unexpected format — please try again")
    except Exception as e:
        _routine_logger.error("AI routine generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)[:120]}")

    # Replace previous AI routine for this user
    db.query(models.RoutineItem).filter(
        models.RoutineItem.user_id == user.user_id
    ).delete()

    saved = []
    for s in steps[:12]:
        item = models.RoutineItem(
            user_id=user.user_id,
            product_name=s.get("product", "Product"),
            brand=s.get("brand", ""),
            time_of_day=s.get("time", "AM"),
            step_order=int(s.get("step", 1)),
            notes=s.get("instructions", ""),
            is_active=True,
        )
        db.add(item)
        db.flush()
        saved.append({
            "item_id": item.item_id,
            "step": item.step_order,
            "time": item.time_of_day,
            "product": item.product_name,
            "brand": item.brand or "",
            "instructions": item.notes or "",
        })

    db.commit()
    return {
        "steps": saved,
        "skin_profile": {
            "name": first_name,
            "age": age,
            "gender": gender,
            "skin_type": skin_type,
            "sensitivity": sensitivity,
            "acne_prone": acne_prone,
            "fitzpatrick": fitzpatrick,
            "location": location,
            "goals": goals,
            "concerns": concerns_list,
        },
    }


@router.get("/my-routine")
def get_my_routine(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Return the current user's saved AI-generated routine."""
    items = (
        db.query(models.RoutineItem)
        .filter(
            models.RoutineItem.user_id == user.user_id,
            models.RoutineItem.is_active == True,
        )
        .order_by(models.RoutineItem.time_of_day, models.RoutineItem.step_order)
        .all()
    )
    return {
        "steps": [
            {
                "item_id": i.item_id,
                "step": i.step_order,
                "time": i.time_of_day,
                "product": i.product_name,
                "brand": i.brand or "",
                "instructions": i.notes or "",
            }
            for i in items
        ]
    }
