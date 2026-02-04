from fastapi import APIRouter, Depends, HTTPException
import os
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models
from pydantic import BaseModel
from typing import List, Dict, Optional
from backend.llm_service import chat_reply, diagnosis_for_lesion, stream_chat_reply
from backend.llm_service import _provider  # type: ignore
from fastapi.responses import StreamingResponse

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    patient_id: Optional[int] = None
    prompt: str
    history: Optional[List[ChatMessage]] = None


@router.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    patient_dict = None
    if req.patient_id:
        patient = db.query(models.Patient).filter(models.Patient.patient_id == req.patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Profile fetch
        profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == patient.user_id).first()
        
        patient_dict = {
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "age": patient.age,
            "gender": patient.gender,
        }
        if profile:
            patient_dict.update({
                "skin_type": profile.skin_type,
                "sensitivity": profile.sensitivity_level,
                "concerns": profile.goals, # goals store concerns list
                "allergies": profile.allergies,
                "location": profile.location_city
            })
    reply = chat_reply(req.prompt, patient=patient_dict, history=[m.dict() for m in (req.history or [])])
    return {"reply": reply}


class DiagnoseRequest(BaseModel):
    patient_id: int
    lesion_id: int


@router.post("/diagnose")
def diagnose(req: DiagnoseRequest, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.patient_id == req.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    lesion = db.query(models.Lesion).filter(models.Lesion.lesion_id == req.lesion_id).first()
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    patient_dict = {
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "age": patient.age,
        "gender": patient.gender,
    }
    lesion_dict = {
        "prediction": lesion.prediction,
        "image_path": lesion.image_path,
    }
    result = diagnosis_for_lesion(patient_dict, lesion_dict)
    return {"diagnosis": result}
@router.get("/status")
def status():
    prov = _provider() or "fallback"
    details = {"provider": prov}
    missing: list[str] = []
    if prov == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            missing.append("GEMINI_API_KEY")
        if not os.getenv("GEMINI_MODEL"):
            details["note"] = "Using default model 'gemini-pro'"
    elif prov == "azure":
        for k in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]:
            if not os.getenv(k):
                missing.append(k)
    elif prov == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if not os.getenv("OPENAI_MODEL"):
            details["note"] = "Using default model 'gpt-4o-mini'"
    elif prov == "ollama":
        for k in ["OLLAMA_BASE_URL", "OLLAMA_MODEL"]:
            if not os.getenv(k):
                missing.append(k)
    else:
        missing.extend(["GEMINI_API_KEY", "AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT", "OPENAI_API_KEY", "OLLAMA_BASE_URL/MODEL"])
    if missing:
        details["missing"] = missing
    return details


@router.post("/chat_stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    patient_dict = None
    if req.patient_id:
        patient = db.query(models.Patient).filter(models.Patient.patient_id == req.patient_id).first()
        if patient:
            patient_dict = {
                "first_name": patient.first_name,
                "last_name": patient.last_name,
                "age": patient.age,
                "gender": patient.gender,
            }
            profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == patient.user_id).first()
            if profile:
                patient_dict.update({
                    "skin_type": profile.skin_type,
                    "sensitivity": profile.sensitivity_level,
                    "concerns": profile.goals,
                    "allergies": profile.allergies,
                    "location": profile.location_city
                })
    history = [m.dict() for m in (req.history or [])]
    gen = stream_chat_reply(req.prompt, patient=patient_dict, history=history)
    return StreamingResponse(gen, media_type="text/plain")


class NoteGenerationRequest(BaseModel):
    room_id: int

@router.post("/generate_notes")
def generate_notes(req: NoteGenerationRequest, db: Session = Depends(get_db)):
    room = db.query(models.ChatRoom).filter(models.ChatRoom.room_id == req.room_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Chat room not found")
    
    # Fetch Messages
    messages = db.query(models.Message)\
        .filter(models.Message.room_id == req.room_id)\
        .order_by(models.Message.created_at.asc())\
        .all()
    
    if not messages:
        return {"notes": "No conversation history available to generate notes."}

    # Fetch Patient Profile
    patient = db.query(models.Patient).filter(models.Patient.patient_id == room.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient_dict = {
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "age": patient.age,
        "gender": patient.gender,
    }
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == patient.user_id).first()
    if profile:
        patient_dict.update({
            "skin_type": profile.skin_type,
            "allergies": profile.allergies,
        })
    
    # Format history
    chat_history = []
    for m in messages:
        # We need to map user_id to role. 
        # Doctor messages -> assistant/doctor, Patient messages -> user/patient
        role = "assistant" if m.sender_user_id == room.doctor.user_id else "user"
        chat_history.append({"role": role, "content": m.content or "[Image/File]"})

    from backend.llm_service import generate_clinical_notes
    notes = generate_clinical_notes(patient_dict, chat_history)
    
    return {"notes": notes}
