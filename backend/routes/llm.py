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
        # minimal patient context
        patient_dict = {
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "age": patient.age,
            "gender": patient.gender,
        }
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
    history = [m.dict() for m in (req.history or [])]
    gen = stream_chat_reply(req.prompt, patient=patient_dict, history=history)
    return StreamingResponse(gen, media_type="text/plain")
