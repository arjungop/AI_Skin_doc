from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime
from typing import Optional
import re


def _validate_password_strength(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one digit")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\;'/`~]", password):
        raise ValueError("Password must contain at least one special character")
    return password


class PatientCreate(BaseModel):
    username: Optional[str] = None
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    age: int
    gender: str

    @field_validator("password")
    @classmethod
    def check_password(cls, v: str) -> str:
        return _validate_password_strength(v)

class PatientLogin(BaseModel):
    email: EmailStr
    password: str

class PatientOut(BaseModel):
    patient_id: int
    user_id: int
    username: str
    email: EmailStr
    first_name: str
    last_name: str
    age: int
    gender: str
    role: str

    model_config = {"from_attributes": True}

class AppointmentCreate(BaseModel):
    patient_id: int
    doctor_id: int
    appointment_date: datetime
    reason: str | None = None

class AppointmentOut(AppointmentCreate):
    appointment_id: int
    status: str

    model_config = {"from_attributes": True}

# Lesion
class LesionCreate(BaseModel):
    patient_id: int
    image_path: Optional[str] = None
    prediction: Optional[str] = None

class LesionOut(LesionCreate):
    lesion_id: int
    created_at: datetime
    risk_score: float | None = None
    explain_url: str | None = None

    model_config = {"from_attributes": True}

# Transaction
class TransactionCreate(BaseModel):
    user_id: int
    amount: float
    status: Optional[str] = None
    category: Optional[str] = None

class TransactionOut(TransactionCreate):
    transaction_id: int
    created_at: datetime

    model_config = {"from_attributes": True}



# Doctors
class DoctorOut(BaseModel):
    doctor_id: int
    user_id: int
    username: str
    email: EmailStr
    specialization: Optional[str] = None

    model_config = {"from_attributes": True}

class DoctorApply(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    specialization: Optional[str] = None
    license_no: Optional[str] = None
    department: Optional[str] = None

    @field_validator("password")
    @classmethod
    def check_password(cls, v: str) -> str:
        return _validate_password_strength(v)

class DoctorApplicationOut(BaseModel):
    application_id: int
    user_id: int
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: EmailStr
    specialization: Optional[str] = None
    license_no: Optional[str] = None
    department: Optional[str] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

# Chat
class ChatRoomCreate(BaseModel):
    patient_id: int
    doctor_id: int

class ChatRoomOut(BaseModel):
    room_id: int
    patient_id: int
    doctor_id: int
    created_at: datetime
    last_message_at: datetime
    is_active: bool
    video_link: Optional[str] = None
    unread_count_patient: int
    unread_count_doctor: int
    patient: Optional[dict] = None
    doctor: Optional[dict] = None
    last_message: Optional[dict] = None

    model_config = {"from_attributes": True}

class MessageCreate(BaseModel):
    content: Optional[str] = None
    reply_to_message_id: Optional[int] = None
    is_urgent: bool = False

class MessageOut(BaseModel):
    message_id: int
    room_id: int
    sender_user_id: int
    message_type: str
    content: Optional[str] = None
    reply_to_message_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    status: str
    is_edited: bool
    is_deleted: bool
    is_urgent: bool = False
    sender: Optional[dict] = None
    reply_to: Optional[dict] = None

    model_config = {"from_attributes": True}

class MessageUpdate(BaseModel):
    content: Optional[str] = None
    is_deleted: Optional[bool] = None

class VideoLinkUpdate(BaseModel):
    video_link: Optional[str] = None

# Availability
class AvailabilityItem(BaseModel):
    weekday: int  # 0-6
    start_time: str  # HH:MM
    end_time: str
    timezone: Optional[str] = None

class AvailabilityOut(AvailabilityItem):
    availability_id: int
    doctor_id: int

    model_config = {"from_attributes": True}

# AI Chat (per-user)
class AIChatSessionCreate(BaseModel):
    title: Optional[str] = None

class AIChatSessionOut(BaseModel):
    session_id: int
    title: str
    created_at: datetime
    model_config = {"from_attributes": True}

class AIChatMessageIn(BaseModel):
    role: str
    content: str

class AIChatMessageOut(BaseModel):
    message_id: int
    role: str
    content: str
    created_at: datetime
    model_config = {"from_attributes": True}

# Diagnosis Reports
class DiagnosisReportOut(BaseModel):
    report_id: int
    lesion_id: int
    patient_id: int
    prediction: str | None = None
    summary: str | None = None
    details: str
    created_at: datetime
    model_config = {"from_attributes": True}

# Lesion review + enriched list
class LesionReviewOut(BaseModel):
    review_id: int
    lesion_id: int
    doctor_id: int
    decision: str
    override_label: str | None = None
    comment: str | None = None
    created_at: datetime
    model_config = {"from_attributes": True}

class LesionWithReviewOut(BaseModel):
    lesion_id: int
    patient_id: int
    image_path: str | None = None
    prediction: str | None = None
    created_at: datetime
    latest_review: dict | None = None

# User Profile (Personalization)
class UserProfileBase(BaseModel):
    skin_type: Optional[str] = None
    sensitivity_level: Optional[str] = None
    acne_prone: bool = False
    fitzpatrick_type: Optional[int] = None
    allergies: Optional[str] = None # JSON or comma-separated
    goals: Optional[str] = None # JSON
    location_city: Optional[str] = None

class UserProfileCreate(UserProfileBase):
    pass

class UserProfileUpdate(UserProfileBase):
    pass

class UserProfileOut(UserProfileBase):
    profile_id: int
    user_id: int
    updated_at: datetime
    model_config = {"from_attributes": True}

# --- Skin Journey Models ---

class SkinLogCreate(BaseModel):
    image_path: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[str] = None

class SkinLogOut(SkinLogCreate):
    log_id: int
    user_id: int
    created_at: datetime
    model_config = {"from_attributes": True}

# ── Treatment Plan Schemas ────────────────────────────────────────────────

class TreatmentPlanCreate(BaseModel):
    patient_id: int
    doctor_id: Optional[int] = None
    diagnosis: str
    notes: Optional[str] = None

class TreatmentStepCreate(BaseModel):
    medication_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = "daily"
    time_of_day: Optional[str] = "PM"
    instructions: Optional[str] = None
    step_order: Optional[int] = 1

class TreatmentStepOut(BaseModel):
    step_id: int
    plan_id: int
    medication_name: str
    dosage: Optional[str] = None
    frequency: str
    time_of_day: str
    instructions: Optional[str] = None
    step_order: int
    is_active: bool
    created_at: datetime
    model_config = {"from_attributes": True}

class TreatmentPlanOut(BaseModel):
    plan_id: int
    patient_id: int
    doctor_id: int
    diagnosis: str
    status: str
    notes: Optional[str] = None
    created_at: datetime
    steps: list[TreatmentStepOut] = []
    model_config = {"from_attributes": True}

class TreatmentAdherenceCreate(BaseModel):
    step_id: int
    date: Optional[datetime] = None
    taken: bool = True
    side_effects: Optional[str] = None
    notes: Optional[str] = None

class TreatmentAdherenceOut(BaseModel):
    adherence_id: int
    step_id: int
    date: datetime
    taken: bool
    side_effects: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    model_config = {"from_attributes": True}

class SideEffectReport(BaseModel):
    step_id: int
    description: str
    severity: str = "mild"  # mild / moderate / severe

# ── Doctor Suggestion Schemas ─────────────────────────────────────────────

class DoctorSuggestionCreate(BaseModel):
    report_id: int
    product_name: str
    product_link: Optional[str] = None
    notes: Optional[str] = None

class DoctorSuggestionOut(BaseModel):
    suggestion_id: int
    report_id: int
    doctor_id: int
    product_name: str
    product_link: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    model_config = {"from_attributes": True}
