from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class PatientCreate(BaseModel):
    username: Optional[str] = None
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    age: int
    gender: str

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

class TransactionStatusUpdate(BaseModel):
    status: str
    reason: Optional[str] = None

class TransactionsSummaryOut(BaseModel):
    pending: float
    completed: float
    failed: float
    refunded: float | None = 0.0
    count: int

class TransactionMetaIn(BaseModel):
    method: Optional[str] = None
    reference: Optional[str] = None
    note: Optional[str] = None

class TransactionDetailOut(TransactionOut):
    method: Optional[str] = None
    reference: Optional[str] = None
    note: Optional[str] = None

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
    unread_count_patient: int
    unread_count_doctor: int
    patient: Optional[dict] = None  # Will be populated with patient info
    doctor: Optional[dict] = None   # Will be populated with doctor info
    last_message: Optional[dict] = None  # Last message preview

    model_config = {"from_attributes": True}

class MessageCreate(BaseModel):
    content: Optional[str] = None
    message_type: str = "text"  # text, image, file, system
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    reply_to_message_id: Optional[int] = None

class MessageOut(BaseModel):
    message_id: int
    room_id: int
    sender_user_id: int
    message_type: str
    content: Optional[str] = None
    file_url: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    reply_to_message_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    status: str
    is_edited: bool
    is_deleted: bool
    sender: Optional[dict] = None  # Will be populated with sender info
    reply_to: Optional[dict] = None  # Referenced message if this is a reply
    reactions: Optional[list] = None  # Message reactions

    model_config = {"from_attributes": True}

class MessageUpdate(BaseModel):
    content: Optional[str] = None
    is_deleted: Optional[bool] = None

class MessageReactionCreate(BaseModel):
    emoji: str

class MessageReactionOut(BaseModel):
    reaction_id: int
    message_id: int
    user_id: int
    emoji: str
    created_at: datetime
    user: Optional[dict] = None

    model_config = {"from_attributes": True}

class UserOnlineStatusOut(BaseModel):
    user_id: int
    status: str
    last_seen: datetime
    last_activity: datetime

    model_config = {"from_attributes": True}

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
