from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Boolean, Text, Enum
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime
import enum

class MessageType(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    SYSTEM = "system"

class MessageStatus(str, enum.Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"

class OnlineStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    AWAY = "away"

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)

    patient = relationship("Patient", back_populates="user", uselist=False)
    doctor = relationship("Doctor", back_populates="user", uselist=False)

class Patient(Base):
    __tablename__ = "patients"
    patient_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)

    user = relationship("User", back_populates="patient")
    appointments = relationship("Appointment", back_populates="patient")
    lesions = relationship("Lesion", back_populates="patient")

class Doctor(Base):
    __tablename__ = "doctors"
    doctor_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    specialization = Column(String(100))

    user = relationship("User", back_populates="doctor")
    appointments = relationship("Appointment", back_populates="doctor")


class DoctorProfile(Base):
    __tablename__ = "doctor_profiles"
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), primary_key=True, index=True)
    bio = Column(String(4000))
    visibility = Column(String(5), default="true")  # 'true' | 'false'

    doctor = relationship("Doctor")

class DoctorApplication(Base):
    __tablename__ = "doctor_applications"
    application_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    specialization = Column(String(100))
    license_no = Column(String(100))
    hospital = Column(String(150))
    status = Column(String(20), default="PENDING")  # PENDING | APPROVED | REJECTED
    created_at = Column(DateTime, default=datetime.utcnow)

class Appointment(Base):
    __tablename__ = "appointments"
    appointment_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), nullable=False)
    appointment_date = Column(DateTime, nullable=False)
    reason = Column(String(255))
    status = Column(String(50), default="Scheduled")

    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")

class Lesion(Base):
    __tablename__ = "lesions"
    lesion_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    image_path = Column(String(255))
    prediction = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="lesions")

class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String(50), default="pending")
    category = Column(String(30), default="general")
    created_at = Column(DateTime, default=datetime.utcnow)


class TransactionMeta(Base):
    __tablename__ = "transaction_meta"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.transaction_id"), nullable=False)
    method = Column(String(30))  # e.g., UPI, Card, Cash, Bank
    reference = Column(String(100))
    note = Column(String(255))
    refund_of = Column(Integer, ForeignKey("transactions.transaction_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", foreign_keys=[transaction_id])


class ChatRoom(Base):
    __tablename__ = "chat_rooms"
    room_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    unread_count_patient = Column(Integer, default=0)  # unread count for patient
    unread_count_doctor = Column(Integer, default=0)   # unread count for doctor

    patient = relationship("Patient")
    doctor = relationship("Doctor")
    messages = relationship("Message", back_populates="room", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    message_id = Column(Integer, primary_key=True, index=True)
    room_id = Column(Integer, ForeignKey("chat_rooms.room_id"), nullable=False)
    sender_user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    message_type = Column(Enum(MessageType), default=MessageType.TEXT)
    content = Column(Text, nullable=True)  # Text content
    file_url = Column(String(500), nullable=True)  # For attachments
    file_name = Column(String(255), nullable=True)  # Original file name
    file_size = Column(Integer, nullable=True)  # File size in bytes
    reply_to_message_id = Column(Integer, ForeignKey("messages.message_id"), nullable=True)  # For replies
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(Enum(MessageStatus), default=MessageStatus.SENT)
    is_edited = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)

    room = relationship("ChatRoom", back_populates="messages")
    sender = relationship("User")
    reply_to = relationship("Message", remote_side=[message_id])
    reactions = relationship("MessageReaction", back_populates="message", cascade="all, delete-orphan")


class MessageReaction(Base):
    __tablename__ = "message_reactions"
    reaction_id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.message_id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    emoji = Column(String(10), nullable=False)  # emoji unicode
    created_at = Column(DateTime, default=datetime.utcnow)

    message = relationship("Message", back_populates="reactions")
    user = relationship("User")


class UserOnlineStatus(Base):
    __tablename__ = "user_online_status"
    user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True, index=True)
    status = Column(Enum(OnlineStatus), default=OnlineStatus.OFFLINE)
    last_seen = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class DoctorAvailability(Base):
    __tablename__ = "doctor_availability"
    availability_id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), nullable=False)
    weekday = Column(Integer, nullable=False)  # 0=Mon .. 6=Sun
    start_time = Column(String(5), nullable=False)  # "HH:MM" (24h)
    end_time = Column(String(5), nullable=False)
    timezone = Column(String(64), default="local")

    doctor = relationship("Doctor")


class LesionReview(Base):
    __tablename__ = "lesion_reviews"
    review_id = Column(Integer, primary_key=True, index=True)
    lesion_id = Column(Integer, ForeignKey("lesions.lesion_id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.doctor_id"), nullable=False)
    decision = Column(String(20), nullable=False)  # 'confirmed' | 'overridden'
    override_label = Column(String(100))
    comment = Column(String(4000))
    created_at = Column(DateTime, default=datetime.utcnow)

    lesion = relationship("Lesion")
    doctor = relationship("Doctor")


# AI Chat sessions (per user)
class AIChatSession(Base):
    __tablename__ = "ai_chat_sessions"
    session_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    title = Column(String(120), default="Chat")
    created_at = Column(DateTime, default=datetime.utcnow)


class AIChatMessage(Base):
    __tablename__ = "ai_chat_messages"
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("ai_chat_sessions.session_id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(String(4000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Diagnosis reports for lesions
class DiagnosisReport(Base):
    __tablename__ = "diagnosis_reports"
    report_id = Column(Integer, primary_key=True, index=True)
    lesion_id = Column(Integer, ForeignKey("lesions.lesion_id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"), nullable=False)
    prediction = Column(String(100))
    summary = Column(String(255))
    details = Column(String(8000))
    created_at = Column(DateTime, default=datetime.utcnow)


class UserStatus(Base):
    __tablename__ = "user_status"
    user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True, index=True)
    status = Column(String(20), default="ACTIVE")  # ACTIVE | SUSPENDED | TERMINATED
    terminated_at = Column(DateTime, nullable=True)
    terminated_by = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    termination_reason = Column(String(1000), nullable=True)

# Simple settings store (key/value)
class Setting(Base):
    __tablename__ = "settings"
    key = Column(String(120), primary_key=True, index=True)
    value = Column(String(2000))
    updated_at = Column(DateTime, default=datetime.utcnow)


class UserTokenVersion(Base):
    __tablename__ = "user_token_versions"
    user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True, index=True)
    version = Column(Integer, default=1)
    updated_at = Column(DateTime, default=datetime.utcnow)


# Audit log for admin actions
class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    action = Column(String(120), nullable=False)
    meta = Column(String(4000))
    created_at = Column(DateTime, default=datetime.utcnow)


# Support tables (ORM so it works across MySQL/SQLite)
class SupportTicket(Base):
    __tablename__ = "support_tickets"
    ticket_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    name = Column(String(100))
    email = Column(String(150), nullable=False)
    subject = Column(String(255))
    message = Column(String(2000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class NewsletterSubscriber(Base):
    __tablename__ = "newsletter_subscribers"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(150), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
