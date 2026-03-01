from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, desc
from typing import List, Optional
from datetime import datetime, timezone

from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user
from backend.notify import NotificationHub

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────

def _authorize_room_access(db: Session, user: models.User, room_id: int) -> models.ChatRoom:
    """Authorize user access to a chat room."""
    room = db.query(models.ChatRoom).filter(models.ChatRoom.room_id == room_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    role = (user.role or "").upper()
    if role == "ADMIN":
        return room

    patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
    doctor = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()

    if patient and room.patient_id == patient.patient_id:
        return room
    if doctor and room.doctor_id == doctor.doctor_id:
        return room

    raise HTTPException(status_code=403, detail="Access denied to this chat room")


def _get_user_role_and_ids(db: Session, user: models.User):
    """Get user role and patient/doctor entities."""
    role = (user.role or "").upper()
    patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
    doctor = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
    return role, patient, doctor


def _audit(db: Session, user_id: int, action: str, meta: str = ""):
    """Write an entry to the audit log."""
    log = models.AuditLog(user_id=user_id, action=action, meta=meta[:4000])
    db.add(log)
    # committed by the caller together with the main transaction


def _populate_room_data(room: models.ChatRoom, db: Session, *, prefetched_patients=None, prefetched_doctors=None) -> dict:
    """Build a dict representation of a room with participant info.
    Uses prefetched maps when available to avoid N+1 queries."""
    if prefetched_patients and room.patient_id in prefetched_patients:
        patient = prefetched_patients[room.patient_id]
    else:
        patient = db.query(models.Patient).options(
            joinedload(models.Patient.user)
        ).filter(models.Patient.patient_id == room.patient_id).first()

    if prefetched_doctors and room.doctor_id in prefetched_doctors:
        doctor = prefetched_doctors[room.doctor_id]
    else:
        doctor = db.query(models.Doctor).options(
            joinedload(models.Doctor.user)
        ).filter(models.Doctor.doctor_id == room.doctor_id).first()

    last_message = db.query(models.Message).filter(
        models.Message.room_id == room.room_id,
        models.Message.is_deleted == False,
    ).order_by(desc(models.Message.created_at)).first()

    return {
        "room_id": room.room_id,
        "patient_id": room.patient_id,
        "doctor_id": room.doctor_id,
        "created_at": room.created_at,
        "last_message_at": room.last_message_at,
        "is_active": room.is_active,
        "video_link": getattr(room, 'video_link', None),
        "unread_count_patient": getattr(room, 'unread_count_patient', 0),
        "unread_count_doctor": getattr(room, 'unread_count_doctor', 0),
        "patient": {
            "patient_id": patient.patient_id,
            "user_id": patient.user.user_id if patient.user else None,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "username": patient.user.username if patient.user else f"patient_{patient.patient_id}",
        } if patient else None,
        "doctor": {
            "doctor_id": doctor.doctor_id,
            "user_id": doctor.user.user_id if doctor.user else None,
            "username": doctor.user.username if doctor.user else f"doctor_{doctor.doctor_id}",
            "first_name": getattr(doctor, 'first_name', None) or (doctor.user.username if doctor.user else f"Doctor {doctor.doctor_id}"),
            "last_name": getattr(doctor, 'last_name', None) or '',
            "specialization": doctor.specialization,
        } if doctor else None,
        "last_message": {
            "content": (last_message.content[:50] + "...")
                       if last_message and len(last_message.content or "") > 50
                       else (last_message.content if last_message else None),
            "created_at": last_message.created_at,
            "sender_user_id": last_message.sender_user_id,
            "is_urgent": getattr(last_message, 'is_urgent', False),
        } if last_message else None,
    }


def _populate_message_data(message: models.Message, db: Session, *, prefetched_users=None, prefetched_replies=None) -> dict:
    """Build a dict representation of a single message."""
    if prefetched_users and message.sender_user_id in prefetched_users:
        sender = prefetched_users[message.sender_user_id]
    else:
        sender = db.query(models.User).filter(
            models.User.user_id == message.sender_user_id
        ).first()

    reply_to = None
    if message.reply_to_message_id:
        if prefetched_replies and message.reply_to_message_id in prefetched_replies:
            reply_msg = prefetched_replies[message.reply_to_message_id]
        else:
            reply_msg = db.query(models.Message).filter(
                models.Message.message_id == message.reply_to_message_id
            ).first()
        if reply_msg:
            if prefetched_users and reply_msg.sender_user_id in prefetched_users:
                reply_sender = prefetched_users[reply_msg.sender_user_id]
            else:
                reply_sender = db.query(models.User).filter(
                    models.User.user_id == reply_msg.sender_user_id
                ).first()
            reply_to = {
                "message_id": reply_msg.message_id,
                "content": (reply_msg.content[:100] + "...")
                           if len(reply_msg.content or "") > 100
                           else reply_msg.content,
                "sender": {"username": reply_sender.username} if reply_sender else None,
            }

    return {
        "message_id": message.message_id,
        "room_id": message.room_id,
        "sender_user_id": message.sender_user_id,
        "message_type": message.message_type,
        "content": message.content,
        "reply_to_message_id": message.reply_to_message_id,
        "created_at": message.created_at,
        "updated_at": message.updated_at,
        "status": message.status,
        "is_edited": message.is_edited,
        "is_deleted": message.is_deleted,
        "is_urgent": getattr(message, 'is_urgent', False),
        "sender": {
            "user_id": sender.user_id,
            "username": sender.username,
            "role": sender.role,
        } if sender else None,
        "reply_to": reply_to,
    }


# ── Rooms ────────────────────────────────────────────────────────────────

@router.get("/rooms", response_model=List[dict])
def list_rooms(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """List all chat rooms for the current user."""
    role, patient, doctor = _get_user_role_and_ids(db, user)

    query = db.query(models.ChatRoom).filter(models.ChatRoom.is_active == True)

    if role == "ADMIN":
        rooms = query.all()
    elif patient:
        rooms = query.filter(models.ChatRoom.patient_id == patient.patient_id).all()
    elif doctor:
        rooms = query.filter(models.ChatRoom.doctor_id == doctor.doctor_id).all()
    else:
        return []

    result = []
    if rooms:
        # Batch load patients and doctors to avoid N+1 queries
        patient_ids = {r.patient_id for r in rooms}
        doctor_ids = {r.doctor_id for r in rooms}
        patients_list = db.query(models.Patient).options(joinedload(models.Patient.user)).filter(models.Patient.patient_id.in_(patient_ids)).all()
        doctors_list = db.query(models.Doctor).options(joinedload(models.Doctor.user)).filter(models.Doctor.doctor_id.in_(doctor_ids)).all()
        prefetched_patients = {p.patient_id: p for p in patients_list}
        prefetched_doctors = {d.doctor_id: d for d in doctors_list}
        result = [_populate_room_data(r, db, prefetched_patients=prefetched_patients, prefetched_doctors=prefetched_doctors) for r in rooms]
    result.sort(key=lambda x: x["last_message_at"] or datetime.min, reverse=True)
    return result


@router.post("/rooms", response_model=dict)
def create_room(
    data: schemas.ChatRoomCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Create a new chat room."""
    role, patient, doctor = _get_user_role_and_ids(db, user)

    if role != "ADMIN":
        if role == "PATIENT":
            if not patient or patient.patient_id != data.patient_id:
                raise HTTPException(403, "Cannot create room for another patient")
        elif role == "DOCTOR":
            if not doctor or doctor.doctor_id != data.doctor_id:
                raise HTTPException(403, "Cannot create room for another doctor")
        else:
            raise HTTPException(403, "Insufficient permissions")

    existing = db.query(models.ChatRoom).filter(
        models.ChatRoom.patient_id == data.patient_id,
        models.ChatRoom.doctor_id == data.doctor_id,
    ).first()
    if existing:
        return _populate_room_data(existing, db)

    room = models.ChatRoom(patient_id=data.patient_id, doctor_id=data.doctor_id)
    db.add(room)

    system_msg = models.Message(
        room_id=0,  # placeholder, updated after flush
        sender_user_id=user.user_id,
        message_type=models.MessageType.SYSTEM,
        content="Secure message room created. Messages are logged for HIPAA compliance.",
    )
    db.flush()  # generates room.room_id
    system_msg.room_id = room.room_id
    db.add(system_msg)

    _audit(db, user.user_id, "chat_room_created", f"room_id={room.room_id}")
    db.commit()

    return _populate_room_data(room, db)


@router.put("/rooms/{room_id}/video-link")
def set_video_link(
    room_id: int,
    body: schemas.VideoLinkUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Set or clear a video consultation link on a room (doctors/admins only)."""
    room = _authorize_room_access(db, user, room_id)
    role = (user.role or "").upper()

    if role not in ("DOCTOR", "ADMIN"):
        raise HTTPException(403, "Only doctors can set video links")

    room.video_link = body.video_link
    _audit(db, user.user_id, "video_link_set", f"room_id={room_id} link={body.video_link}")
    db.commit()

    return {"status": "ok", "video_link": room.video_link}


# ── Messages ─────────────────────────────────────────────────────────────

@router.get("/rooms/{room_id}/messages")
def list_messages(
    room_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """List messages with pagination (polling-friendly)."""
    _authorize_room_access(db, user, room_id)

    # Mark messages as read for this user
    role, patient, doctor = _get_user_role_and_ids(db, user)
    room = db.query(models.ChatRoom).filter(models.ChatRoom.room_id == room_id).first()

    if patient and room.patient_id == patient.patient_id:
        room.unread_count_patient = 0
    elif doctor and room.doctor_id == doctor.doctor_id:
        room.unread_count_doctor = 0

    # Mark individual messages from the OTHER user as READ
    db.query(models.Message).filter(
        models.Message.room_id == room_id,
        models.Message.sender_user_id != user.user_id,
        models.Message.status != models.MessageStatus.READ,
        models.Message.is_deleted == False,
    ).update({"status": models.MessageStatus.READ}, synchronize_session="fetch")

    db.commit()

    offset = (page - 1) * limit
    messages = (
        db.query(models.Message)
        .filter(models.Message.room_id == room_id, models.Message.is_deleted == False)
        .order_by(desc(models.Message.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    messages.reverse()

    # Batch load users and reply messages to avoid N+1 queries
    prefetched_users = {}
    prefetched_replies = {}
    if messages:
        sender_ids = {m.sender_user_id for m in messages}
        reply_ids = {m.reply_to_message_id for m in messages if m.reply_to_message_id}
        if reply_ids:
            reply_msgs = db.query(models.Message).filter(models.Message.message_id.in_(reply_ids)).all()
            prefetched_replies = {rm.message_id: rm for rm in reply_msgs}
            sender_ids.update(rm.sender_user_id for rm in reply_msgs)
        users = db.query(models.User).filter(models.User.user_id.in_(sender_ids)).all()
        prefetched_users = {u.user_id: u for u in users}

    return {
        "messages": [_populate_message_data(m, db, prefetched_users=prefetched_users, prefetched_replies=prefetched_replies) for m in messages],
        "page": page,
        "limit": limit,
        "total": db.query(models.Message).filter(
            models.Message.room_id == room_id,
            models.Message.is_deleted == False,
        ).count(),
    }


@router.post("/rooms/{room_id}/messages", response_model=dict)
def post_message(
    room_id: int,
    body: schemas.MessageCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Post a new message to a chat room."""
    room = _authorize_room_access(db, user, room_id)

    if not (body.content or "").strip():
        raise HTTPException(400, "Message cannot be empty")

    message = models.Message(
        room_id=room_id,
        sender_user_id=user.user_id,
        message_type=models.MessageType.TEXT,
        content=body.content,
        reply_to_message_id=body.reply_to_message_id,
        is_urgent=body.is_urgent or False,
    )
    db.add(message)

    role, patient, doctor = _get_user_role_and_ids(db, user)
    room.last_message_at = datetime.now(timezone.utc)

    if patient and room.patient_id == patient.patient_id:
        room.unread_count_doctor += 1
    elif doctor and room.doctor_id == doctor.doctor_id:
        room.unread_count_patient += 1

    _audit(db, user.user_id, "message_sent", f"room_id={room_id} urgent={body.is_urgent}")
    db.commit()
    db.refresh(message)

    # Push notification to other participant
    try:
        notify_ids = []
        if patient and room.patient_id == patient.patient_id:
            doc_user = db.query(models.User).join(models.Doctor).filter(
                models.Doctor.doctor_id == room.doctor_id
            ).first()
            if doc_user:
                notify_ids.append(doc_user.user_id)
        elif doctor and room.doctor_id == doctor.doctor_id:
            pat_user = db.query(models.User).join(models.Patient).filter(
                models.Patient.patient_id == room.patient_id
            ).first()
            if pat_user:
                notify_ids.append(pat_user.user_id)

        label = "🚨 URGENT: " if message.is_urgent else ""
        for uid in notify_ids:
            NotificationHub.send_many([uid], "new_message", {
                "room_id": room_id,
                "message_id": message.message_id,
                "sender": user.username,
                "content": label + (message.content[:100] if message.content else "New message"),
            })
            # If recipient has active WebSocket, mark as DELIVERED
            if uid in NotificationHub.by_user and NotificationHub.by_user[uid]:
                message.status = models.MessageStatus.DELIVERED
                db.commit()
    except Exception as e:
        print(f"Notification error: {e}")

    return _populate_message_data(message, db)


@router.put("/messages/{message_id}", response_model=dict)
def update_message(
    message_id: int,
    body: schemas.MessageUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Edit or soft-delete a message."""
    message = db.query(models.Message).filter(models.Message.message_id == message_id).first()
    if not message:
        raise HTTPException(404, "Message not found")

    if message.sender_user_id != user.user_id and (user.role or "").upper() != "ADMIN":
        raise HTTPException(403, "Can only edit your own messages")

    _authorize_room_access(db, user, message.room_id)

    if body.content is not None:
        message.content = body.content
        message.is_edited = True
        message.updated_at = datetime.utcnow()

    if body.is_deleted is not None:
        message.is_deleted = body.is_deleted
        message.updated_at = datetime.utcnow()

    _audit(db, user.user_id, "message_updated", f"message_id={message_id}")
    db.commit()

    return _populate_message_data(message, db)


@router.put("/messages/{message_id}/urgent")
def mark_urgent(
    message_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Toggle the urgent flag on a message."""
    message = db.query(models.Message).filter(models.Message.message_id == message_id).first()
    if not message:
        raise HTTPException(404, "Message not found")

    _authorize_room_access(db, user, message.room_id)

    message.is_urgent = not message.is_urgent
    message.updated_at = datetime.utcnow()

    _audit(db, user.user_id, "message_urgent_toggled",
           f"message_id={message_id} is_urgent={message.is_urgent}")
    db.commit()

    return {"message_id": message_id, "is_urgent": message.is_urgent}


# ── Online Status ────────────────────────────────────────────────────────

@router.post("/online")
def heartbeat(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Update user online status (heartbeat). Call every ~30s from frontend."""
    status_row = db.query(models.UserOnlineStatus).filter(
        models.UserOnlineStatus.user_id == user.user_id
    ).first()
    now = datetime.now(timezone.utc)
    if status_row:
        status_row.status = models.OnlineStatus.ONLINE
        status_row.last_seen = now
        status_row.last_activity = now
    else:
        status_row = models.UserOnlineStatus(
            user_id=user.user_id,
            status=models.OnlineStatus.ONLINE,
            last_seen=now,
            last_activity=now,
        )
        db.add(status_row)
    db.commit()
    return {"status": "online"}


@router.post("/offline")
def go_offline(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Mark user as offline (called on logout/tab close)."""
    status_row = db.query(models.UserOnlineStatus).filter(
        models.UserOnlineStatus.user_id == user.user_id
    ).first()
    if status_row:
        status_row.status = models.OnlineStatus.OFFLINE
        status_row.last_seen = datetime.now(timezone.utc)
        db.commit()
    return {"status": "offline"}


@router.get("/online/{user_id}")
def get_online_status(
    user_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Get the online status of another user."""
    status_row = db.query(models.UserOnlineStatus).filter(
        models.UserOnlineStatus.user_id == user_id
    ).first()
    if not status_row:
        return {"user_id": user_id, "status": "offline", "last_seen": None}

    # Auto-degrade to offline if heartbeat too old (>2 min)
    now = datetime.now(timezone.utc)
    last_seen = status_row.last_seen
    if last_seen and last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    if status_row.status == models.OnlineStatus.ONLINE and last_seen:
        diff = (now - last_seen).total_seconds()
        if diff > 120:
            status_row.status = models.OnlineStatus.OFFLINE
            db.commit()

    return {
        "user_id": user_id,
        "status": status_row.status.value if status_row.status else "offline",
        "last_seen": status_row.last_seen,
    }


# ── Read Receipts ────────────────────────────────────────────────────────

@router.post("/rooms/{room_id}/read")
def mark_room_read(
    room_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Mark all messages in a room as read for the current user and reset unread counter."""
    room = _authorize_room_access(db, user, room_id)
    role, patient, doctor = _get_user_role_and_ids(db, user)

    # Reset unread counter
    if patient and room.patient_id == patient.patient_id:
        room.unread_count_patient = 0
    elif doctor and room.doctor_id == doctor.doctor_id:
        room.unread_count_doctor = 0
    elif role == "ADMIN":
        room.unread_count_patient = 0
        room.unread_count_doctor = 0

    # Mark unread messages as READ
    unread_msgs = db.query(models.Message).filter(
        models.Message.room_id == room_id,
        models.Message.sender_user_id != user.user_id,
        models.Message.status != models.MessageStatus.READ,
        models.Message.is_deleted == False,
    ).all()

    for msg in unread_msgs:
        msg.status = models.MessageStatus.READ

    db.commit()

    # Notify the other participant about read receipt
    try:
        notify_target = None
        if patient and room.patient_id == patient.patient_id:
            doc = db.query(models.Doctor).filter(models.Doctor.doctor_id == room.doctor_id).first()
            if doc:
                doc_user = db.query(models.User).filter(models.User.user_id == doc.user_id).first()
                if doc_user:
                    notify_target = doc_user.user_id
        elif doctor and room.doctor_id == doctor.doctor_id:
            pat = db.query(models.Patient).filter(models.Patient.patient_id == room.patient_id).first()
            if pat:
                notify_target = pat.user_id

        if notify_target:
            NotificationHub.send(notify_target, "messages_read", {
                "room_id": room_id,
                "reader_user_id": user.user_id,
            })
    except Exception:
        pass

    return {"status": "ok", "marked_read": len(unread_msgs)}


# ── Unread Counts (Global) ──────────────────────────────────────────────

@router.get("/unread")
def get_unread_counts(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Get total unread message count for the current user across all rooms."""
    role, patient, doctor = _get_user_role_and_ids(db, user)

    rooms = db.query(models.ChatRoom).filter(models.ChatRoom.is_active == True)

    total_unread = 0
    room_unreads = []

    if role == "ADMIN":
        for room in rooms.all():
            unread = max(getattr(room, 'unread_count_patient', 0), getattr(room, 'unread_count_doctor', 0))
            if unread > 0:
                room_unreads.append({"room_id": room.room_id, "unread": unread})
            total_unread += unread
    elif patient:
        for room in rooms.filter(models.ChatRoom.patient_id == patient.patient_id).all():
            unread = getattr(room, 'unread_count_patient', 0)
            if unread > 0:
                room_unreads.append({"room_id": room.room_id, "unread": unread})
            total_unread += unread
    elif doctor:
        for room in rooms.filter(models.ChatRoom.doctor_id == doctor.doctor_id).all():
            unread = getattr(room, 'unread_count_doctor', 0)
            if unread > 0:
                room_unreads.append({"room_id": room.room_id, "unread": unread})
            total_unread += unread

    return {"total_unread": total_unread, "rooms": room_unreads}


# ── Typing Indicator ─────────────────────────────────────────────────────

@router.post("/rooms/{room_id}/typing")
def send_typing(
    room_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Broadcast a typing indicator to the other participant via WebSocket."""
    room = _authorize_room_access(db, user, room_id)
    role, patient, doctor = _get_user_role_and_ids(db, user)

    try:
        notify_target = None
        if patient and room.patient_id == patient.patient_id:
            doc = db.query(models.Doctor).filter(models.Doctor.doctor_id == room.doctor_id).first()
            if doc:
                doc_user = db.query(models.User).filter(models.User.user_id == doc.user_id).first()
                if doc_user:
                    notify_target = doc_user.user_id
        elif doctor and room.doctor_id == doctor.doctor_id:
            pat = db.query(models.Patient).filter(models.Patient.patient_id == room.patient_id).first()
            if pat:
                notify_target = pat.user_id

        if notify_target:
            NotificationHub.send(notify_target, "typing", {
                "room_id": room_id,
                "user_id": user.user_id,
                "username": user.username,
            })
    except Exception:
        pass

    return {"status": "ok"}
