from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, UploadFile, File
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, desc, func
from typing import Dict, Set, List, Optional
import json
import asyncio
from datetime import datetime, timedelta

from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user, _decode_token
from backend.notify import NotificationHub

# Try to import Azure blob function, fallback if not available
try:
    from backend.azure_blob import upload_bytes_get_url
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    upload_bytes_get_url = None

router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {}  # room_id -> set of websockets
        self.user_connections: Dict[int, WebSocket] = {}  # user_id -> websocket
        
    async def connect(self, websocket: WebSocket, room_id: int, user_id: int):
        await websocket.accept()
        
        # Add to room connections
        if room_id not in self.active_connections:
            self.active_connections[room_id] = set()
        self.active_connections[room_id].add(websocket)
        
        # Track user connection
        self.user_connections[user_id] = websocket
        
    def disconnect(self, websocket: WebSocket, room_id: int, user_id: int):
        # Remove from room connections
        if room_id in self.active_connections:
            self.active_connections[room_id].discard(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
        
        # Remove user connection
        if user_id in self.user_connections and self.user_connections[user_id] == websocket:
            del self.user_connections[user_id]
    
    async def send_to_room(self, room_id: int, message: dict):
        if room_id in self.active_connections:
            disconnected = []
            for connection in list(self.active_connections[room_id]):
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.active_connections[room_id].discard(conn)
    
    async def send_to_user(self, user_id: int, message: dict):
        if user_id in self.user_connections:
            try:
                await self.user_connections[user_id].send_json(message)
            except:
                del self.user_connections[user_id]

manager = ConnectionManager()

def _authorize_room_access(db: Session, user: models.User, room_id: int) -> models.ChatRoom:
    """Authorize user access to a chat room"""
    room = db.query(models.ChatRoom).filter(models.ChatRoom.room_id == room_id).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    role = (user.role or "").upper()
    if role == "ADMIN":
        return room
    
    # Check if user is the patient or doctor in this room
    patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
    doctor = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
    
    if patient and room.patient_id == patient.patient_id:
        return room
    if doctor and room.doctor_id == doctor.doctor_id:
        return room
    
    raise HTTPException(status_code=403, detail="Access denied to this chat room")

def _get_user_role_and_ids(db: Session, user: models.User):
    """Get user role and patient/doctor IDs"""
    role = (user.role or "").upper()
    patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
    doctor = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
    
    return role, patient, doctor

def _populate_room_data(room: models.ChatRoom, db: Session):
    """Populate room with additional data"""
    # Get patient and doctor info
    patient = db.query(models.Patient).options(
        joinedload(models.Patient.user)
    ).filter(models.Patient.patient_id == room.patient_id).first()
    
    doctor = db.query(models.Doctor).options(
        joinedload(models.Doctor.user)
    ).filter(models.Doctor.doctor_id == room.doctor_id).first()
    
    # Get last message
    last_message = db.query(models.Message).filter(
        models.Message.room_id == room.room_id,
        models.Message.is_deleted == False
    ).order_by(desc(models.Message.created_at)).first()
    
    room_data = {
        "room_id": room.room_id,
        "patient_id": room.patient_id,
        "doctor_id": room.doctor_id,
        "created_at": room.created_at,
        "last_message_at": room.last_message_at,
        "is_active": room.is_active,
        "unread_count_patient": room.unread_count_patient,
        "unread_count_doctor": room.unread_count_doctor,
        "patient": {
            "patient_id": patient.patient_id,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "username": patient.user.username
        } if patient else None,
        "doctor": {
            "doctor_id": doctor.doctor_id,
            "username": doctor.user.username,
            "specialization": doctor.specialization
        } if doctor else None,
        "last_message": {
            "content": last_message.content[:50] + "..." if last_message and len(last_message.content or "") > 50 else last_message.content,
            "created_at": last_message.created_at,
            "sender_user_id": last_message.sender_user_id
        } if last_message else None
    }
    
    return room_data

def _populate_message_data(message: models.Message, db: Session):
    """Populate message with additional data"""
    # Get sender info
    sender = db.query(models.User).filter(models.User.user_id == message.sender_user_id).first()
    
    # Get reply to message if exists
    reply_to = None
    if message.reply_to_message_id:
        reply_msg = db.query(models.Message).filter(models.Message.message_id == message.reply_to_message_id).first()
        if reply_msg:
            reply_sender = db.query(models.User).filter(models.User.user_id == reply_msg.sender_user_id).first()
            reply_to = {
                "message_id": reply_msg.message_id,
                "content": reply_msg.content[:100] + "..." if len(reply_msg.content or "") > 100 else reply_msg.content,
                "sender": {"username": reply_sender.username} if reply_sender else None
            }
    
    # Get reactions
    reactions = db.query(models.MessageReaction).options(
        joinedload(models.MessageReaction.user)
    ).filter(models.MessageReaction.message_id == message.message_id).all()
    
    reactions_data = [
        {
            "reaction_id": r.reaction_id,
            "emoji": r.emoji,
            "user": {"user_id": r.user.user_id, "username": r.user.username}
        } for r in reactions
    ]
    
    return {
        "message_id": message.message_id,
        "room_id": message.room_id,
        "sender_user_id": message.sender_user_id,
        "message_type": message.message_type,
        "content": message.content,
        "file_url": message.file_url,
        "file_name": message.file_name,
        "file_size": message.file_size,
        "reply_to_message_id": message.reply_to_message_id,
        "created_at": message.created_at,
        "updated_at": message.updated_at,
        "status": message.status,
        "is_edited": message.is_edited,
        "is_deleted": message.is_deleted,
        "sender": {
            "user_id": sender.user_id,
            "username": sender.username,
            "role": sender.role
        } if sender else None,
        "reply_to": reply_to,
        "reactions": reactions_data
    }

@router.get("/rooms", response_model=List[dict])
def list_rooms(db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    """List all chat rooms for the current user"""
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
    
    # Populate additional data for each room
    result = []
    for room in rooms:
        room_data = _populate_room_data(room, db)
        result.append(room_data)
    
    # Sort by last message time
    result.sort(key=lambda x: x["last_message_at"], reverse=True)
    return result


@router.post("/rooms", response_model=dict)
def create_room(
    data: schemas.ChatRoomCreate, 
    db: Session = Depends(get_db), 
    user: models.User = Depends(get_current_user)
):
    """Create a new chat room"""
    role, patient, doctor = _get_user_role_and_ids(db, user)
    
    # Authorization checks
    if role != "ADMIN":
        if role == "PATIENT":
            if not patient or patient.patient_id != data.patient_id:
                raise HTTPException(status_code=403, detail="Cannot create room for another patient")
        elif role == "DOCTOR":
            if not doctor or doctor.doctor_id != data.doctor_id:
                raise HTTPException(status_code=403, detail="Cannot create room for another doctor")
        else:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Check if room already exists
    existing = db.query(models.ChatRoom).filter(
        models.ChatRoom.patient_id == data.patient_id,
        models.ChatRoom.doctor_id == data.doctor_id
    ).first()
    
    if existing:
        return _populate_room_data(existing, db)
    
    # Create new room
    room = models.ChatRoom(
        patient_id=data.patient_id,
        doctor_id=data.doctor_id
    )
    db.add(room)
    db.commit()
    db.refresh(room)
    
    # Send system message
    system_message = models.Message(
        room_id=room.room_id,
        sender_user_id=user.user_id,
        message_type=models.MessageType.SYSTEM,
        content="Chat room created. You can now start messaging!"
    )
    db.add(system_message)
    db.commit()
    
    return _populate_room_data(room, db)

@router.get("/rooms/{room_id}/messages")
def list_messages(
    room_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user)
):
    """List messages in a chat room with pagination"""
    _authorize_room_access(db, user, room_id)
    
    # Mark messages as read for this user
    role, patient, doctor = _get_user_role_and_ids(db, user)
    room = db.query(models.ChatRoom).filter(models.ChatRoom.room_id == room_id).first()
    
    if patient and room.patient_id == patient.patient_id:
        room.unread_count_patient = 0
    elif doctor and room.doctor_id == doctor.doctor_id:
        room.unread_count_doctor = 0
    
    db.commit()
    
    # Get messages with pagination
    offset = (page - 1) * limit
    messages = db.query(models.Message).filter(
        models.Message.room_id == room_id,
        models.Message.is_deleted == False
    ).order_by(desc(models.Message.created_at)).offset(offset).limit(limit).all()
    
    # Reverse to show oldest first
    messages.reverse()
    
    # Populate message data
    result = []
    for message in messages:
        message_data = _populate_message_data(message, db)
        result.append(message_data)
    
    return {
        "messages": result,
        "page": page,
        "limit": limit,
        "total": db.query(models.Message).filter(
            models.Message.room_id == room_id,
            models.Message.is_deleted == False
        ).count()
    }


@router.post("/rooms/{room_id}/messages", response_model=dict)
async def post_message(
    room_id: int,
    body: schemas.MessageCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user)
):
    """Post a new message to a chat room"""
    room = _authorize_room_access(db, user, room_id)
    
    # Validate content
    if body.message_type == "text" and not (body.content or "").strip():
        raise HTTPException(status_code=400, detail="Text message cannot be empty")
    
    if body.message_type in ["image", "file"] and not body.file_url:
        raise HTTPException(status_code=400, detail="File URL is required for file messages")
    
    # Create message
    message = models.Message(
        room_id=room_id,
        sender_user_id=user.user_id,
        message_type=body.message_type,
        content=body.content,
        file_url=body.file_url,
        file_name=body.file_name,
        file_size=body.file_size,
        reply_to_message_id=body.reply_to_message_id
    )
    db.add(message)
    
    # Update room's last message time and unread counts
    role, patient, doctor = _get_user_role_and_ids(db, user)
    room.last_message_at = datetime.utcnow()
    
    if patient and room.patient_id == patient.patient_id:
        # Patient sent message, increment doctor's unread count
        room.unread_count_doctor += 1
    elif doctor and room.doctor_id == doctor.doctor_id:
        # Doctor sent message, increment patient's unread count
        room.unread_count_patient += 1
    
    db.commit()
    db.refresh(message)
    
    # Prepare message data for broadcasting
    message_data = _populate_message_data(message, db)
    
    # Broadcast to room via WebSocket
    await manager.send_to_room(room_id, {
        "type": "new_message",
        "data": message_data
    })
    
    # Send push notification to other participants
    try:
        participants = []
        if patient and room.patient_id == patient.patient_id:
            # Message from patient, notify doctor
            doc_user = db.query(models.User).join(models.Doctor).filter(
                models.Doctor.doctor_id == room.doctor_id
            ).first()
            if doc_user:
                participants.append(doc_user.user_id)
        elif doctor and room.doctor_id == doctor.doctor_id:
            # Message from doctor, notify patient
            pat_user = db.query(models.User).join(models.Patient).filter(
                models.Patient.patient_id == room.patient_id
            ).first()
            if pat_user:
                participants.append(pat_user.user_id)
        
        for participant_id in participants:
            NotificationHub.send_many([participant_id], 'new_message', {
                'room_id': room_id,
                'message_id': message.message_id,
                'sender': user.username,
                'content': message.content[:100] if message.content else f"Sent a {message.message_type}"
            })
    except Exception as e:
        print(f"Failed to send notification: {e}")
    
    return message_data

@router.put("/messages/{message_id}", response_model=dict)
async def update_message(
    message_id: int,
    body: schemas.MessageUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user)
):
    """Update a message (edit or delete)"""
    message = db.query(models.Message).filter(models.Message.message_id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Authorization: Only sender can edit/delete their own messages
    if message.sender_user_id != user.user_id and user.role.upper() != "ADMIN":
        raise HTTPException(status_code=403, detail="Can only edit your own messages")
    
    # Check room access
    _authorize_room_access(db, user, message.room_id)
    
    # Update message
    if body.content is not None:
        message.content = body.content
        message.is_edited = True
        message.updated_at = datetime.utcnow()
    
    if body.is_deleted is not None:
        message.is_deleted = body.is_deleted
        message.updated_at = datetime.utcnow()
    
    db.commit()
    
    # Broadcast update to room
    message_data = _populate_message_data(message, db)
    await manager.send_to_room(message.room_id, {
        "type": "message_updated",
        "data": message_data
    })
    
    return message_data

@router.post("/messages/{message_id}/reactions", response_model=dict)
async def add_reaction(
    message_id: int,
    body: schemas.MessageReactionCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user)
):
    """Add a reaction to a message"""
    message = db.query(models.Message).filter(models.Message.message_id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Check room access
    _authorize_room_access(db, user, message.room_id)
    
    # Check if user already reacted with this emoji
    existing = db.query(models.MessageReaction).filter(
        models.MessageReaction.message_id == message_id,
        models.MessageReaction.user_id == user.user_id,
        models.MessageReaction.emoji == body.emoji
    ).first()
    
    if existing:
        # Remove reaction if it exists
        db.delete(existing)
        db.commit()
        action = "removed"
    else:
        # Add new reaction
        reaction = models.MessageReaction(
            message_id=message_id,
            user_id=user.user_id,
            emoji=body.emoji
        )
        db.add(reaction)
        db.commit()
        action = "added"
    
    # Broadcast reaction update
    message_data = _populate_message_data(message, db)
    await manager.send_to_room(message.room_id, {
        "type": "reaction_updated",
        "data": message_data
    })
    
    return {"status": action, "emoji": body.emoji}

@router.post("/rooms/{room_id}/upload", response_model=dict)
async def upload_file(
    room_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user)
):
    """Upload a file to a chat room"""
    _authorize_room_access(db, user, room_id)
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    max_size = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Upload to Azure Blob Storage
        if AZURE_AVAILABLE and upload_bytes_get_url:
            file_url = await upload_bytes_get_url(file_content, file.filename, "chat-files")
        else:
            # Fallback to local storage or skip file upload
            file_url = f"local://chat-files/{file.filename}"
        
        # Determine message type
        message_type = "image" if file.content_type and file.content_type.startswith("image/") else "file"
        
        return {
            "file_url": file_url,
            "file_name": file.filename,
            "file_size": len(file_content),
            "message_type": message_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.put("/users/{user_id}/status")
async def update_online_status(
    user_id: int,
    status: str,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user)
):
    """Update user's online status"""
    if user.user_id != user_id and user.role.upper() != "ADMIN":
        raise HTTPException(status_code=403, detail="Can only update your own status")
    
    # Update or create status
    user_status = db.query(models.UserOnlineStatus).filter(
        models.UserOnlineStatus.user_id == user_id
    ).first()
    
    if user_status:
        user_status.status = status
        user_status.last_activity = datetime.utcnow()
        if status == "offline":
            user_status.last_seen = datetime.utcnow()
    else:
        user_status = models.UserOnlineStatus(
            user_id=user_id,
            status=status,
            last_activity=datetime.utcnow(),
            last_seen=datetime.utcnow() if status == "offline" else None
        )
        db.add(user_status)
    
    db.commit()
    
    # Broadcast status update to relevant rooms
    rooms = db.query(models.ChatRoom).filter(
        or_(
            models.ChatRoom.patient_id.in_(
                db.query(models.Patient.patient_id).filter(models.Patient.user_id == user_id)
            ),
            models.ChatRoom.doctor_id.in_(
                db.query(models.Doctor.doctor_id).filter(models.Doctor.user_id == user_id)
            )
        )
    ).all()
    
    for room in rooms:
        await manager.send_to_room(room.room_id, {
            "type": "user_status_updated",
            "data": {
                "user_id": user_id,
                "status": status,
                "last_activity": user_status.last_activity.isoformat()
            }
        })
    
    return {"status": "updated"}

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: int = Query(...),
    token: str = Query(""),
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time messaging"""
    # Authenticate user
    try:
        payload = _decode_token(token)
        user_id = int(payload.get("sub", 0))
    except Exception:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        await websocket.close(code=4001, reason="User not found")
        return
    
    # Authorize room access
    try:
        _authorize_room_access(db, user, room_id)
    except HTTPException:
        await websocket.close(code=4003, reason="Access denied")
        return
    
    # Connect to room
    await manager.connect(websocket, room_id, user_id)
    
    # Update user status to online
    try:
        await update_online_status(user_id, "online", db, user)
    except:
        pass
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            if data.get("type") == "typing":
                # Broadcast typing indicator
                await manager.send_to_room(room_id, {
                    "type": "typing",
                    "data": {
                        "user_id": user_id,
                        "username": user.username,
                        "typing": data.get("typing", False)
                    }
                })
                continue
            
            # Handle regular message
            content = (data.get("content") or "").strip()
            message_type = data.get("message_type", "text")
            
            if message_type == "text" and not content:
                continue
            
            # Create message
            message = models.Message(
                room_id=room_id,
                sender_user_id=user_id,
                message_type=message_type,
                content=content,
                file_url=data.get("file_url"),
                file_name=data.get("file_name"),
                file_size=data.get("file_size"),
                reply_to_message_id=data.get("reply_to_message_id")
            )
            db.add(message)
            
            # Update room
            room = db.query(models.ChatRoom).filter(models.ChatRoom.room_id == room_id).first()
            room.last_message_at = datetime.utcnow()
            
            # Update unread counts
            role, patient, doctor = _get_user_role_and_ids(db, user)
            if patient and room.patient_id == patient.patient_id:
                room.unread_count_doctor += 1
            elif doctor and room.doctor_id == doctor.doctor_id:
                room.unread_count_patient += 1
            
            db.commit()
            db.refresh(message)
            
            # Broadcast message
            message_data = _populate_message_data(message, db)
            await manager.send_to_room(room_id, {
                "type": "new_message",
                "data": message_data
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Disconnect and update status
        manager.disconnect(websocket, room_id, user_id)
        try:
            await update_online_status(user_id, "offline", db, user)
        except:
            pass
