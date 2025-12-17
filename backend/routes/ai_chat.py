from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user

router = APIRouter()


@router.get("/sessions", response_model=list[schemas.AIChatSessionOut])
def list_sessions(db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    return (
        db.query(models.AIChatSession)
        .filter(models.AIChatSession.user_id == user.user_id)
        .order_by(models.AIChatSession.created_at.desc())
        .all()
    )


@router.post("/sessions", response_model=schemas.AIChatSessionOut)
def create_session(body: schemas.AIChatSessionCreate, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    title = (body.title or "Chat").strip() or "Chat"
    sess = models.AIChatSession(user_id=user.user_id, title=title)
    db.add(sess); db.commit(); db.refresh(sess)
    return sess


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    sess = db.query(models.AIChatSession).filter(models.AIChatSession.session_id == session_id, models.AIChatSession.user_id == user.user_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Not found")
    db.query(models.AIChatMessage).filter(models.AIChatMessage.session_id == session_id).delete()
    db.delete(sess)
    db.commit()
    return {"ok": True}


@router.get("/sessions/{session_id}/messages", response_model=list[schemas.AIChatMessageOut])
def list_messages(session_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    sess = db.query(models.AIChatSession).filter(models.AIChatSession.session_id == session_id, models.AIChatSession.user_id == user.user_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Not found")
    return (
        db.query(models.AIChatMessage)
        .filter(models.AIChatMessage.session_id == session_id)
        .order_by(models.AIChatMessage.created_at.asc())
        .all()
    )


@router.post("/sessions/{session_id}/messages", response_model=schemas.AIChatMessageOut)
def add_message(session_id: int, body: schemas.AIChatMessageIn, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    sess = db.query(models.AIChatSession).filter(models.AIChatSession.session_id == session_id, models.AIChatSession.user_id == user.user_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Not found")
    m = models.AIChatMessage(session_id=session_id, role=body.role, content=body.content)
    db.add(m); db.commit(); db.refresh(m)
    return m

