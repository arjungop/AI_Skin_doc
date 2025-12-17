from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models
from backend.security import get_current_user

router = APIRouter()


@router.post("/support/ticket")
def create_ticket(payload: dict, db: Session = Depends(get_db), user: models.User | None = Depends(get_current_user)):
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip()
    subject = (payload.get("subject") or "").strip()
    message = (payload.get("message") or "").strip()
    if not email or not message:
        raise HTTPException(status_code=400, detail="Email and message are required")
    uid = getattr(user, 'user_id', None) if user else None
    row = models.SupportTicket(user_id=uid, name=name, email=email, subject=subject, message=message)
    db.add(row)
    db.commit()
    return {"ok": True}


@router.post("/support/newsletter/subscribe")
def subscribe_newsletter(payload: dict, db: Session = Depends(get_db)):
    email = (payload.get("email") or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    # upsert-like behavior: ignore duplicates
    existing = db.query(models.NewsletterSubscriber).filter(models.NewsletterSubscriber.email == email).first()
    if not existing:
        db.add(models.NewsletterSubscriber(email=email))
        try:
            db.commit()
        except Exception:
            db.rollback()
    return {"ok": True}
