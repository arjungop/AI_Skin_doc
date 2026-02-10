from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models, schemas
from backend.auth import get_current_user_id
from typing import List

router = APIRouter()

@router.get("/", response_model=List[schemas.SkinLogOut])
def get_skin_journey(
    user_id: int = Depends(get_current_user_id), 
    db: Session = Depends(get_db)
):
    """Get all skin journey logs for the current user, ordered by date"""
    logs = db.query(models.SkinLog)\
        .filter(models.SkinLog.user_id == user_id)\
        .order_by(models.SkinLog.created_at.desc())\
        .all()
    return logs

@router.post("/", response_model=schemas.SkinLogOut)
def add_skin_log(
    log_data: schemas.SkinLogCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Add a new entry to the skin journey"""
    new_log = models.SkinLog(
        user_id=user_id,
        **log_data.model_dump()
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    return new_log

@router.delete("/{log_id}")
def delete_skin_log(
    log_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Delete a skin journey log entry"""
    log = db.query(models.SkinLog).filter(
        models.SkinLog.log_id == log_id,
        models.SkinLog.user_id == user_id
    ).first()
    
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    db.delete(log)
    db.commit()
    return {"message": "Log deleted successfully"}
