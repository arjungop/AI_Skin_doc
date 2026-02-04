from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models, schemas
from backend.auth import get_current_user_id
from typing import List
from datetime import datetime

router = APIRouter()

# --- Routine Items ---

@router.get("/", response_model=List[schemas.RoutineItemOut])
def get_routine(
    user_id: int = Depends(get_current_user_id), 
    db: Session = Depends(get_db)
):
    """Get all routine items"""
    items = db.query(models.RoutineItem)\
        .filter(models.RoutineItem.user_id == user_id, models.RoutineItem.is_active == True)\
        .order_by(models.RoutineItem.step_order.asc())\
        .all()
    return items

@router.post("/", response_model=schemas.RoutineItemOut)
def add_routine_item(
    item_data: schemas.RoutineItemCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Add a product to the routine"""
    new_item = models.RoutineItem(
        user_id=user_id,
        **item_data.model_dump()
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item

@router.delete("/{item_id}")
def delete_routine_item(
    item_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Remove an item from routine"""
    item = db.query(models.RoutineItem).filter(models.RoutineItem.item_id == item_id, models.RoutineItem.user_id == user_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    item.is_active = False # Soft delete
    db.commit()
    return {"status": "success"}

# --- Completions ---

@router.get("/completions", response_model=List[schemas.RoutineCompletionOut])
def get_completions(
    date: str, # YYYY-MM-DD
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Get completions for a specific date"""
    # Parse date string to verify format/params
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
    # Join with RoutineItem to ensure user ownership
    completions = db.query(models.RoutineCompletion)\
        .join(models.RoutineItem)\
        .filter(
            models.RoutineItem.user_id == user_id,
            # We cast the datetime to date for comparison, depending on DB this might need adjustment
            # detailed implementation: filter where date component matches
        )\
        .all()
    
    # Simple post-filtering for the specific date if DB specific SQL is complex
    # In production, use proper SQLAlchemy date casting
    filtered = [c for c in completions if c.date.date() == target_date]
    
    return filtered

@router.post("/check", response_model=schemas.RoutineCompletionOut)
def toggle_completion(
    completion_data: schemas.RoutineCompletionCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """Mark a routine item as done"""
    # Verify ownership
    item = db.query(models.RoutineItem).filter(models.RoutineItem.item_id == completion_data.routine_item_id, models.RoutineItem.user_id == user_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Routine item not found")
    
    # Check if already exists for this date
    # In this simple logic, we just add a new completion entry. 
    # Front-end should handle untoggling by not sending or we implement a delete endpoint.
    # For now, let's assume this endpoint creates the completion.
    
    new_completion = models.RoutineCompletion(
        routine_item_id=completion_data.routine_item_id,
        date=completion_data.date, # passed as datetime from frontend
        status=completion_data.status
    )
    db.add(new_completion)
    db.commit()
    db.refresh(new_completion)
    return new_completion
