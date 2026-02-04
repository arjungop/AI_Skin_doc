from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models, schemas
from backend.auth import get_current_user_id
from datetime import datetime

router = APIRouter()

@router.get("/me", response_model=schemas.UserProfileOut)
def get_my_profile(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    """Get current user's personalization profile"""
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found. Please complete onboarding.")
    return profile

@router.put("/me", response_model=schemas.UserProfileOut)
def update_my_profile(
    profile_data: schemas.UserProfileUpdate, 
    user_id: int = Depends(get_current_user_id), 
    db: Session = Depends(get_db)
):
    """Create or Update user profile"""
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()
    
    if not profile:
        # Create new
        new_profile = models.UserProfile(
            user_id=user_id,
            **profile_data.model_dump(exclude_unset=True)
        )
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        return new_profile
    else:
        # Update existing
        for key, value in profile_data.model_dump(exclude_unset=True).items():
            setattr(profile, key, value)
        
        profile.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(profile)
        return profile
