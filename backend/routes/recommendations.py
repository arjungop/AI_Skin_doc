
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.ml.recommender import recommender
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class ProductRecommendationRequest(BaseModel):
    query: str
    condition: Optional[str] = None # e.g. "Eczema"
    profile_id: Optional[int] = None

class ProductResponse(BaseModel):
    id: str
    name: str
    brand: str
    image: Optional[str]
    score: float

from backend import models
from backend.security import get_current_user

@router.post("/recommend", response_model=List[ProductResponse])
def recommend_products(req: ProductRecommendationRequest, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    """
    Get personalized product recommendations.
    """
    search_query = req.query
    condition_text = req.condition or ""
    
    # Personalization Injection
    try:
        profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user.user_id).first()
        if profile:
            extras = []
            if profile.skin_type:
                extras.append(f"{profile.skin_type} skin")
            if profile.goals: # "Acne", "Aging" etc
                extras.append(str(profile.goals).replace("[","").replace("]","").replace('"',""))
            if extras:
                condition_text += " " + " ".join(extras)
    except Exception:
        pass # Fallback to generic search if profile fails
        
    # Run search
    try:
        results = recommender.search(search_query, condition_text=condition_text.strip() or None)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context")
def get_environmental_context(city: str):
    """
    Get environmental factors (UV, Humidity) for a city.
    """
    from backend.ml.environment import env_adapter
    context = env_adapter.get_weather_context(city)
    if not context:
        raise HTTPException(status_code=404, detail="Could not fetch weather data")
    return context
