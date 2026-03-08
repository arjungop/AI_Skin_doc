"""
Doctor Suggestions + UV Risk endpoints.

Replaces the embedding-based product recommendation engine with:
1. Doctor-created product suggestions tied to diagnosis reports
2. UV risk assessment with Fitzpatrick-aware advice
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user
from typing import List

router = APIRouter()


# ── Doctor Product Suggestions ───────────────────────────────────────────

@router.post("/suggest", response_model=schemas.DoctorSuggestionOut)
def add_suggestion(
    data: schemas.DoctorSuggestionCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Doctor adds a product suggestion to a diagnosis report."""
    role = (user.role or "").upper()
    if role not in ("DOCTOR", "ADMIN"):
        raise HTTPException(status_code=403, detail="Only doctors can add product suggestions")

    # Resolve doctor_id
    doctor = db.query(models.Doctor).filter(
        models.Doctor.user_id == user.user_id
    ).first()
    if not doctor and role != "ADMIN":
        raise HTTPException(status_code=403, detail="Doctor profile not found")

    doctor_id = doctor.doctor_id if doctor else 0

    # Verify report exists
    report = db.query(models.DiagnosisReport).filter(
        models.DiagnosisReport.report_id == data.report_id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Diagnosis report not found")

    suggestion = models.DoctorSuggestion(
        report_id=data.report_id,
        doctor_id=doctor_id,
        product_name=data.product_name,
        product_link=data.product_link,
        notes=data.notes,
    )
    db.add(suggestion)
    db.commit()
    db.refresh(suggestion)
    return suggestion


@router.get("/suggestions/{patient_id}", response_model=List[schemas.DoctorSuggestionOut])
def get_suggestions(
    patient_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Get all product suggestions for a patient (from all their reports)."""
    role = (user.role or "").upper()
    if role != "ADMIN":
        # Patients can see their own; doctors can see their patients'
        patient = db.query(models.Patient).filter(
            models.Patient.user_id == user.user_id
        ).first()
        doctor = db.query(models.Doctor).filter(
            models.Doctor.user_id == user.user_id
        ).first()
        if not (
            (patient and patient.patient_id == patient_id)
            or doctor  # doctors can view any patient's suggestions
        ):
            raise HTTPException(status_code=403, detail="Access denied")

    # Get all reports for this patient, then their suggestions
    report_ids = [
        r.report_id for r in
        db.query(models.DiagnosisReport.report_id)
        .filter(models.DiagnosisReport.patient_id == patient_id)
        .all()
    ]

    if not report_ids:
        return []

    suggestions = (
        db.query(models.DoctorSuggestion)
        .filter(models.DoctorSuggestion.report_id.in_(report_ids))
        .order_by(models.DoctorSuggestion.created_at.desc())
        .all()
    )
    return suggestions


# ── Public Weather Context ───────────────────────────────────────────────

import time as _time
import threading as _threading

_weather_cache: dict = {}  # {city_lower: (timestamp, data)}
_weather_lock = _threading.Lock()
_CACHE_TTL = 300  # 5 minutes


@router.get("/context")
def get_weather_context(city: str = "Coimbatore"):
    """
    Public endpoint returning current weather data for a city.
    Used by the WeatherWidget on the coach page.
    Results are cached for 5 minutes to avoid rate-limiting the external API.
    """
    key = city.strip().lower()
    now = _time.time()

    with _weather_lock:
        if key in _weather_cache:
            ts, cached = _weather_cache[key]
            if now - ts < _CACHE_TTL:
                return cached

    from backend.ml.environment import env_adapter
    context = env_adapter.get_weather_context(city)
    if not context:
        raise HTTPException(status_code=502, detail="Could not fetch weather data")

    result = {
        "city": context.get("city", city),
        "temp_c": context.get("temp_c"),
        "humidity": context.get("humidity"),
        "uv_index": context.get("uv_index", 0),
        "description": context.get("description", ""),
    }

    with _weather_lock:
        _weather_cache[key] = (now, result)
    return result


# ── AI Product Recommendation (via OpenRouter) ──────────────────────────

import os as _os
import requests as _requests
import json as _json
import logging as _logging

_rec_logger = _logging.getLogger(__name__)

_OPENROUTER_KEY = _os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL = _os.getenv("OPENROUTER_MODEL", "xiaomi/mimo-v2-flash:free")

_PRODUCT_SYSTEM_PROMPT = """You are a dermatology-trained skincare product recommender.
Given a user query about skincare needs, return a JSON array of 3-5 product recommendations.
Each item must have these fields:
- "id": a sequential integer starting at 1
- "name": product name (real products)
- "brand": brand name
- "score": relevance score between 0.0 and 1.0
- "reason": one sentence why this product fits the query

Return ONLY the raw JSON array, no markdown fences, no explanation."""

from pydantic import BaseModel as _BaseModel


class RecommendBody(_BaseModel):
    query: str
    condition: str = ""


@router.post("/recommend")
def recommend_products(body: RecommendBody):
    """
    AI-powered product recommendation using OpenRouter LLM.
    Returns a list of product suggestions based on the user's query.
    """
    if not _OPENROUTER_KEY:
        raise HTTPException(status_code=503, detail="Product recommendation service not configured (missing OPENROUTER_API_KEY)")

    user_msg = body.query.strip()
    if body.condition:
        user_msg += f" (skin condition: {body.condition})"

    try:
        resp = _requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": _PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            _rec_logger.warning("OpenRouter returned %s: %s", resp.status_code, resp.text[:300])
            raise HTTPException(status_code=502, detail="AI service temporarily unavailable")

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "[]")

        # Strip markdown fences if the model wraps them
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        products = _json.loads(content)

        # Normalize: ensure each product has required fields
        results = []
        for i, p in enumerate(products[:5]):
            results.append({
                "id": p.get("id", i + 1),
                "name": p.get("name", "Unknown Product"),
                "brand": p.get("brand", ""),
                "score": min(1.0, max(0.0, float(p.get("score", 0.8)))),
                "image": None,
            })

        return results

    except HTTPException:
        raise
    except _json.JSONDecodeError:
        _rec_logger.warning("Failed to parse LLM response as JSON: %s", content[:200])
        return []
    except Exception as e:
        _rec_logger.error("Product recommendation failed: %s", e)
        raise HTTPException(status_code=502, detail="AI service error")


# ── UV Risk Assessment ───────────────────────────────────────────────────

@router.get("/uv-risk")
def get_uv_risk(
    city: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """
    Get UV risk assessment with Fitzpatrick-aware recommendations.
    Auto-detects city from user profile if not provided.
    """
    # Auto-detect city from user profile
    lookup_city = city
    fitzpatrick = None

    try:
        profile = db.query(models.UserProfile).filter(
            models.UserProfile.user_id == user.user_id
        ).first()
        if profile:
            if not lookup_city and profile.location_city:
                lookup_city = profile.location_city
            fitzpatrick = profile.fitzpatrick_type
    except Exception:
        pass

    if not lookup_city:
        raise HTTPException(
            status_code=400,
            detail="City not provided and not set in your profile. Update your profile with a city or pass ?city=CityName"
        )

    # Fetch weather data
    from backend.ml.environment import env_adapter
    context = env_adapter.get_weather_context(lookup_city)
    if not context:
        raise HTTPException(status_code=502, detail="Could not fetch weather data")

    uv = context.get("uv_index", 0)

    # Fitzpatrick-aware risk assessment
    risk = _assess_uv_risk(uv, fitzpatrick)

    return {
        "city": context.get("city", lookup_city),
        "uv_index": uv,
        "temp_c": context.get("temp_c"),
        "humidity": context.get("humidity"),
        "risk_level": risk["level"],
        "spf_recommendation": risk["spf"],
        "advice": risk["advice"],
        "fitzpatrick_type": fitzpatrick,
        "exposure_limit_minutes": risk["exposure_minutes"],
    }


def _assess_uv_risk(uv: float, fitzpatrick: int | None) -> dict:
    """Return risk level, SPF recommendation, and advice based on UV and skin type."""

    # Fitzpatrick types 1-2 burn easily; 5-6 rarely burn
    # Lower types need MORE protection at LOWER UV levels
    burn_threshold_offset = 0
    if fitzpatrick:
        if fitzpatrick <= 2:
            burn_threshold_offset = -2  # triggers risk 2 points earlier
        elif fitzpatrick == 3:
            burn_threshold_offset = -1
        elif fitzpatrick >= 5:
            burn_threshold_offset = 1  # slightly more tolerant

    effective_uv = uv - burn_threshold_offset  # higher = more dangerous for this person

    if effective_uv <= 2:
        return {
            "level": "Low",
            "spf": "SPF 15 if outdoors for extended periods",
            "advice": "Low UV exposure today. No special precautions needed for brief outdoor time.",
            "exposure_minutes": 60,
        }
    elif effective_uv <= 5:
        return {
            "level": "Moderate",
            "spf": "SPF 30 recommended",
            "advice": "Moderate UV levels. Wear sunscreen if spending more than 30 minutes outdoors. Seek shade during midday.",
            "exposure_minutes": 30,
        }
    elif effective_uv <= 7:
        return {
            "level": "High",
            "spf": "SPF 50+ required",
            "advice": "High UV exposure risk. Apply broad-spectrum SPF 50+ sunscreen every 2 hours. Wear protective clothing and a hat.",
            "exposure_minutes": 15,
        }
    elif effective_uv <= 10:
        return {
            "level": "Very High",
            "spf": "SPF 50+ with reapplication every 90 minutes",
            "advice": "Very high UV risk. Minimize outdoor time between 10 AM and 4 PM. Full sun protection is essential.",
            "exposure_minutes": 10,
        }
    else:
        return {
            "level": "Extreme",
            "spf": "SPF 50+ — avoid direct sun",
            "advice": "Extreme UV levels. Avoid outdoor exposure if possible. If unavoidable, use maximum protection: SPF 50+, wide-brim hat, UV-blocking clothing.",
            "exposure_minutes": 5,
        }
