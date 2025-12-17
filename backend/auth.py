from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, database, crud
from pydantic import BaseModel, EmailStr
from .security import create_access_token
from .security import get_current_user
from datetime import datetime

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == req.email).first()
    if not user or not crud.verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Block login for users awaiting approval or disabled
    role_upper = (user.role or '').upper()
    if role_upper == 'PENDING_DOCTOR':
        raise HTTPException(status_code=403, detail="Your doctor application is awaiting admin approval")
    # Enforce account status at login (avoid issuing tokens)
    try:
        st = db.query(models.UserStatus).filter(models.UserStatus.user_id == user.user_id).first()
        if st and (st.status or '').upper() in {"SUSPENDED", "TERMINATED"}:
            raise HTTPException(status_code=403, detail=f"Account {st.status.lower()}")
    except Exception:
        pass

    # Return role-specific payload
    payload = {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
    }
    if user.role and user.role.upper() == "PATIENT":
        patient = db.query(models.Patient).filter(models.Patient.user_id == user.user_id).first()
        if patient:
            payload.update({
                "patient_id": patient.patient_id,
                "first_name": patient.first_name,
                "last_name": patient.last_name,
            })
    if user.role and user.role.upper() == "DOCTOR":
        doctor = db.query(models.Doctor).filter(models.Doctor.user_id == user.user_id).first()
        if doctor:
            payload.update({
                "doctor_id": doctor.doctor_id,
                "specialization": doctor.specialization,
            })
    # Issue JWT with token version
    tv = 1
    try:
        tv_row = db.query(models.UserTokenVersion).filter(models.UserTokenVersion.user_id == user.user_id).first()
        if not tv_row:
            tv_row = models.UserTokenVersion(user_id=user.user_id, version=1)
            db.add(tv_row); db.commit()
        tv = int(tv_row.version or 1)
    except Exception:
        tv = 1
    token_claims = {"sub": str(user.user_id), "role": user.role, "tv": tv, "iat": int(datetime.utcnow().timestamp())}
    if payload.get("patient_id"):
        token_claims["patient_id"] = payload["patient_id"]
    if payload.get("doctor_id"):
        token_claims["doctor_id"] = payload["doctor_id"]
    token = create_access_token(token_claims)
    payload.update({"access_token": token, "token_type": "bearer"})
    return payload


class ForgotRequest(BaseModel):
    email: EmailStr


@router.post("/forgot")
def forgot_password(req: ForgotRequest):
    # In production: generate token, email user a reset link.
    # For now we simply acknowledge the request to avoid leaking user existence.
    return {"ok": True}


@router.get("/me")
def me(user: models.User = Depends(get_current_user)):
    # Echo minimal details and, if close to expiry, issue a fresh token
    payload = {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
    }
    claims = getattr(user, 'token_claims', {})
    if isinstance(claims, dict):
        payload["claims"] = {k: claims.get(k) for k in ("sub", "role", "patient_id", "doctor_id", "exp")}
        try:
            from datetime import datetime, timezone
            exp = claims.get("exp")
            now_ts = datetime.now(timezone.utc).timestamp()
            exp_ts = float(exp) if isinstance(exp, (int, float)) else float(exp or 0)
            # If token expires within 10 minutes, refresh it
            if exp_ts and exp_ts - now_ts < 600:
                new_claims = {k: v for k, v in claims.items() if k not in {"exp"}}
                token = create_access_token(new_claims)
                payload["access_token"] = token
                payload["token_type"] = "bearer"
        except Exception:
            pass
    return payload


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@router.post("/change_password")
def change_password(req: ChangePasswordRequest, db: Session = Depends(database.get_db), user: models.User = Depends(get_current_user)):
    if not crud.verify_password(req.old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password incorrect")
    if len(req.new_password or '') < 8:
        raise HTTPException(status_code=400, detail="New password too short")
    user.hashed_password = crud.hash_password(req.new_password)
    db.add(user); db.commit()
    return {"ok": True}


@router.post("/logout_all")
def logout_all(db: Session = Depends(database.get_db), user: models.User = Depends(get_current_user)):
    # Increment user's token version to invalidate all existing tokens
    row = db.query(models.UserTokenVersion).filter(models.UserTokenVersion.user_id == user.user_id).first()
    if not row:
        row = models.UserTokenVersion(user_id=user.user_id, version=2)
        db.add(row)
    else:
        row.version = (row.version or 1) + 1
    row.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}
