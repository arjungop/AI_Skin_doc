import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
try:
    from jose.exceptions import ExpiredSignatureError, JWTClaimsError  # type: ignore
except Exception:  # fallback names
    ExpiredSignatureError = JWTError  # type: ignore
    JWTClaimsError = JWTError  # type: ignore
from sqlalchemy.orm import Session

from .database import get_db
from . import models


JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    import warnings
    warnings.warn(
        "JWT_SECRET is not set! Using an insecure default for LOCAL DEV ONLY.",
        stacklevel=2,
    )
    JWT_SECRET = "dev-secret-change-me-in-production"
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
JWT_LEEWAY_SECONDS = int(os.getenv("JWT_LEEWAY_SECONDS", "300"))  # tolerate small clock drift

bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    # Ensure 'sub' is a string to satisfy some JWT validators
    if to_encode.get("sub") is not None and not isinstance(to_encode.get("sub"), str):
        try:
            to_encode["sub"] = str(to_encode.get("sub"))
        except Exception:
            pass
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    # Store exp as numeric timestamp for widest compatibility
    to_encode.update({"exp": int(expire.timestamp())})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def _decode_token(token: str) -> Dict[str, Any]:
    try:
        # Always verify signature — no bypass allowed
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALG],
            options={
                "verify_exp": False,
                "verify_aud": False,
                "verify_iss": False,
            },
        )
        exp = payload.get("exp")
        if exp is not None:
            exp_ts: float
            if isinstance(exp, (int, float)):
                exp_ts = float(exp)
            elif isinstance(exp, str):
                # try numeric string, else ISO8601
                try:
                    exp_ts = float(exp)
                except Exception:
                    try:
                        exp_dt = datetime.fromisoformat(exp)
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                        exp_ts = exp_dt.timestamp()
                    except Exception:
                        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
            elif isinstance(exp, datetime):
                exp_dt = exp
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                exp_ts = exp_dt.timestamp()
            else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts > exp_ts + JWT_LEEWAY_SECONDS:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        return payload
    except ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from e
    except JWTClaimsError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token claims") from e
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from e


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = _decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.query(models.User).filter(models.User.user_id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    # Enforce token version if available
    try:
        tv_claim = payload.get('tv')
        if tv_claim is not None:
            row = db.query(models.UserTokenVersion).filter(models.UserTokenVersion.user_id == user.user_id).first()
            tv_db = int(row.version) if row and row.version is not None else 1
            if int(tv_claim) != tv_db:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired; please sign in again")
    except HTTPException:
        raise
    except Exception:
        pass
    # Enforce account status (allow admins even if suspended/terminated)
    try:
        st = db.query(models.UserStatus).filter(models.UserStatus.user_id == user.user_id).first()
        if st and (st.status or '').upper() in {"SUSPENDED", "TERMINATED"}:
            if (user.role or '').upper() != 'ADMIN':
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Account {st.status.lower()}")
    except Exception:
        pass
    # attach token claims for downstream use
    user.token_claims = payload  # type: ignore
    return user


def require_roles(*roles: str) -> Callable:
    def _dep(user: models.User = Depends(get_current_user)):
        if not roles:
            return user
        role = (user.role or "").upper()
        allowed = {r.upper() for r in roles}
        if role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user
    return _dep


def verify_patient_access(patient_id: int, user: models.User, db: Session) -> None:
    """Enforce row-level access: patients own their data, doctors only see linked patients.

    Raises HTTPException(403) if the user has no legitimate access to this patient.
    Admins bypass all checks.
    """
    role = (user.role or "").upper()

    if role == "ADMIN":
        return  # admins have full access

    if role == "PATIENT":
        patient = db.query(models.Patient).filter(
            models.Patient.user_id == user.user_id
        ).first()
        if not patient or patient.patient_id != patient_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot access another patient's data",
            )
        return

    if role == "DOCTOR":
        doctor = db.query(models.Doctor).filter(
            models.Doctor.user_id == user.user_id
        ).first()
        if not doctor:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Doctor profile not found",
            )
        link = db.query(models.DoctorPatient).filter(
            models.DoctorPatient.doctor_id == doctor.doctor_id,
            models.DoctorPatient.patient_id == patient_id,
        ).first()
        if not link:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not linked to this patient",
            )
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Insufficient permissions",
    )
