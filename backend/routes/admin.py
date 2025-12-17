from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models
from backend.security import require_roles
from fastapi.responses import StreamingResponse
import io
import csv
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


@router.get("/doctor_applications")
def list_doctor_applications(
    status: str | None = None,
    q: str | None = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    user=Depends(require_roles("ADMIN"))
):
    query = db.query(models.DoctorApplication)
    if status:
        query = query.filter(models.DoctorApplication.status == status.upper())
    if q:
        like = f"%{q}%"
        # join users for email/username search
        from sqlalchemy.orm import aliased
        U = models.User
        query = query.join(U, U.user_id == models.DoctorApplication.user_id)
        query = query.filter((U.username.ilike(like)) | (U.email.ilike(like)) | (models.DoctorApplication.first_name.ilike(like)) | (models.DoctorApplication.last_name.ilike(like)))
    total = query.count()
    apps = query.order_by(models.DoctorApplication.created_at.desc()).offset(max(0,(page-1)*page_size)).limit(page_size).all()
    # Attach username/email
    out = []
    for a in apps:
        user = db.query(models.User).filter(models.User.user_id == a.user_id).first()
        out.append({
            "application_id": a.application_id,
            "user_id": a.user_id,
            "username": user.username if user else "",
            "email": user.email if user else "",
            "first_name": a.first_name,
            "last_name": a.last_name,
            "specialization": a.specialization,
            "license_no": a.license_no,
            "department": a.hospital,
            "status": a.status,
            "created_at": a.created_at,
        })
    return {"items": out, "page": page, "page_size": page_size, "total": total}


@router.get("/doctor_applications/export.csv")
def export_doctor_applications_csv(
    status: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
    _=Depends(require_roles("ADMIN"))
):
    data = list_doctor_applications(status, q, 1, 1000, db)  # type: ignore
    items = data["items"] if isinstance(data, dict) else data
    buf = io.StringIO()
    fields = ["application_id","user_id","username","email","first_name","last_name","specialization","license_no","department","status","created_at"]
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for r in items:
        w.writerow({k: r.get(k) for k in fields})
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type='text/csv', headers={"Content-Disposition": "attachment; filename=doctor_applications.csv"})


@router.post("/doctor_applications/{application_id}/approve")
def approve_doctor_application(application_id: int, db: Session = Depends(get_db), user=Depends(require_roles("ADMIN"))):
    app = db.query(models.DoctorApplication).filter(models.DoctorApplication.application_id == application_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    if app.status == "APPROVED":
        return {"message": "Already approved"}

    user = db.query(models.User).filter(models.User.user_id == app.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create doctor profile
    doctor = models.Doctor(user_id=user.user_id, specialization=app.specialization)
    db.add(doctor)
    # Update statuses
    user.role = "DOCTOR"
    # Set display name to first + last for better UX
    try:
        if app.first_name or app.last_name:
            user.username = f"{(app.first_name or '').strip()} {(app.last_name or '').strip()}".strip() or user.username
    except Exception:
        pass
    app.status = "APPROVED"
    db.commit()
    # Audit
    try:
        db.add(models.AuditLog(user_id=user.user_id, action="APPROVE_DOCTOR_APPLICATION", meta=str(application_id)))
        db.commit()
    except Exception:
        db.rollback()
    return {"message": "Approved", "doctor_id": doctor.doctor_id}


@router.post("/doctor_applications/{application_id}/reject")
def reject_doctor_application(application_id: int, db: Session = Depends(get_db), user=Depends(require_roles("ADMIN"))):
    app = db.query(models.DoctorApplication).filter(models.DoctorApplication.application_id == application_id).first()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    app.status = "REJECTED"
    # Keep user role as PENDING_DOCTOR or set to USER
    user = db.query(models.User).filter(models.User.user_id == app.user_id).first()
    if user and user.role == "PENDING_DOCTOR":
        user.role = "USER"
    db.commit()
    try:
        db.add(models.AuditLog(user_id=user.user_id, action="REJECT_DOCTOR_APPLICATION", meta=str(application_id)))
        db.commit()
    except Exception:
        db.rollback()
    return {"message": "Rejected"}


@router.get("/overview")
def overview(db: Session = Depends(get_db), user=Depends(require_roles("ADMIN"))):
    counts = {
        "users": db.query(models.User).count(),
        "patients": db.query(models.Patient).count(),
        # Prefer counting actual doctor profiles; if zero, fall back to role count
        "doctors": (db.query(models.Doctor).count() or db.query(models.User).filter(models.User.role == 'DOCTOR').count()),
        "appointments": db.query(models.Appointment).count(),
        "lesions": db.query(models.Lesion).count(),
        "transactions": db.query(models.Transaction).count(),
    }
    return counts


@router.post("/sync_doctors")
def sync_doctors(db: Session = Depends(get_db), _=Depends(require_roles("ADMIN"))):
    """Backfill doctor profiles for users whose role is DOCTOR but lack a doctors row."""
    # Find DOCTOR users without matching doctor profile
    from sqlalchemy import not_, exists
    U = models.User
    D = models.Doctor
    missing = (
        db.query(U)
        .filter(U.role == 'DOCTOR')
        .filter(~exists().where(D.user_id == U.user_id))
        .all()
    )
    created = 0
    for u in missing:
        try:
            db.add(models.Doctor(user_id=u.user_id, specialization=None))
            created += 1
        except Exception:
            db.rollback()
    db.commit()
    try:
        db.add(models.AuditLog(user_id=None, action="SYNC_DOCTORS", meta=f"created={created}"))
        db.commit()
    except Exception:
        db.rollback()
    return {"created": created}


# Users management
@router.get("/users")
def list_users(
    q: str | None = None,
    role: str | None = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _=Depends(require_roles("ADMIN"))
):
    query = db.query(models.User)
    if role:
        query = query.filter(models.User.role == role.upper())
    if q:
        like = f"%{q}%"
        query = query.filter((models.User.username.ilike(like)) | (models.User.email.ilike(like)))
    total = query.count()
    users = query.order_by(models.User.user_id.asc()).offset(max(0,(page-1)*page_size)).limit(page_size).all()
    out = [
        {
            "user_id": u.user_id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
        }
        for u in users
    ]
    return {"items": out, "page": page, "page_size": page_size, "total": total}


class RoleBody(BaseModel):
    role: str


class StatusBody(BaseModel):
    status: str  # ACTIVE | SUSPENDED


class TerminateBody(BaseModel):
    reason_code: str
    reason_text: str | None = None


@router.patch("/users/{user_id}/role")
def update_user_role(user_id: int, body: RoleBody, db: Session = Depends(get_db), who=Depends(require_roles("ADMIN"))):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = (body.role or "").upper()
    db.add(user)
    db.commit()
    # Audit
    try:
        db.add(models.AuditLog(user_id=who.user_id, action="UPDATE_USER_ROLE", meta=f"{user_id}:{user.role}"))
        db.commit()
    except Exception:
        db.rollback()
    return {"ok": True}


@router.patch("/users/{user_id}/status")
def update_user_status(user_id: int, body: StatusBody, db: Session = Depends(get_db), who=Depends(require_roles("ADMIN"))):
    st = db.query(models.UserStatus).filter(models.UserStatus.user_id == user_id).first()
    if not st:
        st = models.UserStatus(user_id=user_id)
        db.add(st)
    st.status = (body.status or 'ACTIVE').upper()
    if st.status == 'ACTIVE':
        st.terminated_at = None; st.terminated_by = None; st.termination_reason = None
    db.commit()
    try:
        db.add(models.AuditLog(user_id=who.user_id, action="UPDATE_USER_STATUS", meta=f"{user_id}:{st.status}"))
        db.commit()
    except Exception:
        db.rollback()
    return {"ok": True}


@router.post("/users/{user_id}/terminate")
def terminate_user(user_id: int, body: TerminateBody, db: Session = Depends(get_db), who=Depends(require_roles("ADMIN"))):
    u = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    st = db.query(models.UserStatus).filter(models.UserStatus.user_id == user_id).first()
    if not st:
        st = models.UserStatus(user_id=user_id)
        db.add(st)
    st.status = 'TERMINATED'
    st.terminated_at = datetime.utcnow()
    st.terminated_by = who.user_id
    st.termination_reason = f"{body.reason_code}: {(body.reason_text or '').strip()}"
    # Cancel upcoming appointments
    pat = db.query(models.Patient).filter(models.Patient.user_id == user_id).first()
    if pat:
        upcoming = db.query(models.Appointment).filter(models.Appointment.patient_id == pat.patient_id, models.Appointment.appointment_date >= datetime.utcnow()).all()
        for app in upcoming:
            app.status = 'Cancelled'
            db.add(app)
    db.commit()
    try:
        db.add(models.AuditLog(user_id=who.user_id, action="TERMINATE_USER", meta=f"{user_id}:{st.termination_reason}"))
        db.commit()
    except Exception:
        db.rollback()
    return {"ok": True}


@router.get("/doctors")
def list_doctors_admin(
    q: str | None = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _=Depends(require_roles("ADMIN"))
):
    from sqlalchemy import or_
    query = db.query(models.Doctor).join(models.User, models.User.user_id == models.Doctor.user_id)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(models.User.username.ilike(like), models.User.email.ilike(like), models.Doctor.specialization.ilike(like)))
    total = query.count()
    rows = query.order_by(models.Doctor.doctor_id.asc()).offset(max(0,(page-1)*page_size)).limit(page_size).all()
    out = []
    for d in rows:
        out.append({
            "doctor_id": d.doctor_id,
            "user_id": d.user_id,
            "username": getattr(d.user, 'username', ''),
            "email": getattr(d.user, 'email', ''),
            "specialization": d.specialization,
        })
    return {"items": out, "page": page, "page_size": page_size, "total": total}


@router.get("/audit_logs")
def list_audit_logs(page: int = 1, page_size: int = 50, db: Session = Depends(get_db), _=Depends(require_roles("ADMIN"))):
    q = db.query(models.AuditLog).order_by(models.AuditLog.created_at.desc())
    total = q.count()
    rows = q.offset(max(0,(page-1)*page_size)).limit(page_size).all()
    out = [
        {
            "id": r.id,
            "user_id": r.user_id,
            "action": r.action,
            "meta": r.meta,
            "created_at": r.created_at,
        }
        for r in rows
    ]
    return {"items": out, "page": page, "page_size": page_size, "total": total}


@router.get("/settings")
def get_settings(db: Session = Depends(get_db), _=Depends(require_roles("ADMIN"))):
    rows = db.query(models.Setting).all()
    return {s.key: s.value for s in rows}


@router.post("/settings")
def set_settings(data: dict, db: Session = Depends(get_db), who=Depends(require_roles("ADMIN"))):
    for k, v in (data or {}).items():
        row = db.query(models.Setting).filter(models.Setting.key == k).first()
        if not row:
            row = models.Setting(key=k, value=str(v))
            db.add(row)
        else:
            row.value = str(v)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
    # Audit
    try:
        db.add(models.AuditLog(user_id=who.user_id, action="UPDATE_SETTINGS", meta=str(list((data or {}).keys()))))
        db.commit()
    except Exception:
        db.rollback()
    return {"ok": True}
