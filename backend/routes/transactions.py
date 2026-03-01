"""
Simplified transaction / billing log.

Provides:
    POST /              – Admin creates a billing entry
    GET  /              – List own transactions (admin sees all)
    GET  /summary       – Aggregate totals by status
    GET  /export.csv    – Download transactions as CSV
"""

import csv
import io
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user, require_roles
from datetime import datetime, date, timedelta

router = APIRouter()


# ── Create ───────────────────────────────────────────────────────────────
@router.post("/", response_model=schemas.TransactionOut)
def create_transaction(
    data: schemas.TransactionCreate,
    db: Session = Depends(get_db),
    _admin=Depends(require_roles("ADMIN")),
):
    if data.amount is None or data.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")

    txn = models.Transaction(
        user_id=data.user_id,
        amount=data.amount,
        status=(data.status or "pending").lower(),
        category=(data.category or "consultation").lower(),
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)
    return txn


# ── List ─────────────────────────────────────────────────────────────────
@router.get("/", response_model=list[schemas.TransactionOut])
def list_transactions(
    status: str | None = None,
    category: str | None = None,
    start: str | None = None,
    end: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    query = db.query(models.Transaction)

    # Non-admins only see their own
    role = (user.role or "").upper()
    if role != "ADMIN":
        query = query.filter(models.Transaction.user_id == user.user_id)

    if status:
        query = query.filter(models.Transaction.status == status.lower())
    if category:
        query = query.filter(models.Transaction.category == category.lower())

    # Date filters (ISO format YYYY-MM-DD)
    if start:
        try:
            d = date.fromisoformat(start)
            query = query.filter(
                models.Transaction.created_at >= datetime.combine(d, datetime.min.time())
            )
        except ValueError:
            pass
    if end:
        try:
            d = date.fromisoformat(end)
            query = query.filter(
                models.Transaction.created_at < datetime.combine(d, datetime.min.time()) + timedelta(days=1)
            )
        except ValueError:
            pass

    return query.order_by(models.Transaction.created_at.desc()).limit(200).all()


# ── Summary ──────────────────────────────────────────────────────────────
@router.get("/summary")
def summary(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    query = db.query(models.Transaction)
    role = (user.role or "").upper()
    if role != "ADMIN":
        query = query.filter(models.Transaction.user_id == user.user_id)

    rows = (
        query.with_entities(
            models.Transaction.status,
            func.coalesce(func.sum(models.Transaction.amount), 0.0),
        )
        .group_by(models.Transaction.status)
        .all()
    )

    by_status = {k: float(v) for k, v in rows}
    return {
        "pending": by_status.get("pending", 0.0),
        "completed": by_status.get("completed", 0.0),
        "failed": by_status.get("failed", 0.0),
        "count": query.count(),
    }


# ── CSV Export ────────────────────────────────────────────────────────────
@router.get("/export.csv")
def export_csv(
    status: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    query = db.query(models.Transaction)
    role = (user.role or "").upper()
    if role != "ADMIN":
        query = query.filter(models.Transaction.user_id == user.user_id)
    if status:
        query = query.filter(models.Transaction.status == status.lower())

    rows = query.order_by(models.Transaction.created_at.desc()).limit(1000).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Amount", "Status", "Category", "Method", "Reference", "Note", "Created"])
    for r in rows:
        writer.writerow([
            r.transaction_id,
            r.amount,
            r.status,
            getattr(r, "category", ""),
            getattr(r, "method", ""),
            getattr(r, "reference", ""),
            getattr(r, "note", ""),
            r.created_at.isoformat() if r.created_at else "",
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transactions.csv"},
    )
