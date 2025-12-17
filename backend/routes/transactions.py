from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user, require_roles
from datetime import datetime, date, timedelta
from sqlalchemy import String

router = APIRouter()

@router.post("/", response_model=schemas.TransactionOut)
def create_transaction(
    data: schemas.TransactionCreate,
    db: Session = Depends(get_db),
    _admin=Depends(require_roles("ADMIN")),
):
    # Basic validation
    if data.amount is None or data.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")

    new_txn = models.Transaction(
        user_id=data.user_id,
        amount=data.amount,
        status=(data.status or "pending").lower(),
        category=(data.category or "general").lower(),
    )
    db.add(new_txn)
    db.commit()
    db.refresh(new_txn)
    return new_txn

@router.get("/", response_model=list[schemas.TransactionDetailOut])
def get_transactions(
    status: str | None = None,
    search: str | None = None,
    start: str | None = None,
    end: str | None = None,
    category: str | None = None,
    transaction_id: int | None = None,
    user_id: int | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    from sqlalchemy.orm import aliased
    Meta = models.TransactionMeta
    query = db.query(models.Transaction, Meta).outerjoin(Meta, models.Transaction.transaction_id == Meta.transaction_id)
    role = (user.role or "").upper()
    if role != "ADMIN":
        query = query.filter(models.Transaction.user_id == user.user_id)
        # Non-admins cannot filter by another user
        if user_id is not None and user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Forbidden: cannot filter other users' transactions")
    else:
        if user_id is not None:
            query = query.filter(models.Transaction.user_id == user_id)
    if transaction_id is not None:
        query = query.filter(models.Transaction.transaction_id == transaction_id)
    if status:
        query = query.filter(models.Transaction.status == status.lower())
    if category:
        query = query.filter(models.Transaction.category == category.lower())
    if start:
        try:
            # Accept full ISO datetime or date-only (YYYY-MM-DD or DD/MM/YYYY)
            if 'T' in start or ' ' in start:
                start_dt = datetime.fromisoformat(start)
            else:
                try:
                    # Try ISO date first
                    d = date.fromisoformat(start)
                except Exception:
                    # Fallback to dd/mm/yyyy
                    d = datetime.strptime(start, '%d/%m/%Y').date()
                start_dt = datetime.combine(d, datetime.min.time())
            query = query.filter(models.Transaction.created_at >= start_dt)
        except Exception:
            pass
    if end:
        try:
            # Accept full ISO datetime or date-only; for date-only, include the whole day
            if 'T' in end or ' ' in end:
                end_dt = datetime.fromisoformat(end)
                query = query.filter(models.Transaction.created_at <= end_dt)
            else:
                try:
                    d = date.fromisoformat(end)
                except Exception:
                    d = datetime.strptime(end, '%d/%m/%Y').date()
                next_day = datetime.combine(d, datetime.min.time()) + timedelta(days=1)
                query = query.filter(models.Transaction.created_at < next_day)
        except Exception:
            pass
    # text search on reference/note/method
    if search:
        like = f"%{search}%"
        query = query.filter(
            (Meta.reference.ilike(like)) | (Meta.note.ilike(like)) | (Meta.method.ilike(like))
        )
    rows = query.order_by(models.Transaction.created_at.desc()).all()
    out = []
    for txn, meta in rows:
        d = {
            "transaction_id": txn.transaction_id,
            "user_id": txn.user_id,
            "amount": txn.amount,
            "status": txn.status,
            "created_at": txn.created_at,
            "method": getattr(meta, 'method', None) if meta else None,
            "reference": getattr(meta, 'reference', None) if meta else None,
            "note": getattr(meta, 'note', None) if meta else None,
        }
        out.append(d)
    return out


@router.get("/summary", response_model=schemas.TransactionsSummaryOut)
def summary(
    user_id: int | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    role = (user.role or "").upper()
    q = db.query(models.Transaction)
    if role != "ADMIN":
        q = q.filter(models.Transaction.user_id == user.user_id)
    elif user_id:
        q = q.filter(models.Transaction.user_id == user_id)

    # Aggregate sums per status
    sums = (
        db.query(models.Transaction.status, func.coalesce(func.sum(models.Transaction.amount), 0.0))
        .filter(models.Transaction.transaction_id.in_([t.transaction_id for t in q.all()]))
        .group_by(models.Transaction.status)
        .all()
    )
    by_status = {k: float(v or 0.0) for k, v in sums}
    count = q.count()
    return schemas.TransactionsSummaryOut(
        pending=by_status.get("pending", 0.0),
        completed=by_status.get("completed", 0.0),
        failed=by_status.get("failed", 0.0),
        refunded=by_status.get("refunded", 0.0),
        count=count,
    )


@router.get("/monthly")
def monthly(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
    months: int = 12,
):
    # Group by Year-Month using SQLite-friendly strftime
    role = (user.role or "").upper()
    base = db.query(models.Transaction)
    if role != "ADMIN":
        base = base.filter(models.Transaction.user_id == user.user_id)
    month_expr = func.strftime('%Y-%m', models.Transaction.created_at)
    rows = (
        db.query(month_expr.label('ym'), func.coalesce(func.sum(models.Transaction.amount), 0.0).label('total'))
        .filter(models.Transaction.status.in_(['completed','refunded']))
        .filter(models.Transaction.created_at >= datetime.utcnow() - timedelta(days=months*31))
        .group_by('ym')
        .order_by('ym')
        .all()
    )
    return [{"month": ym, "total": float(total)} for ym, total in rows]


@router.get("/{transaction_id}/receipt")
def receipt(transaction_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    txn = db.query(models.Transaction).filter(models.Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Not found")
    role = (user.role or "").upper()
    if role != "ADMIN" and txn.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    meta = db.query(models.TransactionMeta).filter(models.TransactionMeta.transaction_id == transaction_id).first()
    return {
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "amount": txn.amount,
        "status": txn.status,
        "category": txn.category,
        "created_at": txn.created_at,
        "method": meta.method if meta else None,
        "reference": meta.reference if meta else None,
        "note": meta.note if meta else None,
    }


@router.get("/{transaction_id}/receipt.pdf")
def receipt_pdf(transaction_id: int, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        raise HTTPException(status_code=500, detail="PDF export requires 'reportlab' in requirements.txt")
    from fastapi.responses import StreamingResponse
    import io
    txn = db.query(models.Transaction).filter(models.Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Not found")
    role = (user.role or "").upper()
    if role != "ADMIN" and txn.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    meta = db.query(models.TransactionMeta).filter(models.TransactionMeta.transaction_id == transaction_id).first()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4
    y = h-40
    c.setFont("Helvetica-Bold", 16); c.drawString(40,y, "Payment Receipt")
    y-=24; c.setFont("Helvetica", 11)
    pairs = [
      ("Receipt #", str(txn.transaction_id)),
      ("Amount", f"₹ {txn.amount:.2f}"),
      ("Status", txn.status),
      ("Category", txn.category or 'general'),
      ("Created", str(txn.created_at)),
      ("Method", (meta.method if meta else '') or '-'),
      ("Reference", (meta.reference if meta else '') or '-'),
      ("Note", (meta.note if meta else '') or '-'),
    ]
    for k,v in pairs:
        c.drawString(40,y, f"{k}: {v}"); y-=16
    c.showPage(); c.save(); buf.seek(0)
    return StreamingResponse(buf, media_type='application/pdf', headers={"Content-Disposition": f"attachment; filename=receipt-{txn.transaction_id}.pdf"})


@router.patch("/{transaction_id}/status", response_model=schemas.TransactionOut)
def update_status(
    transaction_id: int,
    body: schemas.TransactionStatusUpdate,
    db: Session = Depends(get_db),
    _admin=Depends(require_roles("ADMIN")),
):
    txn = db.query(models.Transaction).filter(models.Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if body.status.lower() not in {"pending", "completed", "failed", "refunded"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    txn.status = body.status.lower()
    db.add(txn)
    db.commit()
    db.refresh(txn)
    # audit
    try:
        from backend.models import AuditLog
        reason = (body.reason or '').strip() if hasattr(body, 'reason') else ''
        meta = f"{transaction_id}:{txn.status}" + (f" reason={reason}" if reason else '')
        db.add(AuditLog(user_id=_admin.user_id, action="UPDATE_TRANSACTION_STATUS", meta=meta))  # type: ignore
        db.commit()
    except Exception:
        db.rollback()
    return txn


@router.post("/{transaction_id}/meta", response_model=schemas.TransactionDetailOut)
def set_meta(
    transaction_id: int,
    body: schemas.TransactionMetaIn,
    db: Session = Depends(get_db),
    _admin=Depends(require_roles("ADMIN")),
):
    txn = db.query(models.Transaction).filter(models.Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    # Admin-only endpoint for updating transaction meta

    # Upsert meta
    meta = db.query(models.TransactionMeta).filter(models.TransactionMeta.transaction_id == transaction_id).first()
    if not meta:
        meta = models.TransactionMeta(transaction_id=transaction_id)
        db.add(meta)
    if body.method is not None:
        meta.method = body.method
    if body.reference is not None:
        meta.reference = body.reference
    if body.note is not None:
        meta.note = body.note
    db.commit()
    db.refresh(meta)
    try:
        from backend.models import AuditLog
        db.add(AuditLog(user_id=_admin.user_id, action="SET_TRANSACTION_META", meta=f"{transaction_id}"))  # type: ignore
        db.commit()
    except Exception:
        db.rollback()
    return {
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "amount": txn.amount,
        "status": txn.status,
        "created_at": txn.created_at,
        "method": meta.method,
        "reference": meta.reference,
        "note": meta.note,
    }


@router.post("/{transaction_id}/refund", response_model=schemas.TransactionOut)
def refund(
    transaction_id: int,
    db: Session = Depends(get_db),
    _admin=Depends(require_roles("ADMIN")),
):
    txn = db.query(models.Transaction).filter(models.Transaction.transaction_id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    txn.status = "refunded"
    db.add(txn)
    # mark meta
    meta = db.query(models.TransactionMeta).filter(models.TransactionMeta.transaction_id == transaction_id).first()
    if not meta:
        meta = models.TransactionMeta(transaction_id=transaction_id)
        db.add(meta)
    meta.refund_of = transaction_id
    db.commit()
    db.refresh(txn)
    try:
        from backend.models import AuditLog
        db.add(AuditLog(user_id=_admin.user_id, action="REFUND_TRANSACTION", meta=f"{transaction_id}"))  # type: ignore
        db.commit()
    except Exception:
        db.rollback()
    return txn


@router.get("/admin_list")
def admin_list(
    status: str | None = None,
    q: str | None = None,
    method: str | None = None,
    category: str | None = None,
    start: str | None = None,
    end: str | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _=Depends(require_roles("ADMIN")),
):
    Meta = models.TransactionMeta
    U = models.User
    query = (
        db.query(models.Transaction, Meta, U)
        .outerjoin(Meta, models.Transaction.transaction_id == Meta.transaction_id)
        .join(U, U.user_id == models.Transaction.user_id)
    )
    if status:
        query = query.filter(models.Transaction.status == status.lower())
    if category:
        query = query.filter(models.Transaction.category == category.lower())
    if method:
        query = query.filter(Meta.method == method)
    if amount_min is not None:
        query = query.filter(models.Transaction.amount >= amount_min)
    if amount_max is not None:
        query = query.filter(models.Transaction.amount <= amount_max)
    if start:
        try:
            start_dt = datetime.fromisoformat(start)
            query = query.filter(models.Transaction.created_at >= start_dt)
        except Exception:
            pass
    if end:
        try:
            end_dt = datetime.fromisoformat(end)
            query = query.filter(models.Transaction.created_at <= end_dt)
        except Exception:
            pass
    if q:
        like = f"%{q}%"
        query = query.filter(
            (Meta.reference.ilike(like)) | (Meta.note.ilike(like)) | (Meta.method.ilike(like)) |
            (U.email.ilike(like)) | (U.username.ilike(like))
        )
    total = query.count()
    rows = query.order_by(models.Transaction.created_at.desc()).offset(max(0,(page-1)*page_size)).limit(page_size).all()
    items = []
    for t, m, u in rows:
        items.append({
            "transaction_id": t.transaction_id,
            "user_id": t.user_id,
            "username": u.username,
            "email": u.email,
            "amount": t.amount,
            "status": t.status,
            "category": t.category,
            "created_at": t.created_at,
            "method": getattr(m,'method',None) if m else None,
            "reference": getattr(m,'reference',None) if m else None,
            "note": getattr(m,'note',None) if m else None,
        })
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/export.csv")
def export_csv(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    # Export current user's data (or all for admin)
    import csv, io
    from fastapi.responses import StreamingResponse
    rows = get_transactions(
        status=None,
        search=None,
        start=None,
        end=None,
        category=None,
        transaction_id=None,
        user_id=None,
        db=db,
        user=user,
    )  # reuse logic
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["transaction_id","amount","status","created_at","method","reference","note"]) 
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k) if isinstance(r, dict) else getattr(r, k, None) for k in writer.fieldnames})
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=transactions.csv"})


@router.get("/export.pdf")
def export_pdf(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        raise HTTPException(status_code=500, detail="PDF export requires 'reportlab' in requirements.txt")
    from fastapi.responses import StreamingResponse
    import io
    rows = get_transactions(
        status=None,
        search=None,
        start=None,
        end=None,
        category=None,
        transaction_id=None,
        user_id=None,
        db=db,
        user=user,
    )
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Transactions Report")
    y -= 20
    c.setFont("Helvetica", 10)
    headers = ["ID","Amount","Status","Created","Method","Reference"]
    c.drawString(40, y, "  |  ".join(headers))
    y -= 16
    for r in rows[:500]:
        line = f"{r['transaction_id']}  |  {r['amount']}  |  {r['status']}  |  {r['created_at']}  |  {r.get('method','')}  |  {r.get('reference','')}"
        c.drawString(40, y, line[:115])
        y -= 14
        if y < 40:
            c.showPage(); y = height - 40
    c.showPage(); c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=transactions.pdf"})
