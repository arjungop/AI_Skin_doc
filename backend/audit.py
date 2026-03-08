"""
HIPAA-grade PHI audit logging.

Every access to Protected Health Information — agent tool calls, action
approvals, lesion uploads, diagnosis views — is recorded in the
``phi_audit_logs`` table via the helpers below.

Usage:
    from backend.audit import log_phi_access
    log_phi_access(db, user_id=1, patient_id=5, action="agent.tool_call",
                   resource_type="lesion", resource_id=42,
                   detail={"tool": "get_scan_history"}, ip="1.2.3.4")
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend import models

_log = logging.getLogger(__name__)


def log_phi_access(
    db: Session,
    *,
    user_id: Optional[int] = None,
    patient_id: Optional[int] = None,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[int] = None,
    detail: Any = None,
    ip_address: Optional[str] = None,
) -> None:
    """Write an immutable audit record for PHI access."""
    detail_str = None
    if detail is not None:
        try:
            detail_str = json.dumps(detail, default=str)
        except Exception:
            detail_str = str(detail)

    entry = models.PHIAuditLog(
        user_id=user_id,
        patient_id=patient_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        detail=detail_str,
        ip_address=ip_address,
    )
    try:
        db.add(entry)
        db.commit()
    except Exception:
        db.rollback()
        _log.error("Failed to write PHI audit log: %s", action, exc_info=True)
