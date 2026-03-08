"""
Agent routes – SSE-streamed analysis + session history + action approval/execution.

Endpoints:
    POST /agent/analyze                – kick off an agent run, stream steps via SSE
    GET  /agent/sessions               – list past agent sessions for the patient
    GET  /agent/sessions/{id}          – full session with steps + actions
    GET  /agent/sessions/{id}/actions  – list proposed actions for a session
    POST /agent/actions/{id}/approve   – approve a proposed action
    POST /agent/actions/{id}/reject    – reject a proposed action
    POST /agent/sessions/{id}/execute  – execute all approved actions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend import models, schemas
from backend.security import get_current_user, verify_patient_access
from backend.agent_engine import run_agent, inject_hint
from backend.agent_actions import execute_action
from backend.audit import log_phi_access

_log = logging.getLogger(__name__)

router = APIRouter()


# ── SSE stream helper ─────────────────────────────────────────────────────

def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


# ── POST /agent/analyze ──────────────────────────────────────────────────

@router.post("/analyze")
async def analyze(
    req: schemas.AgentRunRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Run the agentic skin-health analysis.  Streams steps as SSE."""
    # Row-level access control: patients own data, doctors must be linked
    verify_patient_access(req.patient_id, user, db)

    patient = db.query(models.Patient).filter(
        models.Patient.patient_id == req.patient_id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    trigger = "scan" if req.lesion_id else "manual"

    async def event_stream():
        try:
            async for step in run_agent(
                patient_id=req.patient_id,
                db=db,
                trigger=trigger,
                lesion_id=req.lesion_id,
            ):
                yield _sse_event("step", {
                    "step_id": step.step_id,
                    "step_order": step.step_order,
                    "step_type": step.step_type,
                    "tool_name": step.tool_name,
                    "content": step.content,
                    "created_at": step.created_at.isoformat() if step.created_at else None,
                    "iteration": getattr(step, '_iteration', 0),
                    "max_iterations": getattr(step, '_max_iterations', 15),
                })
            yield _sse_event("done", {"status": "completed"})
        except Exception as exc:
            _log.error("Agent stream error: %s", exc, exc_info=True)
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── GET /agent/sessions ──────────────────────────────────────────────────

@router.get("/sessions", response_model=list[schemas.AgentSessionOut])
def list_sessions(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """List agent sessions for the logged-in patient."""
    patient = db.query(models.Patient).filter(
        models.Patient.user_id == user.user_id
    ).first()
    if not patient:
        return []
    sessions = (
        db.query(models.AgentSession)
        .filter(models.AgentSession.patient_id == patient.patient_id)
        .order_by(models.AgentSession.created_at.desc())
        .limit(20)
        .all()
    )
    return sessions


# ── GET /agent/sessions/{session_id} ─────────────────────────────────────

@router.get("/sessions/{session_id}", response_model=schemas.AgentSessionOut)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Get a single agent session with all steps."""
    session = db.query(models.AgentSession).filter(
        models.AgentSession.session_id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    verify_patient_access(session.patient_id, user, db)

    return session


# ── Helper: verify session ownership ──────────────────────────────────────

def _verify_session_ownership(session_id: int, db: Session, user: models.User):
    session = db.query(models.AgentSession).filter(
        models.AgentSession.session_id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    verify_patient_access(session.patient_id, user, db)
    return session


# ── GET /agent/sessions/{session_id}/actions ──────────────────────────────

@router.get("/sessions/{session_id}/actions", response_model=list[schemas.AgentActionOut])
def list_session_actions(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """List all proposed actions for a session."""
    session = _verify_session_ownership(session_id, db, user)
    return session.actions


# ── POST /agent/actions/{action_id}/approve ───────────────────────────────

@router.post("/actions/{action_id}/approve", response_model=schemas.AgentActionOut)
def approve_action(
    action_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Approve a proposed action for later execution."""
    action = db.query(models.AgentAction).filter(
        models.AgentAction.action_id == action_id
    ).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    _verify_session_ownership(action.session_id, db, user)

    if action.status not in ("proposed", "rejected"):
        raise HTTPException(status_code=400, detail=f"Cannot approve action in '{action.status}' state")

    action.status = "approved"
    db.commit()
    db.refresh(action)

    # Audit: log approval
    session_obj = db.query(models.AgentSession).filter(
        models.AgentSession.session_id == action.session_id
    ).first()
    log_phi_access(
        db,
        user_id=user.user_id,
        patient_id=session_obj.patient_id if session_obj else None,
        action="agent_action.approve",
        resource_type="agent_action",
        resource_id=action.action_id,
        detail={"action_type": action.action_type, "title": action.title},
    )
    return action


@router.post("/actions/{action_id}/reject", response_model=schemas.AgentActionOut)
def reject_action(
    action_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Reject a proposed action."""
    action = db.query(models.AgentAction).filter(
        models.AgentAction.action_id == action_id
    ).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    _verify_session_ownership(action.session_id, db, user)

    if action.status not in ("proposed", "approved"):
        raise HTTPException(status_code=400, detail=f"Cannot reject action in '{action.status}' state")

    action.status = "rejected"
    db.commit()
    db.refresh(action)

    # Audit: log rejection
    session_obj = db.query(models.AgentSession).filter(
        models.AgentSession.session_id == action.session_id
    ).first()
    log_phi_access(
        db,
        user_id=user.user_id,
        patient_id=session_obj.patient_id if session_obj else None,
        action="agent_action.reject",
        resource_type="agent_action",
        resource_id=action.action_id,
        detail={"action_type": action.action_type, "title": action.title},
    )
    return action


# ── POST /agent/sessions/{session_id}/execute ─────────────────────────────

@router.post("/sessions/{session_id}/execute", response_model=list[schemas.AgentActionOut])
def execute_session_actions(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Execute all approved actions for a session."""
    session = _verify_session_ownership(session_id, db, user)

    approved = [a for a in session.actions if a.status == "approved"]
    if not approved:
        # Return current actions list so frontend always gets a consistent response
        return session.actions

    for action in approved:
        try:
            result = execute_action(action, db)
            action.status = "executed"
            action.result = result
            action.executed_at = datetime.now(timezone.utc)
        except Exception as exc:
            _log.error("Action %d execution failed: %s", action.action_id, exc, exc_info=True)
            action.status = "failed"
            action.result = str(exc)
        db.commit()

        # Audit: log each action execution
        log_phi_access(
            db,
            user_id=user.user_id,
            patient_id=session.patient_id,
            action="agent_action.execute",
            resource_type="agent_action",
            resource_id=action.action_id,
            detail={
                "action_type": action.action_type,
                "title": action.title,
                "status": action.status,
            },
        )

    db.refresh(session)
    return session.actions


# ── POST /agent/sessions/{session_id}/hint ────────────────────────────────

class HintRequest(BaseModel):
    hint: str

@router.post("/sessions/{session_id}/hint")
def send_hint(
    session_id: int,
    req: HintRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """Inject a user hint into a running agent session."""
    session = _verify_session_ownership(session_id, db, user)
    if session.status != "running":
        raise HTTPException(status_code=400, detail="Session is not currently running")
    inject_hint(session_id, req.hint)
    return {"ok": True, "message": "Hint queued for next agent iteration"}
