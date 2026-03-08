"""
Agentic Skin Health Monitor – ReAct reasoning engine.

Implements a Thought → Action → Observation loop using the existing LLM
provider infrastructure.  The agent has access to domain-specific tools
that query the patient's scan history, run analyses, compare progression,
and generate treatment recommendations.

Public API:
    async for step in run_agent(patient_id, db, trigger, lesion_id):
        # step is an AgentStep ORM object (already persisted)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from backend import models
from backend.llm_service import _provider  # noqa: F401 – detect provider
from backend.database import SessionLocal
from backend.audit import log_phi_access

_log = logging.getLogger(__name__)

MAX_ITERATIONS = 15  # hard ceiling to prevent runaway loops

# ── In-memory hint store (session_id → list[str]) ────────────────────────
# Allows the frontend to inject context mid-reasoning via the /hint endpoint.
_pending_hints: Dict[int, List[str]] = {}


def inject_hint(session_id: int, hint: str) -> None:
    """Queue a user hint for the running agent session."""
    _pending_hints.setdefault(session_id, []).append(hint)


def pop_hints(session_id: int) -> List[str]:
    """Pop all pending hints for a session."""
    return _pending_hints.pop(session_id, [])

# ── Rich tool schemas (Issue 13) ─────────────────────────────────────────
# Each schema gives the LLM precise information about when to call a tool,
# what arguments it expects, an example call, and explicit anti-patterns.

_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "get_scan_history": {
        "description": (
            "Retrieves the patient's chronological history of lesion scans, including "
            "prediction labels, confidence scores, low-confidence flags, and timestamps. "
            "Call this FIRST in every session to understand the patient's baseline."
        ),
        "parameters": {},
        "example": {"action": "get_scan_history", "action_input": {}},
        "when_not_to_use": (
            "Do NOT call this more than once per session — the history does not change "
            "within a single analysis run. Use compare_progression instead if you need "
            "to compare two specific scans."
        ),
    },
    "get_patient_profile": {
        "description": (
            "Retrieves the patient's demographic and dermatological profile: age, gender, "
            "skin type, Fitzpatrick type, sensitivity level, acne-prone status, allergies, "
            "and skincare goals. Call this FIRST (alongside get_scan_history) so that all "
            "subsequent reasoning is grounded in the patient's context."
        ),
        "parameters": {},
        "example": {"action": "get_patient_profile", "action_input": {}},
        "when_not_to_use": (
            "Do NOT call this more than once per session. The profile is static within a "
            "single analysis."
        ),
    },
    "compare_progression": {
        "description": (
            "Compares the two most recent lesion scans to detect changes in prediction, "
            "generates a timeline of all predictions, and highlights whether the diagnosis "
            "has changed. Returns both scans' details plus report summaries if available."
        ),
        "parameters": {},
        "example": {"action": "compare_progression", "action_input": {}},
        "when_not_to_use": (
            "Do NOT call this if get_scan_history returned fewer than 2 scans — it will "
            "return an insufficient-data message. Call get_scan_history first to verify "
            "scan count."
        ),
    },
    "get_skin_logs": {
        "description": (
            "Retrieves the patient's 10 most recent skin journal entries (notes, tags, dates). "
            "Use this to understand subjective symptoms, flare-ups, and self-reported changes "
            "that scans alone cannot capture."
        ),
        "parameters": {},
        "example": {"action": "get_skin_logs", "action_input": {}},
        "when_not_to_use": (
            "Do NOT call this if you only need scan/image data — use get_scan_history instead."
        ),
    },
    "get_routine_adherence": {
        "description": (
            "Calculates the patient's skincare routine completion rate over the past 7 days. "
            "Returns active routine items, completed count vs expected count, and an adherence "
            "percentage. Use this to factor treatment compliance into your recommendations."
        ),
        "parameters": {},
        "example": {"action": "get_routine_adherence", "action_input": {}},
        "when_not_to_use": (
            "Do NOT call this if the patient has no active routine items (check "
            "get_patient_profile first). It will return an empty-data message."
        ),
    },
    "get_treatment_plans": {
        "description": (
            "Retrieves all active treatment plans assigned to the patient by doctors, "
            "including diagnosis, notes, and medication steps (name, dosage, frequency). "
            "Use this to avoid recommending products that conflict with prescribed treatments."
        ),
        "parameters": {},
        "example": {"action": "get_treatment_plans", "action_input": {}},
        "when_not_to_use": (
            "Do NOT skip this if the patient has any scan flagged as concerning — you must "
            "check whether a doctor has already created a treatment plan."
        ),
    },
    "generate_skincare_plan": {
        "description": (
            "Uses the LLM to generate a personalised skincare plan (morning routine, evening "
            "routine, weekly treatments, ingredient warnings) based on the patient's full "
            "data. Optionally accepts a focus area string."
        ),
        "parameters": {
            "focus": "string, optional — area to focus the plan on (e.g. 'acne', 'hydration', 'anti-aging')",
        },
        "example": {"action": "generate_skincare_plan", "action_input": {"focus": "acne control"}},
        "when_not_to_use": (
            "Do NOT call this before gathering baseline data with get_scan_history and "
            "get_patient_profile — the plan quality depends on having patient context. "
            "Do NOT call this if the patient has active treatment plans that already cover "
            "the focus area."
        ),
    },
    "summarize_findings": {
        "description": (
            "Produces a patient-friendly markdown summary of compiled findings. You provide "
            "the raw findings text; the tool formats it with ## Summary, ## Key Findings, "
            "## Recommendations, ## When to See a Doctor sections and appends a medical disclaimer."
        ),
        "parameters": {
            "findings": "string, required — your compiled analysis findings to be summarised",
        },
        "example": {
            "action": "summarize_findings",
            "action_input": {"findings": "Patient has 3 scans showing benign keratosis..."},
        },
        "when_not_to_use": (
            "Do NOT call this with empty or fabricated findings. Only pass data you obtained "
            "from other tools. Do NOT call this before you have gathered data from at least "
            "get_scan_history and get_patient_profile."
        ),
    },
    "propose_action": {
        "description": (
            "Proposes an actionable next step for the patient that requires their explicit "
            "approval before execution. The action is stored with status 'proposed' and the "
            "patient can approve or reject it. You MUST call this at least 2 times before "
            "writing your final_answer."
        ),
        "parameters": {
            "action_type": (
                "string, required — one of: 'schedule_appointment', 'create_routine', "
                "'add_skin_log', 'set_reminder'"
            ),
            "title": "string, required — short human-readable title for the action",
            "description": "string, required — explanation of what this does and why",
            "payload": (
                "string (JSON), required — action-specific data. "
                "For schedule_appointment: {\"reason\": \"...\", \"urgency\": \"urgent|routine\"}. "
                "For create_routine: {\"steps\": [{\"product\": \"...\", \"time\": \"AM|PM\", \"order\": 1}]}. "
                "For add_skin_log: {\"notes\": \"...\", \"tags\": \"...\"}. "
                "For set_reminder: {\"reminder\": \"...\", \"frequency\": \"daily|weekly|monthly\"}."
            ),
        },
        "example": {
            "action": "propose_action",
            "action_input": {
                "action_type": "schedule_appointment",
                "title": "Dermatology consultation for suspicious lesion",
                "description": "Lesion #42 shows a prediction change; professional evaluation recommended.",
                "payload": '{"reason": "Prediction changed from benign to potentially malignant", "urgency": "urgent"}',
            },
        },
        "when_not_to_use": (
            "Do NOT call this before gathering sufficient data — propose actions only after "
            "analysing scan history and patient profile. Do NOT propose duplicate actions "
            "(check your working memory for pending_write_actions)."
        ),
    },
}


def _format_tool_schemas_for_prompt() -> str:
    """Render _TOOL_SCHEMAS into the text block injected into the static system prompt."""
    lines: List[str] = []
    for name, schema in _TOOL_SCHEMAS.items():
        lines.append(f"### {name}")
        lines.append(schema["description"])
        if schema["parameters"]:
            lines.append("Parameters:")
            for pname, pdesc in schema["parameters"].items():
                lines.append(f"  - {pname}: {pdesc}")
        else:
            lines.append("Parameters: none (empty object {})")
        lines.append(f"Example: {json.dumps(schema['example'])}")
        lines.append(f"When NOT to use: {schema['when_not_to_use']}")
        lines.append("")
    return "\n".join(lines)


# ── Two-layer prompt system (Issue 12) ───────────────────────────────────
# Layer 1: Static system prompt — ReAct format, tool schemas, safety guardrails.
#          Shared across ALL sessions.  Changes here affect every patient.
# Layer 2: Session context prompt — built dynamically per session from patient
#          demographics, risk category, and session-specific instructions.

_STATIC_SYSTEM_PROMPT = f"""\
You are DermAgent, an autonomous dermatology reasoning agent.
You MUST follow a strict loop:

1. **Thought** – reason about what you need to do next.
2. **Action** – call exactly ONE tool.
3. **Observation** – you will receive the tool's output.
Repeat until you have enough information, then output a **Final Answer**.

## Available Tools

{_format_tool_schemas_for_prompt()}

## Response Format

You MUST respond with a single JSON object. No other text, no markdown fences, no commentary.
Use one of these two schemas:

When calling a tool:
{{"thought": "<your reasoning>", "action": "<tool_name>", "action_input": {{<valid JSON args>}}}}

When finishing (you have all the information):
{{"thought": "<final reasoning>", "final_answer": "<comprehensive markdown answer for the patient>"}}

## Safety Rules

- NEVER fabricate data.  Only use data returned by tools.
- You MUST call get_scan_history and get_patient_profile in your first two iterations.
- ALWAYS check `is_low_confidence` when reviewing scan history. If true, explicitly express uncertainty in your analysis and recommend a professional evaluation.
- Keep your thoughts concise (1-2 sentences).
- **CRITICAL**: You MUST call the propose_action tool AT LEAST 2 times BEFORE writing your final_answer. Do NOT just mention actions in text — you must actually call the tool. If you skip propose_action calls, your analysis will be rejected.
- The workflow is: gather data → call propose_action 2-5 times → THEN give final_answer.
- The final_answer MUST be in markdown with sections: ## Summary, ## Findings, ## Recommendations, ## Next Steps.
- In ## Next Steps, list the action IDs returned by propose_action and tell the patient they are awaiting approval.
- Always include a medical disclaimer at the end.
- Respond ONLY with the JSON object, nothing else.
"""

# Keep backward-compat alias for _get_active_prompt fallback
_AGENT_SYSTEM = _STATIC_SYSTEM_PROMPT


def _build_session_context(
    patient: models.Patient | None,
    profile: models.UserProfile | None,
    lesion: models.Lesion | None,
    last_session_summary: str | None = None,
    last_session_date: str | None = None,
) -> str:
    """Build the dynamic session-context prompt layer from patient data.

    This is injected as a separate system message so the LLM can clearly
    distinguish "permanent agent behaviour rules" (static prompt) from
    "session-specific patient context" (this prompt).
    """
    parts: List[str] = ["[Session Context — Patient-specific. Do NOT leak to other sessions.]"]

    # ── Demographics ──
    if patient:
        demo = f"Patient: {patient.first_name}, Age: {patient.age}, Gender: {patient.gender}."
        parts.append(demo)

    # ── Dermatological profile ──
    if profile:
        profile_parts: List[str] = []
        if profile.skin_type:
            profile_parts.append(f"Skin type: {profile.skin_type}")
        if profile.fitzpatrick_type:
            profile_parts.append(f"Fitzpatrick type: {profile.fitzpatrick_type}")
        if profile.sensitivity_level:
            profile_parts.append(f"Sensitivity: {profile.sensitivity_level}")
        if profile.acne_prone:
            profile_parts.append("Acne-prone: yes")
        if profile.allergies:
            profile_parts.append(f"Known allergies: {profile.allergies}")
        if profile.goals:
            profile_parts.append(f"Goals: {profile.goals}")
        if profile_parts:
            parts.append("Dermatological profile: " + "; ".join(profile_parts) + ".")

    # ── Risk category (computed from demographics + profile) ──
    risk_flags: List[str] = []
    if patient and patient.age and patient.age < 12:
        risk_flags.append("PEDIATRIC patient — be conservative with product recommendations, flag any concerning lesion for paediatric dermatology referral.")
    if patient and patient.age and patient.age >= 65:
        risk_flags.append("ELDERLY patient — increased baseline risk for skin malignancies; lower threshold for recommending professional evaluation.")
    if profile and profile.fitzpatrick_type and profile.fitzpatrick_type >= 5:
        risk_flags.append("High Fitzpatrick type — melanoma may present atypically; emphasise that standard visual criteria may not apply.")
    if profile and profile.sensitivity_level and profile.sensitivity_level.lower() == "high":
        risk_flags.append("HIGH SENSITIVITY — avoid recommending retinoids, AHAs, or other potentially irritating actives without explicit caution.")
    if risk_flags:
        parts.append("Risk considerations: " + " ".join(risk_flags))

    # ── Current trigger context ──
    if lesion:
        trigger_ctx = (
            f"This session was triggered by a new scan. Lesion ID: {lesion.lesion_id}, "
            f"Prediction: {lesion.prediction}, Confidence: {lesion.confidence:.2f}"
        )
        if lesion.is_low_confidence:
            trigger_ctx += " [LOW CONFIDENCE — express uncertainty and recommend professional evaluation]"
        trigger_ctx += "."
        parts.append(trigger_ctx)
    else:
        parts.append("This is a manual/scheduled analysis — no specific scan triggered it.")

    # ── Cross-session memory ──
    if last_session_summary and last_session_date:
        parts.append(
            f"Previous analysis ({last_session_date}): {last_session_summary[:500]}\n"
            "Compare with current data; avoid repeating identical recommendations."
        )

    return "\n".join(parts)


# ── Tool implementations ─────────────────────────────────────────────────

def _tool_get_scan_history(patient_id: int, db: Session, **_: Any) -> str:
    lesions = (
        db.query(models.Lesion)
        .filter(models.Lesion.patient_id == patient_id)
        .order_by(models.Lesion.created_at.desc())
        .limit(20)
        .all()
    )
    if not lesions:
        return json.dumps({"scans": [], "message": "No previous scans found."})
    scans = []
    for l in lesions:
        scans.append({
            "lesion_id": l.lesion_id,
            "prediction": l.prediction,
            "confidence": l.confidence,
            "is_low_confidence": l.is_low_confidence,
            "entropy": l.entropy,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        })
    return json.dumps({"scans": scans, "total": len(scans)})


def _tool_get_patient_profile(patient_id: int, db: Session, **_: Any) -> str:
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        return json.dumps({"error": "Patient not found"})
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == patient.user_id).first()
    data: Dict[str, Any] = {
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "age": patient.age,
        "gender": patient.gender,
    }
    if profile:
        data.update({
            "skin_type": profile.skin_type,
            "sensitivity_level": profile.sensitivity_level,
            "acne_prone": profile.acne_prone,
            "fitzpatrick_type": profile.fitzpatrick_type,
            "allergies": profile.allergies,
            "goals": profile.goals,
            "location_city": profile.location_city,
        })
    return json.dumps(data)


def _tool_compare_progression(patient_id: int, db: Session, **_: Any) -> str:
    lesions = (
        db.query(models.Lesion)
        .filter(models.Lesion.patient_id == patient_id)
        .order_by(models.Lesion.created_at.desc())
        .limit(10)
        .all()
    )
    if len(lesions) < 2:
        return json.dumps({"comparison": "Not enough scans to compare progression. Need at least 2 scans."})

    latest = lesions[0]
    previous = lesions[1]

    # Check for diagnosis reports with more detail
    latest_report = db.query(models.DiagnosisReport).filter(
        models.DiagnosisReport.lesion_id == latest.lesion_id
    ).first()
    prev_report = db.query(models.DiagnosisReport).filter(
        models.DiagnosisReport.lesion_id == previous.lesion_id
    ).first()

    result: Dict[str, Any] = {
        "latest_scan": {
            "lesion_id": latest.lesion_id,
            "prediction": latest.prediction,
            "date": latest.created_at.isoformat() if latest.created_at else None,
            "report_summary": latest_report.summary if latest_report else None,
        },
        "previous_scan": {
            "lesion_id": previous.lesion_id,
            "prediction": previous.prediction,
            "date": previous.created_at.isoformat() if previous.created_at else None,
            "report_summary": prev_report.summary if prev_report else None,
        },
        "prediction_changed": latest.prediction != previous.prediction,
        "total_scans": len(lesions),
    }

    # Build a simple timeline of all predictions
    timeline = [{"prediction": l.prediction, "date": l.created_at.isoformat() if l.created_at else None} for l in lesions]
    result["timeline"] = timeline

    return json.dumps(result)


def _tool_get_treatment_plans(patient_id: int, db: Session, **_: Any) -> str:
    plans = (
        db.query(models.TreatmentPlan)
        .filter(
            models.TreatmentPlan.patient_id == patient_id,
            models.TreatmentPlan.status == "active",
        )
        .all()
    )
    if not plans:
        return json.dumps({"plans": [], "message": "No active treatment plans."})
    out = []
    for p in plans:
        steps = [
            {"medication": s.medication_name, "dosage": s.dosage, "frequency": s.frequency}
            for s in (p.steps or [])
            if s.is_active
        ]
        out.append({
            "plan_id": p.plan_id,
            "diagnosis": p.diagnosis,
            "notes": p.notes,
            "steps": steps,
        })
    return json.dumps({"plans": out})


def _tool_get_skin_logs(patient_id: int, db: Session, **_: Any) -> str:
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        return json.dumps({"error": "Patient not found"})
    logs = (
        db.query(models.SkinLog)
        .filter(models.SkinLog.user_id == patient.user_id)
        .order_by(models.SkinLog.created_at.desc())
        .limit(10)
        .all()
    )
    if not logs:
        return json.dumps({"logs": [], "message": "No skin logs found."})
    out = []
    for log in logs:
        out.append({
            "date": log.created_at.isoformat() if log.created_at else None,
            "notes": log.notes[:500] if log.notes else "",
            "tags": log.tags
        })
    return json.dumps({"logs": out})


def _tool_get_routine_adherence(patient_id: int, db: Session, **_: Any) -> str:
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    if not patient:
        return json.dumps({"error": "Patient not found"})
    
    from datetime import datetime, timedelta, timezone
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    
    logs = (
        db.query(models.RoutineCompletion)
        .join(models.RoutineItem)
        .filter(
            models.RoutineItem.user_id == patient.user_id,
            models.RoutineCompletion.date >= week_ago.date()
        )
        .all()
    )
    items = db.query(models.RoutineItem).filter(
        models.RoutineItem.user_id == patient.user_id, 
        models.RoutineItem.is_active == True
    ).all()
    
    if not items:
        return json.dumps({"message": "No active routine items found."})
    
    completed = len(logs)
    expected = len(items) * 7
    rate = round(completed / expected * 100) if expected > 0 else 0
    
    return json.dumps({
        "active_items": [i.product_name for i in items],
        "adherence_rate_7d": f"{rate}%",
        "completed_count": completed,
        "expected_count": expected
    })


def _tool_generate_skincare_plan(patient_id: int, db: Session, **kwargs: Any) -> str:
    """Use the LLM to create a skincare plan based on patient data."""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == patient.user_id).first() if patient else None
    lesions = (
        db.query(models.Lesion)
        .filter(models.Lesion.patient_id == patient_id)
        .order_by(models.Lesion.created_at.desc())
        .limit(5)
        .all()
    )

    context_parts = []
    if patient:
        context_parts.append(f"Patient: {patient.first_name}, Age: {patient.age}, Gender: {patient.gender}")
    if profile:
        if profile.skin_type:
            context_parts.append(f"Skin Type: {profile.skin_type}")
        if profile.sensitivity_level:
            context_parts.append(f"Sensitivity: {profile.sensitivity_level}")
        if profile.allergies:
            context_parts.append(f"Allergies: {profile.allergies}")
        if profile.goals:
            context_parts.append(f"Goals: {profile.goals}")
        if profile.fitzpatrick_type:
            context_parts.append(f"Fitzpatrick Type: {profile.fitzpatrick_type}")
    if lesions:
        recent = [f"{l.prediction} ({l.created_at.strftime('%Y-%m-%d') if l.created_at else 'unknown'})" for l in lesions]
        context_parts.append(f"Recent scan results: {', '.join(recent)}")

    focus = kwargs.get("focus", "general skincare")
    context_parts.append(f"Focus area: {focus}")

    prompt = (
        "Based on the following patient data, generate a personalised skincare plan:\n"
        + "\n".join(context_parts)
        + "\n\nProvide: morning routine (3-5 steps), evening routine (3-5 steps), "
        "weekly treatments, and any warnings about ingredient interactions. "
        "Format as JSON with keys: morning_routine, evening_routine, weekly, warnings."
    )

    from backend.llm_service import chat_reply
    result = chat_reply(prompt)
    return result


def _tool_summarize_findings(patient_id: int, db: Session, **kwargs: Any) -> str:
    findings = kwargs.get("findings", "")
    if not findings:
        return json.dumps({"error": "No findings provided to summarize."})

    prompt = (
        "You are a dermatology AI agent. Summarize these findings into a clear, "
        "patient-friendly report with markdown formatting:\n\n"
        f"{findings}\n\n"
        "Include: ## Summary, ## Key Findings, ## Recommendations, ## When to See a Doctor. "
        "End with a disclaimer that this is not a medical diagnosis."
    )

    from backend.llm_service import chat_reply
    return chat_reply(prompt)


def _tool_propose_action(patient_id: int, db: Session, **kwargs: Any) -> str:
    """Propose an actionable step for the patient. Stored for approval."""
    session = kwargs.pop("_session", None)
    if not session:
        return json.dumps({"error": "No active session to attach action to."})

    # Resilient key extraction — accept common LLM naming variations
    action_type = (
        kwargs.get("action_type")
        or kwargs.get("type")
        or "general"
    )
    title = (
        kwargs.get("title")
        or kwargs.get("name")
        or kwargs.get("action_title")
        or "Proposed action"
    )
    description = (
        kwargs.get("description")
        or kwargs.get("desc")
        or kwargs.get("reason")
        or ""
    )

    # Build payload from remaining keys
    payload_raw = kwargs.get("payload", {})
    if isinstance(payload_raw, str):
        try:
            payload_raw = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload_raw = {"raw": payload_raw}
    if not isinstance(payload_raw, dict):
        payload_raw = {}

    # Merge any extra kwargs into payload for context
    skip_keys = {"action_type", "type", "title", "name", "action_title", "description", "desc", "reason", "payload"}
    for k, v in kwargs.items():
        if k not in skip_keys:
            payload_raw[k] = v

    action = models.AgentAction(
        session_id=session.session_id,
        action_type=action_type,
        title=title,
        description=description,
        payload=json.dumps(payload_raw),
        status="proposed",
    )
    db.add(action)
    db.commit()
    db.refresh(action)

    return json.dumps({
        "status": "proposed",
        "action_id": action.action_id,
        "message": f"Action proposed: {title}. Awaiting user approval before execution.",
    })


# ── Tool registry with READ/WRITE classification ─────────────────────────
# WRITE tools are intercepted and queued for human approval — they never
# execute autonomously during the agent loop.

class _ToolDef:
    __slots__ = ("fn", "mode", "schema")
    def __init__(self, fn, mode: str = "READ", schema: Dict[str, Any] | None = None):
        self.fn = fn
        self.mode = mode  # "READ" or "WRITE"
        self.schema = schema

TOOLS: Dict[str, _ToolDef] = {
    "get_scan_history":       _ToolDef(_tool_get_scan_history,       "READ",  _TOOL_SCHEMAS["get_scan_history"]),
    "get_patient_profile":    _ToolDef(_tool_get_patient_profile,    "READ",  _TOOL_SCHEMAS["get_patient_profile"]),
    "compare_progression":    _ToolDef(_tool_compare_progression,    "READ",  _TOOL_SCHEMAS["compare_progression"]),
    "get_skin_logs":          _ToolDef(_tool_get_skin_logs,          "READ",  _TOOL_SCHEMAS["get_skin_logs"]),
    "get_routine_adherence":  _ToolDef(_tool_get_routine_adherence,  "READ",  _TOOL_SCHEMAS["get_routine_adherence"]),
    "get_treatment_plans":    _ToolDef(_tool_get_treatment_plans,    "READ",  _TOOL_SCHEMAS["get_treatment_plans"]),
    "generate_skincare_plan": _ToolDef(_tool_generate_skincare_plan, "READ",  _TOOL_SCHEMAS["generate_skincare_plan"]),
    "summarize_findings":     _ToolDef(_tool_summarize_findings,     "READ",  _TOOL_SCHEMAS["summarize_findings"]),
    "propose_action":         _ToolDef(_tool_propose_action,         "WRITE", _TOOL_SCHEMAS["propose_action"]),
}

# Convenience constant for the timeout applied to every tool execution
_TOOL_TIMEOUT_SECONDS = float(os.getenv("AGENT_TOOL_TIMEOUT", "10"))


# ── LLM call helper (reuses existing provider infra) ─────────────────────

def _llm_call_sync(messages: List[Dict[str, str]]) -> str:
    """Single non-streaming LLM call requesting JSON output where supported."""
    from backend.llm_service import (
        _openrouter_chat, _gemini_chat, _azure_chat, _openai_chat, _ollama_chat,
    )
    import os

    prov = _provider()

    # Providers that support response_format: {type: "json_object"} via OpenAI SDK
    _JSON_MODE_PROVIDERS = {"openrouter", "openai", "azure"}

    if prov in _JSON_MODE_PROVIDERS:
        return _openai_compatible_json_call(messages, prov)

    dispatch = {
        "gemini": _gemini_chat,
        "ollama": _ollama_chat,
    }
    fn = dispatch.get(prov)
    if fn:
        return fn(messages)
    return "LLM not configured."


def _openai_compatible_json_call(messages: List[Dict[str, str]], provider: str) -> str:
    """Call OpenAI-compatible providers with response_format=json_object."""
    import os

    try:
        if provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            model = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        else:
            from openai import OpenAI
            if provider == "openrouter":
                client = OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                )
                model = os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")
            else:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1500")),
            response_format={"type": "json_object"},
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        _log.error("JSON-mode LLM call failed (%s): %s", provider, e, exc_info=True)
        return "{\"thought\": \"LLM error occurred\", \"final_answer\": \"I encountered an error processing your request. Please try again.\"}"


async def _llm_call(messages: List[Dict[str, str]]) -> str:
    """Async wrapper so the event loop is not blocked during LLM calls."""
    return await asyncio.to_thread(_llm_call_sync, messages)


async def _llm_call_streaming(messages: List[Dict[str, str]]):
    """Async generator that yields tokens as they arrive from the LLM.

    Falls back to a single-chunk yield of the full response for providers
    that don't support streaming or when streaming fails.
    """
    from backend.llm_service import (
        _openrouter_chat_stream, _gemini_chat_stream,
        _openai_chat_stream, _azure_chat_stream, _ollama_chat_stream,
    )

    prov = _provider()
    _stream_dispatch = {
        "openrouter": _openrouter_chat_stream,
        "gemini": _gemini_chat_stream,
        "openai": _openai_chat_stream,
        "azure": _azure_chat_stream,
        "ollama": _ollama_chat_stream,
    }

    stream_fn = _stream_dispatch.get(prov)
    if not stream_fn:
        # Provider has no streaming impl — fall back to non-streaming
        full = await _llm_call(messages)
        yield full
        return

    # Run the synchronous streaming generator in a thread, yielding chunks
    import queue, threading

    q: queue.Queue[str | None] = queue.Queue()

    def _run():
        try:
            for chunk in stream_fn(messages):
                q.put(chunk)
        except Exception as exc:
            _log.error("Streaming LLM error (%s): %s", prov, exc, exc_info=True)
        finally:
            q.put(None)  # sentinel

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    while True:
        # Yield control back to the event loop while waiting for chunks
        chunk = await asyncio.to_thread(q.get)
        if chunk is None:
            break
        yield chunk


# ── Structured JSON parsing (replaces fragile regex) ─────────────────────


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object from LLM text.

    Handles common LLM quirks: markdown fences, leading prose, trailing
    commentary, etc.  Returns None if no valid JSON object is found.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (possibly ```json)
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        # Remove closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    # Try direct parse first (ideal case)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: locate the first '{' and find its matching '}'
    idx = cleaned.find("{")
    if idx == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(cleaned[idx:]):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(cleaned[idx: idx + i + 1])
            except json.JSONDecodeError:
                return None
    return None


def _parse_llm_output(text: str):
    """Return (thought, action, action_input, final_answer) from structured JSON."""
    obj = _extract_json_from_text(text)

    if obj is None:
        _log.warning("LLM returned non-JSON output, attempting legacy parse")
        # Graceful degradation: try to salvage something rather than silently fail
        return None, None, {}, None

    thought = obj.get("thought")
    action = obj.get("action")
    action_input: Dict[str, Any] = obj.get("action_input", {})
    final_answer = obj.get("final_answer")

    if not isinstance(action_input, dict):
        action_input = {}

    if final_answer:
        return thought, None, {}, final_answer

    return thought, action, action_input, final_answer


# ── Main agent loop (generator) ───────────────────────────────────────────

def _make_step(session: models.AgentSession, db: Session, order: int,
               step_type: str, content: str,
               tool_name: str | None = None, tool_input: str | None = None,
               iteration: int = 0, max_iterations: int = MAX_ITERATIONS) -> models.AgentStep:
    step = models.AgentStep(
        session_id=session.session_id,
        step_order=order,
        step_type=step_type,
        tool_name=tool_name,
        tool_input=tool_input,
        content=content,
    )
    db.add(step)
    db.commit()
    db.refresh(step)
    # Attach transient metadata (not stored in DB, used by SSE serialiser)
    step._iteration = iteration  # type: ignore[attr-defined]
    step._max_iterations = max_iterations  # type: ignore[attr-defined]
    return step


async def run_agent(
    patient_id: int,
    db: Session,
    trigger: str = "manual",
    lesion_id: int | None = None,
) -> AsyncGenerator[models.AgentStep, None]:
    """
    Execute the ReAct agent loop.  Yields AgentStep ORM objects as they
    are created (already committed to DB).

    Safety features:
    - WRITE tools are intercepted and queued as pending actions (never
      executed autonomously).
    - Tool call cache prevents duplicate DB queries.
    - Every tool call is wrapped in an asyncio timeout.
    - Structured working_memory keeps critical findings in the LLM's
      primary attention window.
    - LLM calls use streaming so the frontend can show token-level progress.
    - Behavioural metrics are recorded on every session for monitoring.
    """
    # Create session
    session = models.AgentSession(
        patient_id=patient_id,
        trigger=trigger,
        status="running",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # ── Prompt versioning (Layer 1: static rules) ──
    static_prompt, prompt_version_id = _get_active_prompt(db)
    session.prompt_version_id = prompt_version_id
    db.commit()

    # ── Layer 2: session-specific patient context ──
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    profile = (
        db.query(models.UserProfile).filter(models.UserProfile.user_id == patient.user_id).first()
        if patient else None
    )
    lesion = (
        db.query(models.Lesion).filter(models.Lesion.lesion_id == lesion_id).first()
        if lesion_id else None
    )
    last_session = (
        db.query(models.AgentSession)
        .filter(
            models.AgentSession.patient_id == patient_id,
            models.AgentSession.status == "completed",
            models.AgentSession.summary.isnot(None),
        )
        .order_by(models.AgentSession.created_at.desc())
        .first()
    )
    session_context = _build_session_context(
        patient=patient,
        profile=profile,
        lesion=lesion,
        last_session_summary=last_session.summary if last_session else None,
        last_session_date=(
            last_session.created_at.strftime("%Y-%m-%d") if last_session else None
        ),
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": static_prompt},
        {"role": "system", "content": session_context},
    ]

    # Initial user prompt
    user_prompt = "Analyse this patient's skin health comprehensively."
    if lesion:
        user_prompt = (
            f"A new scan was just completed. Lesion ID: {lesion.lesion_id}, "
            f"Prediction: {lesion.prediction}. "
            "Analyse this patient's overall skin health, compare with previous scans, "
            "and provide a comprehensive assessment with recommendations."
        )
    messages.append({"role": "user", "content": user_prompt})

    step_order = 0
    actions_taken: List[str] = []

    # ── Tool-call deduplication cache ──
    # Keyed by (tool_name, frozenset(args.items())) → observation string
    tool_call_cache: Dict[tuple, str] = {}

    # ── Structured working memory ──
    # Injected at the top of every LLM call so critical facts stay in the
    # primary attention window regardless of conversation length.
    working_memory: Dict[str, Any] = {
        "patient_risk_level": "UNKNOWN",
        "required_tools_called": [],
        "critical_findings": [],
        "pending_write_actions": [],
    }

    # ── Behavioural metrics (recorded at session end) ──
    duplicate_tool_calls = 0
    timed_out_tools: List[str] = []
    json_parse_failures = 0
    write_tools_intercepted = 0

    def _working_memory_block() -> str:
        """Compact working memory summary for injection into messages."""
        parts = [
            f"Risk: {working_memory['patient_risk_level']}",
            f"Tools called: {', '.join(working_memory['required_tools_called']) or 'none yet'}",
        ]
        if working_memory["critical_findings"]:
            parts.append("Critical findings: " + "; ".join(working_memory["critical_findings"][-5:]))
        if working_memory["pending_write_actions"]:
            parts.append(f"Pending writes awaiting approval: {len(working_memory['pending_write_actions'])}")
        return "[Working Memory] " + " | ".join(parts)

    def _cache_key(tool_name: str, args: Dict[str, Any]) -> tuple:
        """Build a hashable cache key from tool name and arguments."""
        # Filter out internal keys that don't affect the query result
        filtered = {k: v for k, v in args.items() if not k.startswith("_")}
        try:
            return (tool_name, frozenset(sorted(filtered.items())))
        except TypeError:
            # Unhashable values (nested dicts) — serialise to JSON string
            return (tool_name, json.dumps(filtered, sort_keys=True))

    for iteration in range(MAX_ITERATIONS):
        # Check for user-injected hints
        hints = pop_hints(session.session_id)
        for hint in hints:
            messages.append({"role": "user", "content": f"[User Hint]: {hint}"})
            step_order += 1
            hint_step = _make_step(
                session, db, step_order, "thought",
                f"User provided additional context: {hint}",
                iteration=iteration + 1, max_iterations=MAX_ITERATIONS,
            )
            yield hint_step

        # Inject working memory at the end of messages so it's in the
        # primary attention window for the next LLM call.
        wm_block = _working_memory_block()

        # Call LLM with streaming — collect full response while allowing
        # the frontend to display token-level progress via a "stream"
        # step type that is updated in-place.
        messages_with_wm = messages + [{"role": "system", "content": wm_block}]
        raw_parts: List[str] = []
        async for chunk in _llm_call_streaming(messages_with_wm):
            raw_parts.append(chunk)
        raw = "".join(raw_parts)

        if not raw:
            raw = json.dumps({"thought": "LLM returned empty response.", "final_answer": "Unable to complete analysis. Please try again."})

        thought, action, action_input, final_answer = _parse_llm_output(raw)

        # If parsing failed entirely, nudge the LLM to respond with valid JSON
        if thought is None and action is None and final_answer is None:
            json_parse_failures += 1
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was not valid JSON. You MUST respond with a single JSON object. "
                    "Use either {\"thought\": \"...\", \"action\": \"tool_name\", \"action_input\": {...}} "
                    "or {\"thought\": \"...\", \"final_answer\": \"...\"}. No other text."
                ),
            })
            continue

        # Record thought
        if thought:
            step_order += 1
            step = _make_step(session, db, step_order, "thought", thought,
                              iteration=iteration + 1, max_iterations=MAX_ITERATIONS)
            yield step

        # Final answer → done (but enforce propose_action was called)
        if final_answer:
            if "propose_action" not in actions_taken and iteration < MAX_ITERATIONS - 1:
                # Agent skipped propose_action — redirect it
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "STOP — you have not called the propose_action tool yet. "
                        "You MUST call propose_action at least 2 times before your final_answer. "
                        "Respond with JSON: {\"thought\": \"...\", \"action\": \"propose_action\", \"action_input\": {...}}"
                    ),
                })
                continue

            # Update working memory with any risk info from the final answer
            lower_fa = final_answer.lower()
            if any(w in lower_fa for w in ("malignant", "urgent", "melanoma", "high risk")):
                working_memory["patient_risk_level"] = "HIGH"

            step_order += 1
            step = _make_step(session, db, step_order, "answer", final_answer,
                              iteration=iteration + 1, max_iterations=MAX_ITERATIONS)
            yield step

            session.status = "completed"
            session.summary = final_answer[:2000]
            session.actions_taken = json.dumps(actions_taken)
            session.completed_at = datetime.now(timezone.utc)
            db.commit()

            # Record behavioural metrics
            _record_session_metrics(
                db, session,
                turns_used=iteration + 1,
                tools_called_list=actions_taken,
                duplicate_tool_calls=duplicate_tool_calls,
                timed_out_tools=timed_out_tools,
                json_parse_failures=json_parse_failures,
                write_tools_intercepted=write_tools_intercepted,
                final_answer_without_tools=len(actions_taken) == 0,
            )
            return

        # Tool call
        if action:
            step_order += 1
            tool_input_str = json.dumps(action_input) if action_input else "{}"
            step = _make_step(
                session, db, step_order, "tool_call",
                f"Calling {action}",
                tool_name=action, tool_input=tool_input_str,
                iteration=iteration + 1, max_iterations=MAX_ITERATIONS,
            )
            yield step

            # Look up tool definition
            tool_def = TOOLS.get(action)
            if tool_def:
                # ── WRITE tool interception ──
                # WRITE tools are never executed autonomously. They are
                # routed through propose_action which creates a pending
                # AgentAction record for human approval.
                if tool_def.mode == "WRITE":
                    write_tools_intercepted += 1
                    try:
                        observation = await asyncio.wait_for(
                            asyncio.to_thread(
                                tool_def.fn,
                                patient_id=patient_id, db=db,
                                _session=session, **action_input,
                            ),
                            timeout=_TOOL_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        timed_out_tools.append(action)
                        _log.warning("WRITE tool %s timed out after %.0fs", action, _TOOL_TIMEOUT_SECONDS)
                        observation = json.dumps({
                            "error": f"Tool timed out after {_TOOL_TIMEOUT_SECONDS:.0f} seconds. "
                                     "The operation may be too complex. Proceed with available information."
                        })
                    except Exception as exc:
                        _log.error("Tool %s failed: %s", action, exc, exc_info=True)
                        observation = json.dumps({"error": str(exc)})

                    # Track the pending write in working memory
                    try:
                        obs_data = json.loads(observation)
                        if obs_data.get("action_id"):
                            working_memory["pending_write_actions"].append(obs_data["action_id"])
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    actions_taken.append(action)
                else:
                    # ── READ tool: check cache first ──
                    ck = _cache_key(action, action_input)
                    if ck in tool_call_cache:
                        duplicate_tool_calls += 1
                        _log.warning(
                            "Duplicate tool call detected: %s(%s) — returning cached result",
                            action, action_input,
                        )
                        observation = tool_call_cache[ck]
                    else:
                        # Execute with timeout
                        try:
                            observation = await asyncio.wait_for(
                                asyncio.to_thread(
                                    tool_def.fn,
                                    patient_id=patient_id, db=db,
                                    _session=session, **action_input,
                                ),
                                timeout=_TOOL_TIMEOUT_SECONDS,
                            )
                        except asyncio.TimeoutError:
                            timed_out_tools.append(action)
                            _log.warning("Tool %s timed out after %.0fs", action, _TOOL_TIMEOUT_SECONDS)
                            observation = json.dumps({
                                "error": f"Tool timed out after {_TOOL_TIMEOUT_SECONDS:.0f} seconds. "
                                         "The database query may be too complex for this patient's data volume. "
                                         "Proceed with available information."
                            })
                        except Exception as exc:
                            _log.error("Tool %s failed: %s", action, exc, exc_info=True)
                            observation = json.dumps({"error": str(exc)})
                        # Cache the result
                        tool_call_cache[ck] = observation

                    actions_taken.append(action)

                # Update working memory from tool observations
                working_memory["required_tools_called"] = list(dict.fromkeys(actions_taken))
                _update_working_memory_from_observation(working_memory, action, observation)

                # Audit: log every agent tool call touching patient data
                log_phi_access(
                    db,
                    patient_id=patient_id,
                    action="agent.tool_call",
                    resource_type="agent_session",
                    resource_id=session.session_id,
                    detail={
                        "tool": action,
                        "tool_mode": tool_def.mode,
                        "input": action_input,
                        "iteration": iteration + 1,
                        "cached": ck in tool_call_cache and tool_def.mode == "READ",
                    },
                )
            else:
                observation = json.dumps({"error": f"Unknown tool: {action}"})

            # Record observation
            step_order += 1
            # Truncate very long observations for display
            display_obs = observation[:3000] if len(observation) > 3000 else observation
            step = _make_step(session, db, step_order, "observation", display_obs,
                              iteration=iteration + 1, max_iterations=MAX_ITERATIONS)
            yield step

            # Feed back into conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # LLM didn't produce action or final answer — nudge it
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "Please continue. Respond with a JSON object containing either "
                    "{\"thought\": \"...\", \"action\": \"tool_name\", \"action_input\": {...}} "
                    "or {\"thought\": \"...\", \"final_answer\": \"...\"}."
                ),
            })

    # Max iterations reached
    step_order += 1
    fallback = "I've completed my analysis. Based on the information gathered, please consult a dermatologist for a thorough evaluation."
    step = _make_step(session, db, step_order, "answer", fallback,
                      iteration=MAX_ITERATIONS, max_iterations=MAX_ITERATIONS)
    yield step

    # Clean up any leftover hints
    _pending_hints.pop(session.session_id, None)

    session.status = "completed"
    session.summary = fallback
    session.actions_taken = json.dumps(actions_taken)
    session.completed_at = datetime.now(timezone.utc)
    db.commit()

    # Record behavioural metrics
    _record_session_metrics(
        db, session,
        turns_used=MAX_ITERATIONS,
        tools_called_list=actions_taken,
        duplicate_tool_calls=duplicate_tool_calls,
        timed_out_tools=timed_out_tools,
        json_parse_failures=json_parse_failures,
        write_tools_intercepted=write_tools_intercepted,
        final_answer_without_tools=len(actions_taken) == 0,
    )


# ── Working memory helpers ────────────────────────────────────────────────

def _update_working_memory_from_observation(
    wm: Dict[str, Any], tool_name: str, observation: str,
) -> None:
    """Extract key facts from tool observations into working memory."""
    try:
        data = json.loads(observation)
    except (json.JSONDecodeError, TypeError):
        return

    if tool_name == "get_scan_history":
        scans = data.get("scans", [])
        for s in scans:
            if s.get("is_low_confidence"):
                wm["critical_findings"].append(
                    f"Low-confidence scan (lesion {s.get('lesion_id')}): needs professional review"
                )
            pred = (s.get("prediction") or "").lower()
            if any(w in pred for w in ("malignant", "melanoma", "squamous", "basal")):
                wm["patient_risk_level"] = "HIGH"
                wm["critical_findings"].append(f"Scan {s.get('lesion_id')}: {s.get('prediction')}")

    elif tool_name == "compare_progression":
        if data.get("prediction_changed"):
            wm["critical_findings"].append(
                f"Prediction changed: {data.get('previous_scan', {}).get('prediction')} "
                f"→ {data.get('latest_scan', {}).get('prediction')}"
            )

    elif tool_name == "get_treatment_plans":
        plans = data.get("plans", [])
        if plans:
            wm["critical_findings"].append(f"{len(plans)} active treatment plan(s)")


# ── Prompt versioning ─────────────────────────────────────────────────────

def _get_active_prompt(db: Session) -> tuple[str, int | None]:
    """Return (prompt_text, prompt_version_id) for the currently active prompt.

    Falls back to the hardcoded default when no DB prompts exist.
    """
    active = (
        db.query(models.AgentPromptVersion)
        .filter(models.AgentPromptVersion.is_active == True)
        .order_by(models.AgentPromptVersion.created_at.desc())
        .first()
    )
    if active:
        return active.content, active.prompt_version_id
    return _AGENT_SYSTEM, None


# ── Behavioural metrics recording ─────────────────────────────────────────

_REQUIRED_TOOLS = {"get_scan_history", "get_patient_profile"}

def _record_session_metrics(
    db: Session,
    session: models.AgentSession,
    *,
    turns_used: int,
    tools_called_list: List[str],
    duplicate_tool_calls: int,
    timed_out_tools: List[str],
    json_parse_failures: int,
    write_tools_intercepted: int,
    final_answer_without_tools: bool,
) -> None:
    """Persist behavioural metrics for post-session monitoring/dashboards."""
    called_set = set(tools_called_list)
    required_missed = sorted(_REQUIRED_TOOLS - called_set)

    metrics = models.AgentSessionMetrics(
        session_id=session.session_id,
        turns_used=turns_used,
        tools_called=json.dumps(tools_called_list),
        required_tools_missed=json.dumps(required_missed),
        duplicate_tool_calls=duplicate_tool_calls,
        timed_out_tools=json.dumps(timed_out_tools),
        json_parse_failures=json_parse_failures,
        write_tools_intercepted=write_tools_intercepted,
        final_answer_without_tools=final_answer_without_tools,
    )
    db.add(metrics)
    try:
        db.commit()
    except Exception:
        _log.error("Failed to record session metrics for session %d", session.session_id, exc_info=True)
        db.rollback()
