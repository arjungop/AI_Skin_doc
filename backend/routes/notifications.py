from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models
from backend.notify import NotificationHub

router = APIRouter()


@router.websocket("/notifications/ws")
async def notifications_ws(websocket: WebSocket, token: str = Query("")):
    from backend.security import _decode_token  # type: ignore
    from backend.database import SessionLocal
    try:
        payload = _decode_token(token)
        user_id = int(payload.get("sub", 0))
    except Exception:
        await websocket.close(code=4401)
        return

    # Use a short-lived DB session for auth only, then close it
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.user_id == user_id).first()
    finally:
        db.close()

    if not user:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    NotificationHub.register(user.user_id, websocket)
    try:
        # Keep connection alive; ignore incoming data
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        NotificationHub.unregister(user.user_id, websocket)

