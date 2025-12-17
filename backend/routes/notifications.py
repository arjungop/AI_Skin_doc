from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.orm import Session
from backend.database import get_db
from backend import models
from backend.notify import NotificationHub

router = APIRouter()


@router.websocket("/notifications/ws")
async def notifications_ws(websocket: WebSocket, token: str = Query(""), db: Session = Depends(get_db)):
    from backend.security import _decode_token  # type: ignore
    try:
        payload = _decode_token(token)
        user_id = int(payload.get("sub", 0))
    except Exception:
        await websocket.close(code=4401)
        return
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
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

