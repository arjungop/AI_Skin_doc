from typing import Dict, Set
from fastapi import WebSocket


class NotificationHub:
    by_user: Dict[int, Set[WebSocket]] = {}

    @classmethod
    def register(cls, user_id: int, ws: WebSocket):
        cls.by_user.setdefault(user_id, set()).add(ws)

    @classmethod
    def unregister(cls, user_id: int, ws: WebSocket):
        try:
            cls.by_user.get(user_id, set()).discard(ws)
            if not cls.by_user.get(user_id):
                cls.by_user.pop(user_id, None)
        except Exception:
            pass

    @classmethod
    def send(cls, user_id: int, event: str, payload: dict):
        data = {"event": event, **payload}
        for ws in list(cls.by_user.get(user_id, set())):
            try:
                ws.send_json(data)
            except Exception:
                try:
                    ws.close()
                except Exception:
                    pass
                cls.unregister(user_id, ws)

    @classmethod
    def send_many(cls, user_ids: Set[int] | list[int], event: str, payload: dict):
        for uid in set(user_ids):
            cls.send(uid, event, payload)

