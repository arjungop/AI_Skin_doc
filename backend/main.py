from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import os
import pathlib

# Load env early so modules reading os.getenv during import see correct values
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from backend import models, crud
from backend.database import engine, SessionLocal, run_simple_migrations
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from backend.routes.patients import router as patients_router
from backend.routes.appointments import router as appointments_router
from backend.routes.lesions import router as lesions_router
from backend.routes.transactions import router as transactions_router
from backend.routes.llm import router as llm_router
from backend.auth import router as auth_router
from backend.routes.doctors import router as doctors_router
from backend.routes.admin import router as admin_router
from backend.routes.chat import router as chat_router
from backend.routes.ai_chat import router as ai_chat_router
from backend.routes.support import router as support_router
from backend.routes.notifications import router as notifications_router
from backend.routes.profile import router as profile_router
from backend.routes.journey import router as journey_router
from backend.routes.routine import router as routine_router
from backend.routes.recommendations import router as recommendations_router
from backend.security import require_roles

# Create tables and run lightweight migrations for dev
models.Base.metadata.create_all(bind=engine)
try:
    run_simple_migrations()
except Exception:
    pass

app = FastAPI()

# Rate limiting
from backend.auth import limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

try:
    # Ensure both '/path' and '/path/' are accepted with 307 redirect
    app.router.redirect_slashes = True  # type: ignore
except Exception:
    pass

# CORS
frontend_origins = os.getenv("FRONTEND_ORIGINS")
if frontend_origins:
    origins = [o.strip() for o in frontend_origins.split(",") if o.strip()]
else:
    frontend_origin = os.getenv("FRONTEND_ORIGIN")
    if frontend_origin:
        origins = [frontend_origin]
    else:
        # Sensible defaults for local dev (React/Vite)
        origins = [
            "http://127.0.0.1:3000", "http://localhost:3000",
            "http://127.0.0.1:3001", "http://localhost:3001",
            "http://127.0.0.1:5173", "http://localhost:5173",
            "http://127.0.0.1:5174", "http://localhost:5174",
            "https://localhost:5173", "https://127.0.0.1:5173",
        ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)


# Security headers middleware (#14)
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(self)"
        return response

app.add_middleware(SecurityHeadersMiddleware)


# Upload size limit middleware (#10) — 15 MB max
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large. Maximum is {MAX_UPLOAD_BYTES // (1024*1024)}MB."},
            )
        return await call_next(request)

app.add_middleware(MaxBodySizeMiddleware)

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# DB status — admin only, no internal details leaked
@app.get("/db/status")
def db_status(_=Depends(require_roles("ADMIN"))):
    from backend.database import engine as _engine
    ok = True
    try:
        with _engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception:
        ok = False
    return {"ok": ok}

# Bootstrap admin (only when env vars are explicitly set)
try:
    _admin_email = os.getenv("ADMIN_EMAIL", "").strip()
    _admin_username = os.getenv("ADMIN_USERNAME", "").strip()
    _admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    if _admin_email and _admin_username and _admin_password:
        db = SessionLocal()
        any_admin = db.query(models.User).filter(models.User.role == "ADMIN").first()
        if not any_admin:
            user = db.query(models.User).filter(models.User.email == _admin_email).first()
            if not user:
                user = models.User(
                    username=_admin_username,
                    email=_admin_email,
                    hashed_password=crud.hash_password(_admin_password),
                    role="ADMIN",
                )
                db.add(user)
                db.commit()
            else:
                user.username = _admin_username or user.username
                user.hashed_password = crud.hash_password(_admin_password)
                user.role = "ADMIN"
                db.commit()
        db.close()
except Exception:
    pass

# Routers
app.include_router(patients_router, prefix="/patients", tags=["patients"])
app.include_router(appointments_router, prefix="/appointments", tags=["appointments"])
app.include_router(lesions_router, prefix="/lesions", tags=["lesions"])
app.include_router(transactions_router, prefix="/transactions", tags=["transactions"])
app.include_router(llm_router, prefix="/llm", tags=["llm"])
app.include_router(auth_router)
app.include_router(doctors_router, prefix="/doctors", tags=["doctors"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(ai_chat_router, prefix="/ai_chat", tags=["ai_chat"])
app.include_router(support_router, tags=["support"])
app.include_router(notifications_router, tags=["notifications"])
app.include_router(recommendations_router, prefix="/recommendations", tags=["recommendations"])

app.include_router(profile_router, prefix="/profile", tags=["profile"])
app.include_router(journey_router, prefix="/journey", tags=["journey"])
app.include_router(routine_router, prefix="/routine", tags=["routine"])

# Root redirect to frontend
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    """Redirect root to frontend"""
    return RedirectResponse(url="http://localhost:3000")

# Serve built frontend if present (single-port mode)
try:
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DIST_DIR = BASE_DIR / "frontend-react" / "dist"
    if DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
except Exception:
    pass
