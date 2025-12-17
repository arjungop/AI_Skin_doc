from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Create tables and run lightweight migrations for dev
models.Base.metadata.create_all(bind=engine)
try:
    run_simple_migrations()
except Exception:
    pass

app = FastAPI()
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
    allow_methods=["*"],
    # Allow all request headers (including Authorization and multipart boundaries)
    allow_headers=["*"],
)

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# DB status (shows sanitized URL and last error if any)
@app.get("/db/status")
def db_status():
    from backend.database import engine
    url_sanitized = None
    try:
        url_sanitized = engine.url.render_as_string(hide_password=True)
    except Exception:
        url_sanitized = None
    last_error = None
    ok = True
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception as e:
        ok = False
        last_error = str(e)
    return {
        "ok": ok,
        "dialect": getattr(engine, 'dialect', None) and engine.dialect.name,
        "url": url_sanitized,
        "error": last_error,
    }

# Bootstrap admin (dev default if none exists)
try:
    db = SessionLocal()
    any_admin = db.query(models.User).filter(models.User.role == "ADMIN").first()
    if not any_admin:
        email = os.getenv("ADMIN_EMAIL", "admin@example.com")
        username = os.getenv("ADMIN_USERNAME", "admin")
        password = os.getenv("ADMIN_PASSWORD", "Admin@12345")
        user = db.query(models.User).filter(models.User.email == email).first()
        if not user:
            user = models.User(
                username=username,
                email=email,
                hashed_password=crud.hash_password(password),
                role="ADMIN",
            )
            db.add(user)
            db.commit()
        else:
            user.username = username or user.username
            user.hashed_password = crud.hash_password(password)
            user.role = "ADMIN"
            db.commit()
except Exception:
    pass
finally:
    try:
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

# Serve built frontend if present (single-port mode)
try:
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DIST_DIR = BASE_DIR / "frontend-react" / "dist"
    if DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
except Exception:
    pass
