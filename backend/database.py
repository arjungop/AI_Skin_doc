import os
import urllib.parse
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

# Connection priority:
# 1) DATABASE_URL (generic SQLAlchemy URL, e.g., mysql+pymysql://user:pass@host:3306/db)
# 2) AZURE_SQL_CONNECTION_STRING (mssql+pyodbc://...)
# 3) sqlite:///./dev.db (fallback for quick local dev)

def _build_mysql_url():
    host = os.getenv("MYSQL_HOST")
    db = os.getenv("MYSQL_DB")
    user = os.getenv("MYSQL_USER")
    pwd = os.getenv("MYSQL_PASSWORD")
    port = os.getenv("MYSQL_PORT", "3306")
    if host and db and user and pwd:
        # URL-encode credentials to safely handle special chars like '@' or ':'
        user_q = urllib.parse.quote_plus(user)
        pwd_q = urllib.parse.quote_plus(pwd)
        return f"mysql+pymysql://{user_q}:{pwd_q}@{host}:{port}/{db}?charset=utf8mb4"
    return None

DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or _build_mysql_url()
    or os.getenv("AZURE_SQL_CONNECTION_STRING")
    or "sqlite:///./dev.db"
)

# Extra connect args for SQLite
is_sqlite = DATABASE_URL.startswith("sqlite")
is_mysql = DATABASE_URL.startswith("mysql+") or DATABASE_URL.startswith("mysql")

# Optional hard enforcement: require MySQL only
if os.getenv("DB_REQUIRE", "").lower() in {"mysql", "mysql_only", "mysql-only"} and not is_mysql:
    raise RuntimeError(
        "DB_REQUIRE=mysql is set but DATABASE_URL does not point to MySQL. "
        "Set DATABASE_URL or MYSQL_* to a MySQL connection."
    )

connect_args = {"check_same_thread": False} if is_sqlite else {}
if is_mysql:
    # Ensure UTC time and utf8mb4
    connect_args.update({
        "charset": "utf8mb4",
        "init_command": "SET time_zone='+00:00'",
    })

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
    pool_recycle=280 if is_mysql else None,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_simple_migrations():
    """Best-effort, idempotent migrations for SQLite dev DB.

    Adds columns that were introduced after the initial table creation.
    This avoids needing Alembic for local development.
    """
    try:
        if engine.dialect.name != 'sqlite':
            return
        with engine.begin() as conn:
            # Ensure 'category' column on transactions
            res = conn.exec_driver_sql("PRAGMA table_info(transactions)")
            cols = [row[1] for row in res.fetchall()]
            if 'category' not in cols:
                conn.exec_driver_sql("ALTER TABLE transactions ADD COLUMN category VARCHAR(30) DEFAULT 'general'")
    except Exception:
        # Non-fatal; keep app running
        pass

# Enforce foreign keys on SQLite
if is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: N802
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception:
            pass
