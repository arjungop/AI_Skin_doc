#!/usr/bin/env python3
import os
from typing import Type, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url, URL
from sqlalchemy.orm import sessionmaker
from backend import models


def copy_table(src_sess, dst_sess, Model: Type[models.Base], batch: int = 500):
    """Copy rows from SQLite source into destination, tolerating schema drift.

    Selects only columns that exist in the source table and maps them into the
    destination ORM model, letting destination defaults fill missing columns.
    """
    table = Model.__tablename__
    dest_cols = {c.name for c in Model.__table__.columns}
    pk_cols = [c.name for c in Model.__table__.primary_key.columns]
    # discover source columns
    try:
        src_cols = [row[1] for row in src_sess.execute(text(f"PRAGMA table_info({table})")).fetchall()]
    except Exception:
        # Fallback: select all and infer keys from first row
        src_cols = []
    select_cols = [c for c in src_cols if c in dest_cols] if src_cols else list(dest_cols)
    if not select_cols:
        print(f"  skipping {table}: no overlapping columns")
        return
    # Optionally truncate destination table first
    if os.getenv("MIGRATE_TRUNCATE", "0") in {"1", "true", "TRUE", "yes", "on"}:
        try:
            dst_sess.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        except Exception:
            pass
        dst_sess.execute(text(f"TRUNCATE TABLE `{table}`"))
        try:
            dst_sess.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        except Exception:
            pass
        dst_sess.commit()

    # Build set of existing PKs to skip duplicates (default behavior)
    existing_pks = set()
    skip_existing = os.getenv("MIGRATE_SKIP_EXISTING", "1") in {"1", "true", "TRUE", "yes", "on"}
    if skip_existing and len(pk_cols) == 1:
        try:
            res = dst_sess.execute(text(f"SELECT `{pk_cols[0]}` FROM `{table}`"))
            existing_pks = {row[0] for row in res.fetchall()}
        except Exception:
            existing_pks = set()

    # Preload FK parent ID sets for certain tables to avoid FK errors
    parent_sets = {}
    if table == 'appointments':
        # Build parent ID sets from destination; these will be updated as we backfill
        try:
            ids = dst_sess.execute(text("SELECT doctor_id FROM doctors")).fetchall()
            parent_sets['doctor_id'] = {r[0] for r in ids}
        except Exception:
            parent_sets['doctor_id'] = set()
        try:
            ids = dst_sess.execute(text("SELECT patient_id FROM patients")).fetchall()
            parent_sets['patient_id'] = {r[0] for r in ids}
        except Exception:
            parent_sets['patient_id'] = set()

        # Helpers to backfill parent chains from SQLite into MySQL
        def _get_src_row(table: str, pk: str, value: Any) -> Optional[Dict[str, Any]]:
            try:
                r = src_sess.execute(text(f"SELECT * FROM {table} WHERE {pk}=:v"), {"v": value}).mappings().first()
                return dict(r) if r else None
            except Exception:
                return None

        def _insert_row(model_cls: Type[models.Base], data: Dict[str, Any]) -> bool:
            try:
                dest_cols_local = {c.name for c in model_cls.__table__.columns}
                clean = {k: v for k, v in data.items() if k in dest_cols_local}
                obj = model_cls(**clean)
                dst_sess.add(obj)
                dst_sess.commit()
                return True
            except Exception:
                dst_sess.rollback()
                return False

        def _ensure_user(user_id: Any) -> bool:
            if user_id is None:
                return False
            exists = dst_sess.execute(text("SELECT 1 FROM users WHERE user_id=:i"), {"i": user_id}).first()
            if exists:
                return True
            src = _get_src_row('users', 'user_id', user_id)
            if not src:
                return False
            return _insert_row(models.User, src)

        def _ensure_patient(patient_id: Any) -> bool:
            if patient_id is None:
                return False
            exists = dst_sess.execute(text("SELECT 1 FROM patients WHERE patient_id=:i"), {"i": patient_id}).first()
            if exists:
                return True
            src = _get_src_row('patients', 'patient_id', patient_id)
            if not src:
                return False
            # ensure parent user
            if not _ensure_user(src.get('user_id')):
                return False
            ok = _insert_row(models.Patient, src)
            if ok:
                parent_sets['patient_id'].add(patient_id)
            return ok

        def _ensure_doctor(doctor_id: Any) -> bool:
            if doctor_id is None:
                return False
            exists = dst_sess.execute(text("SELECT 1 FROM doctors WHERE doctor_id=:i"), {"i": doctor_id}).first()
            if exists:
                return True
            src = _get_src_row('doctors', 'doctor_id', doctor_id)
            if not src:
                return False
            if not _ensure_user(src.get('user_id')):
                return False
            ok = _insert_row(models.Doctor, src)
            if ok:
                parent_sets['doctor_id'].add(doctor_id)
            return ok

    # Count rows in source
    total = src_sess.execute(text(f"SELECT COUNT(1) FROM {table}")).scalar() or 0
    offset = 0
    while offset < total:
        rows = src_sess.execute(
            text(f"SELECT {', '.join(select_cols)} FROM {table} LIMIT :limit OFFSET :offset"),
            {"limit": batch, "offset": offset},
        ).mappings().all()
        if not rows:
            break
        objs = []
        for r in rows:
            data = {k: r.get(k) for k in select_cols if k in dest_cols}
            # Filter rows that would violate FKs (appointments -> doctors/patients)
            if table == 'appointments':
                # Backfill parents if missing
                did = data.get('doctor_id')
                pid = data.get('patient_id')
                if did is not None and did not in parent_sets.get('doctor_id', set()):
                    if not _ensure_doctor(did):
                        continue
                if pid is not None and pid not in parent_sets.get('patient_id', set()):
                    if not _ensure_patient(pid):
                        continue
            # Skip if PK already exists
            if skip_existing and len(pk_cols) == 1 and pk_cols[0] in data and data[pk_cols[0]] in existing_pks:
                continue
            objs.append(Model(**data))
        if objs:
            dst_sess.bulk_save_objects(objs)
            dst_sess.commit()
        offset += len(rows)
        print(f"  copied {offset}/{total} -> {table}")


def main():
    sqlite_path = os.getenv("SQLITE_PATH", "dev.db")
    src_url = f"sqlite:///{sqlite_path}"
    dst_url = os.getenv("DATABASE_URL")
    if not dst_url or not dst_url.startswith("mysql"):
        raise SystemExit("Set DATABASE_URL to a MySQL URL (mysql+pymysql://...)")

    src_engine = create_engine(src_url)
    # Ensure target database exists; if not, create it
    url = make_url(dst_url)
    dbname = url.database
    try:
        dst_engine = create_engine(dst_url, pool_pre_ping=True)
        with dst_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        # Connect to server-level URL (without database) and create DB
        server_url = URL.create(
            url.drivername,
            username=url.username,
            password=url.password,
            host=url.host,
            port=url.port,
            database=None,
            query=url.query,
        )
        srv_engine = create_engine(server_url, pool_pre_ping=True)
        with srv_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{dbname}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        srv_engine.dispose()
        dst_engine = create_engine(dst_url, pool_pre_ping=True)
    SrcSession = sessionmaker(bind=src_engine)
    DstSession = sessionmaker(bind=dst_engine)
    src = SrcSession()
    dst = DstSession()

    # Ensure tables exist on destination
    models.Base.metadata.create_all(bind=dst_engine)

    tables = [
        models.User,
        models.Patient,
        models.Doctor,
        models.DoctorApplication,
        models.Appointment,
        models.Lesion,
        models.Transaction,
        models.TransactionMeta,
        models.ChatRoom,
        models.Message,
        models.DoctorAvailability,
        models.AIChatSession,
        models.AIChatMessage,
    ]

    try:
        for Model in tables:
            print(f"Copying {Model.__tablename__}...")
            copy_table(src, dst, Model)
        print("Done.")
    finally:
        try:
            src.close(); dst.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
