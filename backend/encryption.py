"""
Field-level encryption for sensitive medical data (PHI).

Uses Fernet symmetric encryption (AES-128-CBC with HMAC-SHA256) from the
`cryptography` library.  The encryption key is derived from the
FIELD_ENCRYPTION_KEY environment variable.

Usage in SQLAlchemy models:
    from backend.encryption import EncryptedText
    class MyModel(Base):
        sensitive_col = Column(EncryptedText(), nullable=True)

The column stores ciphertext in the database.  Values are transparently
decrypted when read via the ORM.
"""

from __future__ import annotations

import base64
import os
import logging

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import String, TypeDecorator

_log = logging.getLogger(__name__)

# Maximum ciphertext length (Fernet overhead + reasonable plaintext)
_MAX_CT_LEN = 8000


def _derive_fernet_key() -> bytes:
    """Derive a Fernet-compatible key from the env secret.

    If FIELD_ENCRYPTION_KEY looks like a raw 32-byte-URL-safe-base64 Fernet
    key it is used directly; otherwise PBKDF2 is used to derive one.
    """
    raw = os.getenv("FIELD_ENCRYPTION_KEY", "")
    if not raw:
        import warnings
        warnings.warn(
            "FIELD_ENCRYPTION_KEY is not set! Sensitive columns will NOT be encrypted. "
            "Set this in production to a strong random secret.",
            stacklevel=2,
        )
        return b""
    raw_bytes = raw.encode()
    # If it's already a valid Fernet key (44 url-safe base64 chars), use directly
    if len(raw_bytes) == 44:
        try:
            Fernet(raw_bytes)
            return raw_bytes
        except Exception:
            pass
    # Derive via PBKDF2
    salt = os.getenv("FIELD_ENCRYPTION_SALT", "skin-doc-phi-salt").encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(raw_bytes))


_FERNET_KEY = _derive_fernet_key()
_fernet = Fernet(_FERNET_KEY) if _FERNET_KEY else None


class EncryptedText(TypeDecorator):
    """SQLAlchemy type that transparently encrypts/decrypts text columns."""

    impl = String(_MAX_CT_LEN)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not _fernet:
            return value  # no-op when key not configured (dev only)
        return _fernet.encrypt(value.encode()).decode()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if not _fernet:
            return value
        try:
            return _fernet.decrypt(value.encode()).decode()
        except InvalidToken:
            # Value was stored before encryption was enabled — return as-is
            _log.debug("Could not decrypt value (possibly pre-encryption plaintext)")
            return value
