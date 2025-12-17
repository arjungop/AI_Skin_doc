# backend/azure_blob.py
import os
import uuid
from azure.storage.blob import BlobServiceClient, ContentSettings

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "lesion-images")

_blob_service = None
_container_client = None

def _ensure_container():
    global _blob_service, _container_client
    if _blob_service is None:
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
        _blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    if _container_client is None:
        _container_client = _blob_service.get_container_client(AZURE_BLOB_CONTAINER)
        try:
            _container_client.create_container()
        except Exception:
            pass  # already exists
    return _container_client

def upload_bytes_get_url(data: bytes, filename: str, content_type: str) -> str:
    container = _ensure_container()
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "bin").lower()
    blob_name = f"{uuid.uuid4()}.{ext}"
    content_settings = ContentSettings(content_type=content_type or "application/octet-stream")
    blob_client = container.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
    return blob_client.url