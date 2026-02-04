#!/bin/bash
source .venv/bin/activate
pip install sentence-transformers pandas sqlalchemy python-multipart
python scripts/ingest_products.py
