#!/usr/bin/env python3
"""
Product Ingestion Script
Parses OpenBeautyFacts CSV, filters for skincare, generates embeddings, 
and saves to a local SQLite database for the recommender system.
"""
import pandas as pd
import sqlite3
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Config
DATA_DIR = Path("data")
DATASETS_DIR = Path("datasets")
CSV_PATH = DATASETS_DIR / "en.openbeautyfacts.org.products.csv"
DB_PATH = DATA_DIR / "products.db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Keywords to filter Skincare products
SKINCARE_KEYWORDS = [
    "cream", "serum", "moisturizer", "cleanser", "lotion", "sunscreen", "spf", 
    "toner", "exfoliant", "mask", "peel", "acne", "anti-aging", "wrinkle", 
    "eye cream", "face wash", "gel", "balm", "oil"
]

# Keywords to exclude
EXCLUDE_KEYWORDS = [
    "shampoo", "conditioner", "hair", "scalp", "toothpaste", "dental", 
    "supplement", "pill", "capsule", "makeup", "lipstick", "mascara"
]

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT,
            brand TEXT,
            categories TEXT,
            image_url TEXT,
            ingredients_text TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

def ingest():
    print(f"Loading CSV from {CSV_PATH}...")
    # Read relevant columns only to save memory
    cols = ['code', 'product_name', 'brands', 'categories', 'image_url', 'ingredients_text']
    
    # Process in chunks because the file might be huge
    chunksize = 10000
    model = SentenceTransformer(MODEL_NAME)
    conn = setup_db()
    
    total_inserted = 0
    
    for chunk in pd.read_csv(CSV_PATH, sep='\t', usecols=cols, chunksize=chunksize, on_bad_lines='skip', low_memory=False):
        # Filter for Skincare
        chunk = chunk.dropna(subset=['product_name'])
        
        # Keyword Filer
        mask = chunk['product_name'].str.lower().str.contains('|'.join(SKINCARE_KEYWORDS), na=False)
        mask_exclude = chunk['product_name'].str.lower().str.contains('|'.join(EXCLUDE_KEYWORDS), na=False)
        
        # Also check categories column if available
        if 'categories' in chunk.columns:
            mask |= chunk['categories'].str.lower().str.contains('|'.join(SKINCARE_KEYWORDS), na=False)
            
        filtered = chunk[mask & ~mask_exclude].copy()
        
        if filtered.empty:
            continue
            
        # Create text for embedding: "Name: X. Brand: Y. Categories: Z. Ingredients: W."
        filtered['embedding_text'] = (
            "Name: " + filtered['product_name'].fillna('') + ". " +
            "Brand: " + filtered['brands'].fillna('') + ". " +
            "Categories: " + filtered['categories'].fillna('') + ". " +
            "Ingredients: " + filtered['ingredients_text'].fillna('')
        )
        
        # Generate Embeddings
        print(f"Embedding {len(filtered)} items...")
        embeddings = model.encode(filtered['embedding_text'].tolist(), show_progress_bar=False)
        
        # Insert into DB
        rows = []
        for i, row in enumerate(filtered.itertuples(index=False)):
            # row: code, product_name, brands, categories, image_url, ingredients_text, embedding_text
            # We need to map correctly based on dataframe column order
            # The order in itertuples matches the df columns
            
            # Pack embedding as binary (pickle or bytes)
            # Standard way: array.tobytes() -> requires reshaping on load
            emb_blob = embeddings[i].tobytes()
            
            rows.append((
                str(row.code),
                row.product_name,
                row.brands,
                row.categories,
                row.image_url,
                row.ingredients_text,
                emb_blob
            ))
            
        conn.executemany('''
            INSERT OR IGNORE INTO products (id, name, brand, categories, image_url, ingredients_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        conn.commit()
        
        total_inserted += len(rows)
        print(f"Inserted {total_inserted} skincare products so far...")
        
        # Limit for Dev (remove break for full ingestion)
        # if total_inserted > 5000: break 

    print(f"Done. Total Skincare Products: {total_inserted}")
    conn.close()

if __name__ == "__main__":
    ingest()
