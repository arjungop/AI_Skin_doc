
import sqlite3
import numpy as np
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load model once
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

class ProductRecommender:
    def __init__(self, db_path: str = "data/products.db"):
        self.db_path = db_path

    def search(self, query: str, limit: int = 5, min_score: float = 0.3, condition_text: str = None):
        """
        Semantic search for products.
        condition_text: e.g. "oily skin acne safe"
        """
        full_query = query
        if condition_text:
            full_query = f"{query} {condition_text}"
            
        # Encody query
        query_vec = model.encode(full_query)
        
        # Load all products (Optimized: In prod, use FAISS/ScaNN. For <10k, brute force is fine)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all embeddings
        # TODO: Caching mechanism for production
        cursor.execute("SELECT id, name, brand, image_url, embedding FROM products")
        products = cursor.fetchall()
        conn.close()
        
        results = []
        for p in products:
            # Decode binary embedding
            emb_blob = p['embedding']
            if not emb_blob: continue
            
            # Using frombuffer for speed
            prod_vec = np.frombuffer(emb_blob, dtype=np.float32)
            
            # Cosine Similarity
            # (A . B) / (|A| * |B|)
            # SBert vectors are normalized, so dot product == cosine sim
            score = np.dot(query_vec, prod_vec)
            
            if score >= min_score:
                results.append({
                    "id": p['id'],
                    "name": p['name'],
                    "brand": p['brand'],
                    "image": p['image_url'],
                    "score": float(score)
                })
        
        # Sort by score desc
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

# Singleton instance
recommender = ProductRecommender()
