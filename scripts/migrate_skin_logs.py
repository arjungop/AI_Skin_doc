#!/usr/bin/env python3
"""
Migration script to fix the skin_logs.image_path column size issue.
This allows storing base64 image data instead of just file paths.
"""

from sqlalchemy import create_engine, text
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database import SQLALCHEMY_DATABASE_URL

def migrate():
    print("🔧 Applying database migration: Fix skin_logs.image_path column...")
    
    try:
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        
        with engine.connect() as conn:
            # Change image_path from String(500) to TEXT
            conn.execute(text("ALTER TABLE skin_logs MODIFY COLUMN image_path TEXT NULL"))
            conn.commit()
            print("✅ Migration completed successfully!")
            print("   - skin_logs.image_path now supports base64 images")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("\nYou can manually run this SQL:")
        print("ALTER TABLE skin_logs MODIFY COLUMN image_path TEXT NULL;")
        return False
    
    return True

if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)
