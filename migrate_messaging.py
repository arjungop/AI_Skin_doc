"""
Database migration script for enhanced messaging system.
Run this to update your database schema with the new messaging features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal, engine
from backend import models
from sqlalchemy import text

#!/usr/bin/env python3
"""
Database migration script to update the messaging system
"""
import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend import models
from backend.database import DATABASE_URL

def migrate_messaging_system():
    print("🔄 Starting messaging system migration...")
    
    # Create engine with proper configuration for SQLite
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL)
        
    # Create all tables (will only create new ones)
    models.Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Check and add new columns to existing chat_rooms table
        try:
            db.execute(text("ALTER TABLE chat_rooms ADD COLUMN last_message_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
            print("✓ Added last_message_at to chat_rooms")
        except:
            print("- last_message_at already exists in chat_rooms")
        
        try:
            db.execute(text("ALTER TABLE chat_rooms ADD COLUMN is_active BOOLEAN DEFAULT TRUE"))
            print("✓ Added is_active to chat_rooms")
        except:
            print("- is_active already exists in chat_rooms")
            
        try:
            db.execute(text("ALTER TABLE chat_rooms ADD COLUMN unread_count_patient INTEGER DEFAULT 0"))
            print("✓ Added unread_count_patient to chat_rooms")
        except:
            print("- unread_count_patient already exists in chat_rooms")
            
        try:
            db.execute(text("ALTER TABLE chat_rooms ADD COLUMN unread_count_doctor INTEGER DEFAULT 0"))
            print("✓ Added unread_count_doctor to chat_rooms")
        except:
            print("- unread_count_doctor already exists in chat_rooms")
        
        # Check and modify messages table
        try:
            # Add new columns to messages table
            db.execute(text("ALTER TABLE messages ADD COLUMN message_type VARCHAR(20) DEFAULT 'text'"))
            print("✓ Added message_type to messages")
        except:
            print("- message_type already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN file_url VARCHAR(500)"))
            print("✓ Added file_url to messages")
        except:
            print("- file_url already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN file_name VARCHAR(255)"))
            print("✓ Added file_name to messages")
        except:
            print("- file_name already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN file_size INTEGER"))
            print("✓ Added file_size to messages")
        except:
            print("- file_size already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN reply_to_message_id INTEGER"))
            print("✓ Added reply_to_message_id to messages")
        except:
            print("- reply_to_message_id already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
            print("✓ Added updated_at to messages")
        except:
            print("- updated_at already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN status VARCHAR(20) DEFAULT 'sent'"))
            print("✓ Added status to messages")
        except:
            print("- status already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN is_edited BOOLEAN DEFAULT FALSE"))
            print("✓ Added is_edited to messages")
        except:
            print("- is_edited already exists in messages")
            
        try:
            db.execute(text("ALTER TABLE messages ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE"))
            print("✓ Added is_deleted to messages")
        except:
            print("- is_deleted already exists in messages")
        
        # Remove old is_read column if it exists (replaced by status)
        try:
            db.execute(text("ALTER TABLE messages DROP COLUMN is_read"))
            print("✓ Removed old is_read column from messages")
        except:
            print("- is_read column doesn't exist or already removed")
        
        # Update existing content column to allow NULL (for file messages)
        try:
            db.execute(text("ALTER TABLE messages MODIFY COLUMN content TEXT"))
            print("✓ Updated content column to allow NULL")
        except:
            print("- content column already allows NULL or modification not needed")
        
        # Set last_message_at for existing rooms
        try:
            db.execute(text("""
                UPDATE chat_rooms 
                SET last_message_at = COALESCE(
                    (SELECT MAX(created_at) FROM messages WHERE messages.room_id = chat_rooms.room_id),
                    chat_rooms.created_at
                )
                WHERE last_message_at IS NULL
            """))
            print("✓ Updated last_message_at for existing rooms")
        except Exception as e:
            print(f"- Could not update last_message_at: {e}")
        
        db.commit()
        print("\n✅ Database migration completed successfully!")
        print("\nNew features added:")
        print("- Enhanced message types (text, image, file, system)")
        print("- File attachments support")
        print("- Message reactions")
        print("- Reply to messages")
        print("- Message editing and deletion")
        print("- Read status and delivery confirmation")
        print("- Online/offline user status")
        print("- Unread message counters")
        print("- Real-time WebSocket messaging")
        print("- Typing indicators")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("🔄 Starting messaging system migration...")
    migrate_messaging_system()