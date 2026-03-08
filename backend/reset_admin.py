from dotenv import load_dotenv
load_dotenv()
from backend.database import SessionLocal
from backend.models import User
from backend.crud import get_password_hash

db = SessionLocal()
admin = db.query(User).filter_by(email="admin@example.com").first()
if admin:
    print(f"Old hash: {admin.hashed_password[:10]}")
    admin.hashed_password = get_password_hash("Admin@12345")
    db.commit()
    print("Admin password reset successfully to Admin@12345!")
else:
    print("Admin not found!")
db.close()
