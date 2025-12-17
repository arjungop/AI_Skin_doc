#!/usr/bin/env python3
"""
Setup admin user and approve doctor applications
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def create_admin():
    """Admin user is automatically created on app startup"""
    print("ℹ️ Admin user is automatically created on application startup")
    return True

def login_admin():
    """Login as admin user"""
    # Use default admin credentials from main.py bootstrap
    login_data = {
        "email": "admin@example.com",  # Default from ADMIN_EMAIL
        "password": "Admin@12345"       # Default from ADMIN_PASSWORD
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        token = response.json()["access_token"]
        print("✅ Admin logged in successfully")
        return token
    else:
        print(f"❌ Admin login failed: {response.text}")
        print("Note: Backend auto-creates admin user with default credentials")
        return None

def get_pending_doctors(token):
    """Get all pending doctor applications"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/admin/doctor_applications?status=PENDING", headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        doctors = result.get("items", [])
        print(f"📋 Found {len(doctors)} pending doctor applications")
        return doctors
    else:
        print(f"❌ Failed to get pending doctors: {response.text}")
        return []

def approve_doctor(token, application_id):
    """Approve a doctor application"""
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.post(f"{BASE_URL}/admin/doctor_applications/{application_id}/approve", 
                           headers=headers)
    
    if response.status_code == 200:
        print(f"✅ Doctor application {application_id} approved successfully")
        return True
    else:
        print(f"❌ Failed to approve doctor application {application_id}: {response.text}")
        return False

def main():
    print("🚀 Setting up admin and approving doctors...")
    
    # Step 1: Create admin user
    if not create_admin():
        return
    
    # Step 2: Login as admin
    token = login_admin()
    if not token:
        return
    
    # Step 3: Get pending doctors
    pending_doctors = get_pending_doctors(token)
    if not pending_doctors:
        print("ℹ️ No pending doctor applications found")
        return
    
    # Step 4: Approve all pending doctors
    print("\n📝 Approving doctor applications...")
    for doctor in pending_doctors:
        application_id = doctor.get("application_id")
        if application_id:
            print(f"Approving: {doctor.get('first_name', '')} {doctor.get('last_name', '')} ({doctor.get('email', 'Unknown')})")
            approve_doctor(token, application_id)
    
    print("\n🎉 Doctor approval process completed!")
    print("Now you can test the messaging system with proper doctor and patient lists.")

if __name__ == "__main__":
    main()