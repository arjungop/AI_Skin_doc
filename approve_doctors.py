#!/usr/bin/env python3
"""
Script to approve doctor applications and create admin user
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def login_admin():
    """Login as admin user (assume admin already exists)"""
    # Login as admin with correct format
    login_data = {
        "email": "admin@hospital.com",
        "password": "admin123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        token = response.json()["access_token"]
        print("Admin logged in successfully")
        return token
    else:
        print(f"Admin login failed: {response.text}")
        print("Please ensure admin user exists or try different credentials")
        return None

def approve_doctors(token):
    """Approve all pending doctor applications"""
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get pending applications
    response = requests.get(f"{BASE_URL}/admin/doctor_applications", headers=headers)
    if response.status_code != 200:
        print(f"Failed to get applications: {response.text}")
        return
    
    applications = response.json()
    print(f"Found {len(applications)} applications")
    
    for app in applications:
        if app["status"] == "PENDING":
            app_id = app["application_id"]
            approve_response = requests.post(
                f"{BASE_URL}/admin/doctor_applications/{app_id}/approve", 
                headers=headers
            )
            print(f"Approved application {app_id}: {approve_response.status_code}")
            if approve_response.status_code == 200:
                print(f"  Doctor {app['first_name']} {app['last_name']} approved")

if __name__ == "__main__":
    print("Approving doctor applications...")
    
    token = login_admin()
    if token:
        approve_doctors(token)
        print("Doctor applications processed!")
    else:
        print("Failed to login as admin")