#!/usr/bin/env python3
"""
Test messaging system - verify patients can see doctors and vice versa
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def login_user(email, password):
    """Login and get token"""
    login_data = {"email": email, "password": password}
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Logged in as: {data.get('username', email)} (Role: {data.get('role', 'Unknown')})")
        return data["access_token"]
    else:
        print(f"❌ Login failed for {email}: {response.text}")
        return None

def test_list_doctors(token):
    """Test if we can see doctors list"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/doctors/", headers=headers)
    if response.status_code == 200:
        doctors = response.json()
        print(f"✅ Doctors list: {len(doctors)} doctors found")
        for doc in doctors[:3]:  # Show first 3
            print(f"   - {doc.get('first_name', '')} {doc.get('last_name', '')} ({doc.get('specialization', 'No specialization')})")
        return True
    else:
        print(f"❌ Failed to get doctors: {response.text}")
        return False

def test_list_patients(token):
    """Test if we can see patients list"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/patients/", headers=headers)
    if response.status_code == 200:
        patients = response.json()
        print(f"✅ Patients list: {len(patients)} patients found")
        for pat in patients[:3]:  # Show first 3
            print(f"   - {pat.get('first_name', '')} {pat.get('last_name', '')} ({pat.get('age', 'No age')} years)")
        return True
    else:
        print(f"❌ Failed to get patients: {response.text}")
        return False

def test_messaging_endpoints():
    """Test messaging endpoints with different user roles"""
    print("🧪 Testing messaging system endpoints...\n")
    
    # Test as patient (use one of our created patients)
    print("👤 Testing as PATIENT:")
    patient_token = login_user("john@example.com", "password123")
    if patient_token:
        print("   Testing access to doctors list:")
        test_list_doctors(patient_token)
        print("   Testing access to patients list:")
        test_list_patients(patient_token)  # Should fail for patient
    
    print("\n👩‍⚕️ Testing as DOCTOR:")
    doctor_token = login_user("brown@hospital.com", "password123")
    if doctor_token:
        print("   Testing access to patients list:")
        test_list_patients(doctor_token)
        print("   Testing access to doctors list:")
        test_list_doctors(doctor_token)  # Should also work
    
    print("\n👑 Testing as ADMIN:")
    admin_token = login_user("admin@example.com", "Admin@12345")
    if admin_token:
        print("   Testing access to doctors list:")
        test_list_doctors(admin_token)
        print("   Testing access to patients list:")
        test_list_patients(admin_token)

if __name__ == "__main__":
    test_messaging_endpoints()