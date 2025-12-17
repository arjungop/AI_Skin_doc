#!/usr/bin/env python3
"""
Quick script to create sample doctors and patients for testing messaging
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def create_patient(username, email, password, first_name, last_name, age, gender):
    """Create a patient"""
    data = {
        "username": username,
        "email": email,
        "password": password,
        "first_name": first_name,
        "last_name": last_name,
        "age": age,
        "gender": gender
    }
    
    response = requests.post(f"{BASE_URL}/patients/register", json=data)
    print(f"Patient {username}: {response.status_code}")
    if response.status_code == 200:
        print(f"Created patient: {response.json()}")
    else:
        print(f"Error: {response.text}")
    return response

def apply_doctor(username, email, password, first_name, last_name, specialization):
    """Apply as a doctor"""
    data = {
        "email": email,
        "password": password,
        "first_name": first_name,
        "last_name": last_name,
        "specialization": specialization
    }
    
    response = requests.post(f"{BASE_URL}/doctors/apply", json=data)
    print(f"Doctor application {email}: {response.status_code}")
    if response.status_code == 200:
        print(f"Applied as doctor: {response.json()}")
    else:
        print(f"Error: {response.text}")
    return response

if __name__ == "__main__":
    print("Creating sample users...")
    
    # Create some patients
    create_patient("john_doe", "john@example.com", "password123", "John", "Doe", 35, "Male")
    create_patient("jane_smith", "jane@example.com", "password123", "Jane", "Smith", 28, "Female")
    create_patient("bob_johnson", "bob@example.com", "password123", "Bob", "Johnson", 45, "Male")
    
    # Apply for doctors
    apply_doctor("dr_wilson", "wilson@hospital.com", "password123", "Sarah", "Wilson", "Dermatology")
    apply_doctor("dr_clark", "clark@hospital.com", "password123", "Michael", "Clark", "General Practice")
    apply_doctor("dr_brown", "brown@hospital.com", "password123", "Emily", "Brown", "Dermatology")
    
    print("\nSample users created! Note: Doctor applications need to be approved by admin.")