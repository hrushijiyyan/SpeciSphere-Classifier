import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import requests
import json
import hashlib
import os
import sqlite3
from activity_tracker import init_activity_db, save_user_activity, get_user_history

# Initialize activity DB
init_activity_db()

# ================
# AUTHENTICATION
# ================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

users = load_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "mode" not in st.session_state:
    st.session_state.mode = "login"

def login_ui():
    st.title("Login to Continue")

    if st.session_state.mode == "login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            hashed_pw = hash_password(password)
            if username in users and users[username] == hashed_pw:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("Don't have an account?")
        if st.button("Sign Up"):
            st.session_state.mode = "signup"
            st.rerun()

    elif st.session_state.mode == "signup":
        username = st.text_input("Create Username")
        password = st.text_input("Create Password", type="password")
        confirm_pw = st.text_input("Confirm Password", type="password")
        if st.button("Create Account"):
            if username in users:
                st.error("Username already exists.")
            elif password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                users[username] = hash_password(password)
                save_users(users)
                st.success("Account created successfully!")
                st.session_state.mode = "login"
                st.rerun()

        st.markdown("Already have an account?")
        if st.button("Back to Login"):
            st.session_state.mode = "login"
            st.rerun()

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# =========================
# LOGOUT & HISTORY SIDEBAR
# =========================

with st.sidebar:
    st.write(f"👤 Logged in as {st.session_state.username}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.mode = "login"
        st.rerun()

    st.markdown("---")
    st.subheader("🕒 Recent Activity")
    history = get_user_history(st.session_state.username)
    if history:
        for ts, breed, symptoms, diagnoses_str in history:
            st.markdown(f"**{ts}** — *{breed}*\nSymptoms: {symptoms}")
            try:
                diagnoses = json.loads(diagnoses_str)
                for diag in diagnoses:
                    st.markdown(f"• {diag['actual_diagnosis']} (Treatment: {diag['treatment']})")
            except Exception as e:
                st.markdown(f"(Error loading history: {e})")
    else:
        st.markdown("No history found.")

# =========================
# MAIN APP
# =========================

st.title("SpeciSphere Classifier")

from torchvision.models import ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_imagenet_labels():
    LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    try:
        response = requests.get(LABELS_URL)
        response.raise_for_status()
        return json.loads(response.text)
    except Exception:
        st.error("Failed to fetch ImageNet labels.")
        return None

class_idx_to_label = get_imagenet_labels()

def get_diseases_for_breed(breed):
    conn = sqlite3.connect("dog_disease.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT symptoms, typical_diagnosis, actual_diagnosis, treatment
        FROM diseases
        WHERE breed = ?
    ''', (breed,))
    rows = cursor.fetchall()
    conn.close()

    diseases = []
    for row in rows:
        symptoms_list = [s.strip().lower() for s in row[0].split(",")]
        diseases.append({
            "symptoms": symptoms_list,
            "typical_diagnosis": row[1],
            "actual_diagnosis": row[2],
            "treatment": row[3]
        })
    return diseases

use_webcam = st.checkbox("Use Webcam")

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

if use_webcam:
    st.write("Click below to capture an image:")
    if st.button("Capture Image"):
        image = capture_frame()
        if image:
            st.image(image, caption="Captured Image", use_column_width=True)
        else:
            st.error("Failed to capture image.")
else:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

if 'image' in locals():
    try:
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_idx = torch.argmax(probabilities).item()

        if class_idx_to_label:
            predicted_breed = class_idx_to_label[str(top_idx)][1]
            predicted_breed = predicted_breed.lower().replace("_", " ")
            st.write(f"*Predicted Breed:* {predicted_breed}")

            breed_diseases = get_diseases_for_breed(predicted_breed)
            if breed_diseases:
                st.write("### Enter Symptoms to Predict Disease")
                symptoms_input = st.text_area("Describe your dog's symptoms (comma-separated):", placeholder="e.g., vomiting, diarrhea, lethargy")

                if symptoms_input:
                    symptoms = [sym.strip().lower() for sym in symptoms_input.split(",")]
                    matched_diseases = []

                    for row in breed_diseases:
                        typical_diagnosis = row["typical_diagnosis"]
                        actual_diagnosis = row["actual_diagnosis"]
                        disease_symptoms = row["symptoms"]
                        treatment = row["treatment"]
                        match_count = sum(sym in disease_symptoms for sym in symptoms)

                        if match_count > 0:
                            matched_diseases.append({
                                "typical_diagnosis": typical_diagnosis,
                                "actual_diagnosis": actual_diagnosis,
                                "treatment": treatment,
                                "match_count": match_count
                            })

                    if matched_diseases:
                        st.write("### Possible Diagnoses and Treatments:")
                        for diag in matched_diseases:
                            st.write(f"- *Typical Diagnosis:* {diag['typical_diagnosis']}")
                            st.write(f"- *Actual Diagnosis:* {diag['actual_diagnosis']}")
                            st.write(f"  Treatment: {diag['treatment']} (matching symptoms: {diag['match_count']})")

                        save_user_activity(st.session_state.username, predicted_breed, symptoms, matched_diseases)
                    else:
                        st.write("No matching diseases found for the given symptoms.")
            else:
                st.write("This breed has no known diseases in the database.")
        else:
            st.error("Could not load ImageNet labels.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
