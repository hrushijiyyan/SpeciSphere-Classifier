import sqlite3
from datetime import datetime
import json

# Initialize SQLite DB
def init_activity_db():
    conn = sqlite3.connect('user_activity.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            breed TEXT,
            symptoms TEXT,
            diagnoses TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Save user activity to DB
def save_user_activity(username, breed, symptoms, matched_diseases):
    conn = sqlite3.connect('user_activity.db')
    c = conn.cursor()

    # Convert matched_diseases to JSON string for storage
    matched_diseases_json = json.dumps(matched_diseases)

    c.execute('''
        INSERT INTO activity (username, breed, symptoms, diagnoses)
        VALUES (?, ?, ?, ?)
    ''', (username, breed, ', '.join(symptoms), matched_diseases_json))

    conn.commit()
    conn.close()

# Fetch user activity history (with optional limit)
def get_user_history(username, limit=5):
    conn = sqlite3.connect('user_activity.db')
    c = conn.cursor()

    c.execute('''
        SELECT timestamp, breed, symptoms, diagnoses
        FROM activity
        WHERE username = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (username, limit))

    rows = c.fetchall()
    conn.close()

    return rows
