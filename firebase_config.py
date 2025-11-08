# firebase_config.py
import firebase_admin
from firebase_admin import credentials, firestore, auth

# --- KONFIGURASI ---
SERVICE_ACCOUNT_KEY_PATH = 'hafidz.json' 
# Pastikan ini sesuai dengan nama file yang Anda unduh

try:
    # Inisialisasi Firebase
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH) 
    firebase_app = firebase_admin.initialize_app(cred)
    
    # Klien Firestore (untuk interaksi database)
    db = firestore.client() 
    
    # Auth object (untuk verifikasi token)
    firebase_auth = auth 
    
    try:
        project_id = cred.project_id
    except Exception:
        project_id = None
    print(f"Firebase Admin SDK & Firestore berhasil diinisialisasi. PROJECT_ID={project_id}")
except Exception as e:
    print(f"ERROR: Gagal menginisialisasi Firebase. Pastikan file '{SERVICE_ACCOUNT_KEY_PATH}' benar: {e}")
    # Jika gagal, set ke None
    db = None
    firebase_auth = None