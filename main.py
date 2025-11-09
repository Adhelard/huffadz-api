# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.users_chats import router as users_chats_router # Impor router

app = FastAPI(
    title="FastAPI Firebase Chat API",
    description="API untuk manajemen User dan Chat menggunakan FastAPI dan Firestore."
)

# CORS untuk pengembangan lokal (Vite default: 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://huffadz.portalsi.com",     
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["*"],
    allow_credentials=True,
)

# Daftarkan semua router API
app.include_router(users_chats_router, prefix="/api/v1")

@app.get("/")
def home():
    return {"message": "Selamat datang di API Chat! Kunjungi /docs untuk melihat endpoint."}

# CARA MENJALANKAN (di terminal):
# uvicorn main:app --reload