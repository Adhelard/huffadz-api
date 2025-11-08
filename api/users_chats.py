# api/users_chats.py - Diperbarui untuk Skema Baru (Prompt/Answer)

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from firebase_config import db # Klien Firestore
from .auth import get_current_user_data # Dependency Auth
from typing import List, Optional
from google.cloud import firestore # Untuk SERVER_TIMESTAMP

router = APIRouter(tags=["Users, Conversations & Messages (Prompt/Answer)"])

# Root of /api/v1 for quick health check
@router.get("")
def api_root():
    return {"message": "API v1 OK"}
# -----------------------------------------------------------------
# --- DEBUG AUTH ENDPOINT (untuk verifikasi token) ---
# -----------------------------------------------------------------

@router.get("/debug/me")
def debug_me(user_data: dict = Depends(get_current_user_data)):
    """Mengembalikan payload token terverifikasi untuk debug (JANGAN gunakan di produksi)."""
    return user_data

# --- DEFENISI MODEL PYDANTIC (MIRIP STRUKTUR TABEL/DOKUMEN) ---

# Model untuk koleksi 'users' (Profile Data)
class UserProfile(BaseModel):
    # uid diambil dari token, tidak perlu di sini
    username: str
    photo_url: Optional[str] = None

# Model untuk koleksi 'conversations' (tidak berubah)
class Conversation(BaseModel):
    # conversation_id akan dibuat oleh Firestore
    user_id: str  # UID pengguna yang membuat/memiliki obrolan ini
    title: str
    # created_at dibuat oleh backend

# --- MODEL BARU: PROMPT (Pengganti Message untuk Sisi Pengguna) ---
class Prompt(BaseModel):
    # prompt_id akan dibuat oleh Firestore
    conversation_id: str # Kunci asing ke koleksi conversations
    sender_uid: str      # UID pengirim (Pengguna)
    prompt_text: str     # Konten prompt (Teks pertanyaan/input)
    # timestamp dibuat oleh backend

# --- MODEL BARU: ANSWER (Untuk Respon Sistem/AI) ---
class Answer(BaseModel):
    # answer_id akan dibuat oleh Firestore
    conversation_id: str # Kunci asing ke koleksi conversations
    # sender_uid (sistem/AI) tidak perlu jika kita selalu tahu ini dari AI
    prompt_id: str       # Kunci asing ke Prompt yang direspon
    answer_text: str     # Konten jawaban (Respon dari AI/Sistem)
    # timestamp dibuat oleh backend
# --- KONSTANTA NAMA KOLEKSI FIREBASE ---
USERS_COLLECTION = "users"
CONVERSATIONS_COLLECTION = "conversations"
PROMPTS_COLLECTION = "prompts"   # KOLEKSI BARU
ANSWERS_COLLECTION = "answers"


# -----------------------------------------------------------------
# --- ENDPOINT USER (USERS) ---
# -----------------------------------------------------------------

@router.post("/users/register", status_code=status.HTTP_201_CREATED)
def register_user_profile(profile: UserProfile, user_data: dict = Depends(get_current_user_data)):
    """Mendaftarkan/memperbarui profil pengguna (UID dari token digunakan sebagai ID Dokumen)."""
    user_uid = user_data.get('uid')
    
    try:
        data_to_save = profile.model_dump()
        data_to_save['email'] = user_data.get('email') # Simpan email dari Auth juga
        
        # Menambahkan/Mengganti dokumen dengan UID sebagai ID dokumen
        db.collection(USERS_COLLECTION).document(user_uid).set(data_to_save)
        return {"id": user_uid, **data_to_save}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/users/me")
def get_my_profile(user_data: dict = Depends(get_current_user_data)):
    """Mengambil detail profil pengguna saat ini."""
    user_uid = user_data.get('uid')
    doc = db.collection(USERS_COLLECTION).document(user_uid).get()
    
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail="Profil pengguna belum terdaftar di Firestore.")
        
    return {"id": doc.id, **doc.to_dict()}


# -----------------------------------------------------------------
# --- ENDPOINT CONVERSATIONS ---
# -----------------------------------------------------------------

# users_chats.py (Perbaikan)

@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(conv: Conversation, user_data: dict = Depends(get_current_user_data)):
    """Membuat obrolan baru (Conversation)."""
    # Pastikan user_id pada payload cocok dengan user yang terotentikasi
    if conv.user_id != user_data.get('uid'):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail="Anda tidak diizinkan membuat obrolan atas nama pengguna lain.")
    
    try:
        # 1. Data yang AMAN untuk respons JSON
        data_for_response = conv.model_dump()
        
        # 2. Data untuk WRITE ke Firestore (dengan Sentinel)
        data_for_db = data_for_response.copy() # Membuat salinan
        data_for_db['created_at'] = firestore.SERVER_TIMESTAMP # Menambahkan Sentinel ke salinan
        
        doc_ref = db.collection(CONVERSATIONS_COLLECTION).document()
        doc_ref.set(data_for_db) # <-- Gunakan data_for_db di sini
        
        # 3. Kembalikan data yang aman dari Sentinel (data_for_response)
        # Kita juga bisa menambahkan created_at dengan nilai None untuk kepastian skema
        # data_for_response['created_at'] = None 
        
        return {"conversation_id": doc_ref.id, **data_for_response}
    except Exception as e:
        # Pertahankan penanganan error Anda
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

@router.get("/conversations", response_model=List[dict])
def get_my_conversations(user_data: dict = Depends(get_current_user_data)):
    """Mengambil semua obrolan yang dibuat oleh pengguna saat ini."""
    user_uid = user_data.get('uid')
    conversations = []
    
    # PERHATIKAN: Indeks Firestore harus sudah Enabled agar kueri ini tidak 500 lagi.
    stream = db.collection(CONVERSATIONS_COLLECTION)\
               .where('user_id', '==', user_uid)\
               .order_by('created_at', direction=firestore.Query.DESCENDING)\
               .stream()
    
    for doc in stream:
        conversations.append({"conversation_id": doc.id, **doc.to_dict()})
        
    return conversations


# -----------------------------------------------------------------
# --- ENDPOINT MESSAGES ---
# -----------------------------------------------------------------

@router.post("/prompts", status_code=status.HTTP_201_CREATED)
def post_prompt(prmt: Prompt, user_data: dict = Depends(get_current_user_data)):
    """Mengirim Prompt baru ke Conversation tertentu."""
    sender_uid = user_data.get('uid')
    
    # Validasi 1: Pastikan sender_uid pada payload cocok dengan user yang terotentikasi
    if prmt.sender_uid != sender_uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail="ID pengirim prompt tidak cocok dengan pengguna yang terotentikasi.")
    
    # Validasi 2: Opsional - Pastikan Conversation_id itu valid dan pengguna memiliki akses
    # (Seperti di kode Anda, bagian ini di-komentari)
        
    try:
        # PENTING: Untuk aplikasi nyata, di sini adalah tempat Anda akan memanggil model AI
        # untuk mendapatkan 'answer' sebelum menyimpannya!
        
        # Simpan Prompt
        data_prompt = prmt.model_dump()
        data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP 
        
        doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
        doc_ref_prompt.set(data_prompt)
        prompt_id = doc_ref_prompt.id

        # Simulasikan/Proses Jawaban (Biasanya Asynchronous/Dipanggil ke AI Model)
        # --- SIMULASI JAWABAN (Ganti dengan Logic AI Nyata) ---
        simulated_answer_text = f"Ini adalah jawaban simulasi dari AI untuk prompt: '{prmt.prompt_text}'"
        
        data_answer = Answer(
            conversation_id=prmt.conversation_id, 
            prompt_id=prompt_id,
            answer_text=simulated_answer_text
        ).model_dump()
        data_answer['timestamp'] = firestore.SERVER_TIMESTAMP
        
        doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
        doc_ref_answer.set(data_answer)
        answer_id = doc_ref_answer.id
        # --------------------------------------------------------
        
        return {
            "prompt_id": prompt_id, 
            "answer_id": answer_id,
            "conversation_id": prmt.conversation_id, 
            "status": "prompt_and_answer_recorded"
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# -----------------------------------------------------------------
# --- ENDPOINT FETCH HISTORY (Menggantikan GET /messages/{conversation_id}) ---
# -----------------------------------------------------------------

@router.get("/history/{conversation_id}", response_model=List[dict])
def get_conversation_history(conversation_id: str, user_data: dict = Depends(get_current_user_data)):
    """Mengambil semua pasangan Prompt dan Answer untuk conversation_id tertentu."""
    
    # PENTING: Untuk keamanan, Anda harus memverifikasi bahwa user_data.uid 
    # memiliki izin untuk melihat Conversation ini sebelum mengambil data!

    # 1. Ambil semua Prompt untuk conversation_id ini, urutkan berdasarkan waktu
    prompts_stream = db.collection(PROMPTS_COLLECTION)\
                   .where('conversation_id', '==', conversation_id)\
                   .order_by('timestamp')\
                   .stream()
    
    # Buat dictionary untuk memetakan prompt_id ke data prompt
    prompts_map = {doc.id: doc.to_dict() for doc in prompts_stream}
    prompt_ids = list(prompts_map.keys())

    if not prompt_ids:
        return []

    # 2. Ambil semua Answer yang sesuai dengan prompt_ids yang ditemukan
    # NOTE: Firestore limit array 'in' clause is 10. For >10 prompts, you need multiple queries.
    if len(prompt_ids) > 100:
        # Logika untuk kueri batch diperlukan di sini (disederhanakan untuk contoh)
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Terlalu banyak prompt. Batas 10.")

    answers_stream = db.collection(ANSWERS_COLLECTION)\
                   .where('conversation_id', '==', conversation_id)\
                   .where('prompt_id', 'in', prompt_ids)\
                   .order_by('timestamp')\
                   .stream()
    
    # Buat dictionary untuk memetakan prompt_id ke data answer
    answers_map = {}
    for doc in answers_stream:
        data = doc.to_dict()
        # Simpan Answer ID (doc.id) dan data di dalam map
        answers_map[data['prompt_id']] = {"answer_id": doc.id, **data}
    
    # 3. Gabungkan Prompt dan Answer menjadi satu daftar kronologis
    history = []
    
    for prompt_id, prompt_data in prompts_map.items():
        answer_data = answers_map.get(prompt_id)
        
        # Tambahkan Prompt (ID tetap prompt_id)
        history.append({
            "type": "prompt",
            "id": prompt_id, # ID unik untuk Prompt
            "text": prompt_data.get('prompt_text'),
            "timestamp": prompt_data.get('timestamp')
        })
        
        # Tambahkan Answer yang sesuai jika ada
        if answer_data:
            history.append({
                "type": "answer",
                "id": answer_data['answer_id'], # <-- PERBAIKAN: Gunakan ID unik Answer!
                "text": answer_data.get('answer_text'),
                "timestamp": answer_data.get('timestamp')
            })
            
    # Urutkan ulang berdasarkan timestamp untuk memastikan urutan kronologis yang benar
    # (Meskipun harusnya sudah urut dari query, ini adalah pengamanan)
    history.sort(key=lambda x: x['timestamp'])

    return history