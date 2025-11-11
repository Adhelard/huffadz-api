from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_client import QdrantClient
from dotenv import load_dotenv # Pastikan .env dimuat
import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from firebase_config import db # Klien Firestore
from .auth import get_current_user_data # Dependency Auth
from typing import List, Optional, Dict, Any
from google.cloud import firestore # Untuk SERVER_TIMESTAMP
import time # Untuk simulasi timestamp unik jika SERVER_TIMESTAMP tidak realtime saat testing


load_dotenv()
router = APIRouter(tags=["Users, Conversations & Messages (Prompt/Answer)"])

COLLECTION_NAME = "quran_hadith_data"

# Global variable untuk menyimpan RAG Chain
RAG_CHAIN = None

def initialize_rag_chain():
    """Fungsi untuk inisialisasi semua komponen RAG."""
    global RAG_CHAIN
    
    try:
        # Client dan Embeddings
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

        # Vector Store (Terhubung ke koleksi yang sudah di-index)
        vectorstore = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
        
        # Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Ambil 5 dokumen teratas

        # LLM
        llm = ChatOpenAI(model="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
        
        # Prompt Template untuk Generasi Jawaban
        system_prompt = (
            "Anda adalah asisten AI yang ahli dalam Al-Qur'an dan Hadis. "
            "Jawab pertanyaan pengguna HANYA berdasarkan konteks yang diberikan. "
            "Jika jawaban tidak ditemukan dalam konteks, katakan 'Saya tidak dapat menemukan jawaban yang spesifik dari data Al-Qur'an dan Hadis yang tersedia.' "
            "Sebutkan sumber (Surah dan Ayat atau Perawi Hadis) jika memungkinkan."
            "\n\nKonteks:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                # MessagesPlaceholder(variable_name="chat_history"), # Untuk riwayat
                ("human", "{input}"),
            ]
        )

        # Chain untuk menggabungkan dokumen dan menghasilkan jawaban
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # RAG Chain Final
        RAG_CHAIN = create_retrieval_chain(retriever, question_answer_chain)
        print("RAG Chain berhasil diinisialisasi.")

    except Exception as e:
        print(f"Error inisialisasi RAG Chain: {e}. Pastikan Qdrant berjalan dan .env benar.")
        RAG_CHAIN = None

# Jalankan inisialisasi saat startup
initialize_rag_chain()

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

# -----------------------------------------------------------------
# --- DEFENISI MODEL PYDANTIC (MIRIP STRUKTUR TABEL/DOKUMEN) ---
# -----------------------------------------------------------------

# Model untuk koleksi 'users' (Profile Data)

class RAGQuery(BaseModel):
    query: str = Field(..., description="Pertanyaan yang diajukan ke RAG.")
    conversation_id: str = Field(..., description="ID percakapan saat ini.")
class UserProfile(BaseModel):
    username: str
    photo_url: Optional[str] = None

# Model untuk koleksi 'conversations'
class Conversation(BaseModel):
    user_id: str  # UID pengguna yang membuat/memiliki obrolan ini
    title: str

# --- PROMPT (Pengganti Message untuk Sisi Pengguna) ---
class Prompt(BaseModel):
    conversation_id: str # Kunci asing ke koleksi conversations
    sender_uid: str      # UID pengirim (Pengguna)
    prompt_text: str     # Konten prompt (Teks pertanyaan/input)

# --- SUB-MODEL UNTUK KONTEN JAWABAN KOMPLEKS (SMART ANSWER) ---
class LetterContent(BaseModel):
    """Struktur untuk konten Surat."""
    letter_type: str
    recipient: str
    sender: str
    date: str
    salutation: str
    body_paragraphs: List[str]
    closing: str
    

class QuranicContent(BaseModel):
    """Struktur untuk konten Ayat Al-Qur'an."""
    surah_name: str = Field(..., description="Nama Surah (e.g., Al-Baqarah).")
    surah_number: int = Field(..., description="Nomor Surah.")
    ayah_number: str = Field(..., description="Nomor Ayat.")
    arabic_text: Optional[str] = Field(None, description="Teks Ayat dalam Bahasa Arab.")
    translation: str = Field(..., description="Terjemahan Ayat.")
    tafsir_summary: Optional[str] = Field(None, description="Ringkasan tafsir atau konteks ayat.")

class HadithContent(BaseModel):
    """Struktur untuk konten Hadits."""
    book: str
    number: str
    narrator: Optional[str] = None
    arabic_text: Optional[str] = None
    translation: str
    details: Optional[str] = None

# --- MODEL BARU: SMART ANSWER (Untuk Respon Sistem/AI) ---
class SmartAnswer(BaseModel):
    # Properti wajib untuk relasi
    conversation_id: str
    prompt_id: str 
    
    # Konten utama (fallback atau ringkasan)
    summary_text: str = Field(..., description="Ringkasan atau teks utama dari jawaban AI.")
    
    # Konten terstruktur opsional
    letter_example: Optional[LetterContent] = None
    hadith_example: Optional[HadithContent] = None
    quran_example: Optional[QuranicContent] = None 
    sources: List[str] = Field(default_factory=list, description="Daftar sumber yang digunakan.")


# --- KONSTANTA NAMA KOLEKSI FIREBASE ---
USERS_COLLECTION = "users"
CONVERSATIONS_COLLECTION = "conversations"
PROMPTS_COLLECTION = "prompts"
ANSWERS_COLLECTION = "answers" # Menggunakan 'answers' untuk koleksi SmartAnswer


@router.post("/rag/chat", status_code=status.HTTP_200_OK)
async def rag_chat(
    data: RAGQuery,
    user_data: dict = Depends(get_current_user_data) # Menggunakan otentikasi
):
    """Endpoint untuk mengajukan pertanyaan ke RAG berbasis data Quran/Hadis."""
    if RAG_CHAIN is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Service belum siap. Cek log server untuk inisialisasi."
        )
    
    # 1. Invoke RAG Chain
    try:
        response = RAG_CHAIN.invoke({"input": data.query})
        
        answer = response["answer"]
        source_documents = response.get("context", [])
        
        sources = []
        for doc in source_documents:
            # Ambil hanya metadata penting untuk ditampilkan
            source_info = {
                "source_type": doc.metadata.get('data_type', 'N/A'),
                "content_snippet": doc.page_content[:150] + "...",
                "metadata_surah": f"Q.S. {doc.metadata.get('surah_latin')} ayat {doc.metadata.get('ayah')}" if doc.metadata.get('surah_latin') else None,
                "metadata_hadith": f"Perawi: {doc.metadata.get('Perawi')}" if doc.metadata.get('Perawi') else None,
            }
            sources.append({k: v for k, v in source_info.items() if v is not None})
        
        # 2. Simpan Riwayat ke Firestore (Menggunakan fungsi yang sudah ada)
        # Asumsikan Anda memiliki fungsi untuk menyimpan prompt dan jawaban
        # user_id = user_data['user_id']
        # Simpan prompt:
        # prompt_id = await save_prompt(user_id, data.conversation_id, data.query)
        # Simpan jawaban (sertakan sources di 'content'):
        # await save_answer(user_id, data.conversation_id, prompt_id, answer, sources)
        
        return {
            "answer": answer,
            "sources": sources,
            "model": "RAG (Qdrant + OpenAI LLM)"
        }
    
    except Exception as e:
        print(f"Error RAG Chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat memproses kueri: {e}"
        )


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

@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(conv: Conversation, user_data: dict = Depends(get_current_user_data)):
    """Membuat obrolan baru (Conversation)."""
    if conv.user_id != user_data.get('uid'):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail="Anda tidak diizinkan membuat obrolan atas nama pengguna lain.")
    
    try:
        data_for_response = conv.model_dump()
        data_for_db = data_for_response.copy()
        data_for_db['created_at'] = firestore.SERVER_TIMESTAMP
        
        doc_ref = db.collection(CONVERSATIONS_COLLECTION).document()
        doc_ref.set(data_for_db)
        
        return {"conversation_id": doc_ref.id, **data_for_response}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

@router.get("/conversations", response_model=List[dict])
def get_my_conversations(user_data: dict = Depends(get_current_user_data)):
    """Mengambil semua obrolan yang dibuat oleh pengguna saat ini."""
    user_uid = user_data.get('uid')
    conversations = []
    
    stream = db.collection(CONVERSATIONS_COLLECTION)\
               .where('user_id', '==', user_uid)\
               .order_by('created_at', direction=firestore.Query.DESCENDING)\
               .stream()
    
    for doc in stream:
        conversations.append({"conversation_id": doc.id, **doc.to_dict()})
        
    return conversations


# -----------------------------------------------------------------
# --- ENDPOINT PROMPTS (Mencakup Logic Dummy Smart Answer) ---
# -----------------------------------------------------------------

def generate_smart_answer_dummy(conversation_id: str, prompt_id: str, prompt_text: str) -> SmartAnswer:
    """Fungsi dummy yang menghasilkan SmartAnswer berdasarkan prompt_text."""
    text_lower = prompt_text.lower()

    if "hadits" in text_lower or "ilmu" in text_lower:
        return SmartAnswer(
            conversation_id=conversation_id, 
            prompt_id=prompt_id,
            summary_text="Jawaban hadits terstruktur mengenai menuntut ilmu.",
            hadith_example=HadithContent(
                book="Shahih Muslim",
                number="2699",
                narrator="Abu Hurairah",
                translation="Barang siapa menempuh suatu jalan untuk menuntut ilmu, niscaya Allah mudahkan baginya jalan menuju surga.",
                details="Hadits ini menekankan keutamaan mencari ilmu sebagai jalan menuju Jannah (Surga)."
            ),
            sources=["Shahih Muslim, No. 2699"]
        )
    
    elif "surat" in text_lower or "lamaran" in text_lower:
        return SmartAnswer(
            conversation_id=conversation_id, 
            prompt_id=prompt_id,
            summary_text="Draf Surat Lamaran Kerja untuk posisi Junior Dev.",
            letter_example=LetterContent(
                letter_type="Surat Lamaran Kerja",
                recipient="Kepala HRD, PT Teknologi Sukses",
                sender="Andi Pratama",
                date="10 Nov 2025",
                salutation="Dengan hormat,",
                body_paragraphs=[
                    "Saya mengajukan permohonan untuk posisi Junior Developer, sesuai iklan di website perusahaan.",
                    "Saya menguasai Python dan FastAPI, dan sangat antusias untuk bergabung dengan tim Anda."
                ],
                closing="Atas perhatiannya, saya ucapkan terima kasih."
            ),
            sources=["Template standar HR"]
        )
    
    else:
        # Jawaban default sederhana
        return SmartAnswer(
            conversation_id=conversation_id, 
            prompt_id=prompt_id,
            summary_text="Ayat ini menjelaskan pentingnya sabar dan shalat sebagai penolong.",
            quran_example=QuranicContent(
                surah_name="Al-Baqarah",
                surah_number=2,
                ayah_number="153",
                arabic_text="يَا أَيُّهَا الَّذِينَ آمَنُوا اسْتَعِينُوا بِالصَّبْرِ وَالصَّلَاةِ ۚ إِنَّ اللَّهَ مَعَ الصَّابِرِينَ",
                translation="Wahai orang-orang yang beriman! Mohonlah pertolongan (kepada Allah) dengan sabar dan salat. Sungguh, Allah beserta orang-orang yang sabar.",
                tafsir_summary="Ayat ini memerintahkan kepada kaum mukminin untuk menjadikan sabar dan salat sebagai media utama dalam memohon pertolongan dan menghadapi kesulitan hidup."
            ),
            sources=["QS. Al-Baqarah: 153"]
        )


@router.post("/prompts", status_code=status.HTTP_201_CREATED)
def post_prompt(prmt: Prompt, user_data: dict = Depends(get_current_user_data)):
    """Mengirim Prompt baru dan memicu pembuatan SmartAnswer simulasi."""
    sender_uid = user_data.get('uid')
    
    if prmt.sender_uid != sender_uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail="ID pengirim prompt tidak cocok dengan pengguna yang terotentikasi.")
        
    try:
        # 1. Simpan Prompt
        data_prompt = prmt.model_dump()
        data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP 
        
        doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
        doc_ref_prompt.set(data_prompt)
        prompt_id = doc_ref_prompt.id

        # 2. Proses/Simulasikan Jawaban (Panggil fungsi dummy yang memuat logika)
        # 🚨 PASTIKAN PANGGILAN INI ADA DAN MENGGUNAKAN LOGIKA YANG BENAR 🚨
        smart_answer_obj = generate_smart_answer_dummy(
            conversation_id=prmt.conversation_id, 
            prompt_id=prompt_id, 
            prompt_text=prmt.prompt_text # Kirim teks prompt untuk penentuan jenis jawaban
        )
        
        # 3. Simpan SmartAnswer ke Firestore
        # Gunakan exclude_none=True agar field yang kosong (null) tidak disimpan, menghemat penyimpanan.
        data_answer = smart_answer_obj.model_dump(exclude_none=True)
        data_answer['timestamp'] = firestore.SERVER_TIMESTAMP
        
        doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
        doc_ref_answer.set(data_answer)
        answer_id = doc_ref_answer.id
        
        return {
            "prompt_id": prompt_id, 
            "answer_id": answer_id,
            "conversation_id": prmt.conversation_id, 
            "status": "prompt_and_answer_recorded"
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error saat menyimpan data: {str(e)}")

# -----------------------------------------------------------------
# --- ENDPOINT FETCH HISTORY ---
# -----------------------------------------------------------------

@router.get("/history/{conversation_id}", response_model=List[dict])
def get_conversation_history(conversation_id: str, user_data: dict = Depends(get_current_user_data)):
    """Mengambil semua pasangan Prompt dan SmartAnswer untuk conversation_id tertentu."""
    
    # 1. Ambil semua Prompt
    prompts_stream = db.collection(PROMPTS_COLLECTION)\
                       .where('conversation_id', '==', conversation_id)\
                       .order_by('timestamp')\
                       .stream()
    
    prompts_map = {doc.id: doc.to_dict() for doc in prompts_stream}
    prompt_ids = list(prompts_map.keys())

    if not prompt_ids:
        return []

    # 2. Ambil semua Answer (dengan limit 10 untuk 'in', disederhanakan)
    if len(prompt_ids) > 10:
        # Implementasi kueri batch diperlukan di sini jika > 10.
        # Untuk kasus ini, kita hanya akan mengambil 10 prompt pertama
        prompt_ids_to_query = prompt_ids[:10]
    else:
        prompt_ids_to_query = prompt_ids

    answers_stream = db.collection(ANSWERS_COLLECTION)\
                       .where('conversation_id', '==', conversation_id)\
                       .where('prompt_id', 'in', prompt_ids_to_query)\
                       .order_by('timestamp')\
                       .stream()
    
    answers_map = {}
    for doc in answers_stream:
        data = doc.to_dict()
        answers_map[data['prompt_id']] = {"answer_id": doc.id, **data}
    
    # 3. Gabungkan Prompt dan Answer
    history = []
    
    for prompt_id in prompt_ids: # Iterasi melalui semua prompt_ids
        prompt_data = prompts_map[prompt_id]
        answer_data = answers_map.get(prompt_id)
        
        # Tambahkan Prompt
        history.append({
            "type": "prompt",
            "id": prompt_id, 
            "text": prompt_data.get('prompt_text'),
            "timestamp": prompt_data.get('timestamp')
        })
        
        # Tambahkan Answer (termasuk struktur data kompleks jika ada)
        if answer_data:
            # Hapus metadata relasi agar data lebih bersih untuk frontend
            del answer_data['conversation_id']
            del answer_data['prompt_id']
            
            history.append({
                "type": "answer",
                "id": answer_data['answer_id'],
                "content": answer_data, # <-- KONTEN JAWABAN LENGKAP DI SINI
                "timestamp": answer_data.get('timestamp')
            })
            
    # Urutkan ulang (Pengamanan)
    history.sort(key=lambda x: x['timestamp'])

    return history