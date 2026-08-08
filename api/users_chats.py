# --- IMPOR LAMA ANDA ---
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from firebase_config import db # Klien Firestore
from .auth import get_current_user_data # Dependency Auth
from typing import List, Optional, Dict, Any
from google.cloud import firestore # Untuk SERVER_TIMESTAMP
from qdrant_client import QdrantClient, models
import time
import re # Anda mungkin masih membutuhkannya
import asyncio

# --- ✨ IMPOR BARU UNTUK RAG + OPENAI ---
import openai
import chromadb
import os
from dotenv import load_dotenv

# Muat file .env secara dinamis
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = "islamic_ai_vs" # Sesuai info Anda

try:
    # Hubungkan ke Qdrant di VPS Anda secara dinamis
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    
    # Cek apakah koleksi ada
    qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
    
    print(f"Berhasil terhubung ke Qdrant di {QDRANT_URL}")
    print(f"Menggunakan koleksi: {QDRANT_COLLECTION_NAME}")

except Exception as e:
    print(f"CRITICAL: Gagal terhubung ke Qdrant. {e}")
    print("Pastikan Qdrant server berjalan dan koleksi sudah ada.")

# -----------------------------------------------------------------
# --- ✨ 1. KLIEN OPENAI & VECTOR DB (KONEKSI "OTAK") ---
# -----------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY tidak ditemukan di environment variables.")

# Gunakan AsyncClient untuk FastAPI dengan API key dari .env
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
# Hubungkan ke database vektor PERSISTENT (folder) yang Anda buat
try:
    client_chroma = chromadb.PersistentClient(path="./vector_db")
    collection_quran = client_chroma.get_collection(name="quran")
    collection_hadith = client_chroma.get_collection(name="hadith")
    print(f"Berhasil terhubung ke ChromaDB di ./vector_db (Quran: {collection_quran.count()}, Hadits: {collection_hadith.count()})")
    CHROMA_ENABLED = True
except Exception as e:
    print(f"WARNING: Gagal terhubung ke ChromaDB. {e}. Retrieval Chroma dinonaktifkan.")
    CHROMA_ENABLED = False
    # Anda bisa exit(1) di sini jika koneksi DB ini wajib

QDRANT_ENABLED = True

    

# -----------------------------------------------------------------
# --- 2. MODEL PYDANTIC (Sama seperti sebelumnya) ---
# -----------------------------------------------------------------

router = APIRouter(tags=["Users, Conversations & Messages (Prompt/Answer)"])

# Root of /api/v1 for quick health check
@router.get("")
def api_root():
    return {"message": "API v1 OK"}

# (Model UserProfile, Conversation, Prompt... tetap sama)
class UserProfile(BaseModel):
    username: str
    photo_url: Optional[str] = None
class Conversation(BaseModel):
    user_id: str
    title: str
class MessageHistory(BaseModel):
    role: str = Field(..., description="Role 'user' atau 'assistant'.")
    content: str = Field(..., description="Teks konten dari pesan.")

class Prompt(BaseModel):
    conversation_id: str
    sender_uid: str
    prompt_text: str
    # 櫨 TAMBAHKAN BARIS INI
    history: Optional[List[MessageHistory]] = Field(default_factory=list, description="Riwayat obrolan sebelumnya untuk konteks.")

# --- 櫨 MODEL BARU UNTUK PROMPT TAMU (GUEST) ---
class GuestPrompt(BaseModel):
    prompt_text: str = Field(..., description="Teks prompt dari pengguna anonim.")
    # 櫨 TAMBAHKAN BARIS INI
    history: Optional[List[MessageHistory]] = Field(default_factory=list, description="Riwayat obrolan sebelumnya untuk konteks.")

# (Sub-model Letter, Quran, Hadith... tetap sama)
class LetterContent(BaseModel):
    letter_type: str
    recipient: str
    sender: str
    date: str
    salutation: str
    body_paragraphs: List[str]
    closing: str
class QuranicContent(BaseModel):
    surah_name: str = Field(..., description="Nama Surah (e.g., Al-Baqarah).")
    surah_number: int = Field(..., description="Nomor Surah.")
    ayah_number: str = Field(..., description="Nomor Ayat.")
    arabic_text: Optional[str] = Field(None, description="Teks Ayat dalam Bahasa Arab.")
    translation: str = Field(..., description="Terjemahan Ayat.")
    tafsir_summary: Optional[str] = Field(None, description="Ringkasan tafsir atau konteks ayat.")
class HadithContent(BaseModel):
    book: str
    number: str
    narrator: Optional[str] = None
    arabic_text: Optional[str] = None
    translation: str
    details: Optional[str] = None

# (Model SmartAnswer... tetap sama)
class SmartAnswer(BaseModel):
    conversation_id: str
    prompt_id: str 
    summary_text: str = Field(..., description="Ringkasan atau teks utama dari jawaban AI.")
    letter_example: Optional[LetterContent] = None
    hadith_example: Optional[HadithContent] = None
    quran_example: Optional[QuranicContent] = None 
    sources: List[str] = Field(default_factory=list, description="Daftar sumber yang digunakan.")

# (Model SmartAnswerFormatV2... tetap sama)
class SmartAnswerFormatV2(BaseModel):
    introductory_text: str = Field(description="Sapaan/Intro yang ramah dan komunikatif dalam 1-2 kalimat. Tetap gunakan persona 'Al-Faqih'.")
    long_form_content: Optional[str] = Field(None, description="Teks jawaban yang panjang dan terstruktur, misal isi kultum, artikel, atau diskusi mendalam. Gunakan Markdown untuk format (Heading, List, dll.).")
    quran_examples: List[QuranicContent] = Field(default_factory=list, description="DAFTAR (maks 3) Ayat Quran yang paling relevan.")
    hadith_examples: List[HadithContent] = Field(default_factory=list, description="DAFTAR (maks 3) Hadits yang paling relevan.")
    letter_example: Optional[LetterContent] = Field(None, description="Objek surat resmi/pribadi JIKA prompt memintanya.")
    sources: List[str] = Field(default_factory=list, description="Daftar sumber yang digunakan untuk referensi.")


# --- KONSTANTA NAMA KOLEKSI FIREBASE ---
USERS_COLLECTION = "users"
CONVERSATIONS_COLLECTION = "conversations"
PROMPTS_COLLECTION = "prompts"
ANSWERS_COLLECTION = "answers"


# -----------------------------------------------------------------
# --- 3. ENDPOINT USER & CONVERSATIONS (Tidak Berubah) ---
# -----------------------------------------------------------------

@router.post("/users/register", status_code=status.HTTP_201_CREATED)
def register_user_profile(profile: UserProfile, user_data: dict = Depends(get_current_user_data)):
    # ... (kode lama Anda, tidak perlu diubah)
    user_uid = user_data.get('uid')
    try:
        data_to_save = profile.model_dump()
        data_to_save['email'] = user_data.get('email')
        db.collection(USERS_COLLECTION).document(user_uid).set(data_to_save)
        return {"id": user_uid, **data_to_save}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/users/me")
def get_my_profile(user_data: dict = Depends(get_current_user_data)):
    # ... (kode lama Anda, tidak perlu diubah)
    user_uid = user_data.get('uid')
    doc = db.collection(USERS_COLLECTION).document(user_uid).get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail="Profil pengguna belum terdaftar di Firestore.")
    return {"id": doc.id, **doc.to_dict()}

@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(conv: Conversation, user_data: dict = Depends(get_current_user_data)):
    # ... (kode lama Anda, tidak perlu diubah)
    if conv.user_id != user_data.get('uid'):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail="Anda tidak diizinkan membuat obrolan atas nama pengguna lain.")
    try:
        data_for_response = conv.model_dump()
        data_for_db = data_for_response.copy()
        data_for_db['created_at'] = firestore.SERVER_TIMESTAMP
        doc_ref = db.collection(CONVERSATIONS_COLLECTION).document()
        doc_ref.set(data_for_db)
        
        # Kirim balik data yang sudah terisi timestamp (jika mungkin)
        # Untuk kesederhanaan, kita kembalikan data yang dikirim + ID
        data_for_response['created_at'] = time.time() # Kirim epoch time
        return {"conversation_id": doc_ref.id, **data_for_response}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/conversations", response_model=List[dict])
def get_my_conversations(user_data: dict = Depends(get_current_user_data)):
    # ... (kode lama Anda, tidak perlu diubah)
    user_uid = user_data.get('uid')
    conversations = []
    stream = db.collection(CONVERSATIONS_COLLECTION)\
               .where('user_id', '==', user_uid)\
               .order_by('created_at', direction=firestore.Query.DESCENDING)\
               .stream()
    for doc in stream:
        data = doc.to_dict()
        # Konversi Firestore timestamp ke string/epoch untuk JSON
        if 'created_at' in data and hasattr(data['created_at'], 'timestamp'):
             data['created_at'] = data['created_at'].timestamp()
        conversations.append({"conversation_id": doc.id, **data})
    return conversations

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, user_data: dict = Depends(get_current_user_data)):
    """Menghapus percakapan dan semua prompt/jawaban terkait."""
    
    user_uid = user_data.get('uid')
    
    # 1. Verifikasi kepemilikan percakapan
    conv_ref = db.collection(CONVERSATIONS_COLLECTION).document(conversation_id)
    conv_doc = conv_ref.get()
    
    if not conv_doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Percakapan tidak ditemukan.")
    
    if conv_doc.to_dict().get('user_id') != user_uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Anda tidak diizinkan menghapus percakapan ini.")

    try:
        # 2. Hapus semua prompt dan jawaban terkait (Batch delete)
        batch = db.batch()

        # Hapus semua prompts dalam percakapan
        prompts_query = db.collection(PROMPTS_COLLECTION).where('conversation_id', '==', conversation_id).stream()
        for doc in prompts_query:
            batch.delete(doc.reference)
        
        # Hapus semua answers dalam percakapan
        answers_query = db.collection(ANSWERS_COLLECTION).where('conversation_id', '==', conversation_id).stream()
        for doc in answers_query:
            batch.delete(doc.reference)
        
        # 3. Hapus percakapan itu sendiri
        batch.delete(conv_ref)
        
        # 4. Commit batch
        batch.commit()
        
        # Status 204 (No Content) akan otomatis dikirim
        return
        
    except Exception as e:
        print(f"Error saat menghapus percakapan: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gagal menghapus data terkait: {str(e)}")


# -----------------------------------------------------------------
# --- 櫨 4. FUNGSI RAG (RETRIEVAL & GENERATION) BARU ---
# -----------------------------------------------------------------
# users_chats.py (Di dalam fungsi _is_islamic_topic_check, sekitar Baris 526)

async def _is_islamic_topic_check(prompt_text: str, history: List[Dict[str, Any]] = None) -> bool:
    """
    Menggunakan model kecil untuk mengklasifikasikan apakah prompt adalah topik Islam, 
    dengan mempertimbangkan riwayat percakapan secara teliti.
    """
    
    # 1. SIAPKAN RIWAYAT SEBAGAI KONTEKS (Ambil 2-3 percakapan terakhir)
    history_str = ""
    if history:
        # Ambil maksimal 4 item terakhir (2 pasang prompt/answer)
        history_str = "\n--- RIWAYAT (Context) ---\n"
        for hist_item in history[-4:]:
            
            # 櫨 PERBAIKAN: Selalu konversi ke dictionary jika itu model, jika tidak, pastikan itu dictionary
            try:
                # Jika item adalah Pydantic model (MessageHistory), ubah ke dict
                item = hist_item.model_dump()
            except AttributeError:
                # Jika bukan model (misal: sudah dict atau string), gunakan langsung
                item = hist_item
                
            # 櫨 CHECK KRITIS: Lewati item jika bukan dictionary untuk mencegah error 'str' object has no attribute 'get' 櫨
            if not isinstance(item, dict):
                 continue

            # Ambil teks dan role sesuai model MessageHistory (role & content)
            message_text = item.get('content')
            role = item.get('role')
            
            # Check jika field yang dibutuhkan ada dan teksnya adalah string
            if isinstance(message_text, str) and role in ['user', 'assistant']:
                if role == 'user':
                    # Batasi teks riwayat agar tidak terlalu panjang
                    history_str += f"USER SEBELUMNYA: {message_text[:200]}...\n"
                elif role == 'assistant':
                    history_str += f"ASISTEN SEBELUMNYA: {message_text[:200]}...\n"
            
        history_str += "-------------------------\n"
    
    # ... (Baris 2. PERKUAT INSTRUKSI KLASIFIKASI dan sisa fungsi tetap sama)
    system_content = (
        "You are a **WISE and FLEXIBLE** topic reviewer system. Your goal is to be **OPEN-MINDED** but maintain a boundary. "
        "You **MUST** answer only with a single word: **'Yes'** or **'no'**. \n\n"
        
        "### PRIMARY RULE ###\n"
        "Be inclusive. If a question has even a **slight connection** to Islamic values, ethics, history, or can be answered from an Islamic perspective, categorize as **'Yes'**.\n\n"

        "### 'Yes' CRITERIA (Broad Islamic/Ethical Scope) ###\n"
        "Categorize as **'Yes'** if the question involves:\n"
        " - Anything related to religion, spirituality, or faith.\n"
        " - Moral dilemmas, ethics, or general life advice (as these can be addressed via Islamic wisdom).\n"
        " - History, culture, or social issues that intersect with Islamic identity.\n"
        " - Any topic where an Islamic perspective would be meaningful (e.g., 'how to be a good neighbor', 'environmental ethics').\n"
        
        "### 'no' CRITERIA (Zero Islamic Side) ###\n"
        "Categorize as **'no'** ONLY if the topic has **ABSOLUTELY ZERO** connection to Islam, morality, or religion, such as:\n"
        " - Purely technical or scientific questions with no ethical component (e.g., 'how to compile C++ code', 'chemical formula of water').\n"
        " - Purely secular entertainment or sports trivia (e.g., 'who won the 2022 World Cup', 'latest movie cast').\n"
        " - Mundane tasks with no religious/ethical angle (e.g., 'how to fix a leaky faucet')."
    )

    prompt_for_classifier = history_str + "USER SAAT INI: " + prompt_text
    
    try:
        response = await openai_client.chat.completions.create(
            # GANTI MODEL DI SINI:
            model="gpt-4o-mini", # <-- DIGANTI DARI gpt-4o
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt_for_classifier}
            ],
            temperature=0.0,
            # Max tokens tetap 3, karena kita hanya ingin jawaban 'YA' atau 'TIDAK'
            max_tokens=3, 
            top_p=1
        )
        result = response.choices[0].message.content.strip().lower()
        print(f"--- Klasifikasi Topik Cepat (GPT-4o-mini): {result} ---")
        return result == 'yes'
    except Exception as e:
        print(f"Error saat klasifikasi topik: {e}. Menganggap YA (Topik Islam) secara default.")
        return True
async def _generate_rejection_answer(conversation_id: str, prompt_id: str, prompt_text: str) -> dict:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini", # <-- DIGANTI DARI gpt-4o
            messages=[
                {"role": "system", "content": (
                    "You are a wise and friendly Islamic AI assistant. "
                    "The user's prompt is outside the scope of Islam, Quran, Hadith, or Sharia. "
                    "Your task is to generate a polite, warm, and wise rejection. "
                    "CRITICAL: You MUST detect the language of the user's prompt and respond ENTIRELY in that SAME LANGUAGE. "
                    "If the user asks in English, you MUST reject in English. If in Indonesian, reject in Indonesian. "
                )},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        rejection_text = response.choices[0].message.content.strip()
        
        return {
            "conversation_id": conversation_id, 
            "prompt_id": prompt_id,
            "summary_text": rejection_text,
            "sources": ["Topic Guardrail"],
            "quran_examples": [], "hadith_examples": [], "letter_example": None,
            "introductory_text": rejection_text,
            "long_form_content": None
        }
    except Exception as e:
        print(f"Error generating rejection: {e}")
        return {
            "conversation_id": conversation_id, 
            "prompt_id": prompt_id,
            "summary_text": "Maaf, pertanyaan ini di luar fokus utama Huffadz.",
            "sources": ["Topic Guardrail (Fallback)"],
            "quran_examples": [], "hadith_examples": [], "letter_example": None,
            "introductory_text": "Mohon maaf, sepertinya topik ini di luar jangkauan diskusi Islami saya.",
            "long_form_content": None
        }

async def _generate_hypothetical_answer(prompt_text: str) -> str:
    """
    Fungsi BARU (HyDE): Mengambil prompt pengguna dan menghasilkan draf
    jawaban/tafsir hipotetis untuk digunakan sebagai vektor pencarian.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini", # <-- DIGANTI DARI gpt-4o
            messages=[
                {"role": "system", "content": (
                    "You are a brilliant expert in tafsir and hadith. "
                    "A user will ask something. Your task is to write a concise general answer (one paragraph) "
                    "to their question, as if you are explaining the concept. "
                    "Just explain the concept. "
                    "IMPORTANT: You MUST write the answer in the SAME LANGUAGE as the user's question."
                )},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,
            max_tokens=75
        )
        hypothetical_answer = response.choices[0].message.content.strip()
        print(f"--- Jawaban Hipotetis (HyDE) Dibuat (GPT-4o-mini): '{hypothetical_answer}' ---")
        return hypothetical_answer
    except Exception as e:
        print(f"Error saat generasi HyDE: {e}")
        return prompt_text


def _parse_qdrant_filter(prompt_text: str) -> Optional[models.Filter]:
    """
    Fungsi helper untuk membuat filter Qdrant 
    untuk pencarian eksak (misal: "Baqarah 255").
    """
    # Mencari pola: (Surat) [Nama Surah/Latin] (Ayat|:) [Nomor Ayat]
    # Kami akan mencari 'surah_latin' dan 'ayah' di metadata Qdrant
    match = re.search(r"(?:surat\s)?([\w\-]+)\s*(?:ayat|:)\s*(\d+)", prompt_text, re.IGNORECASE)
    
    if match:
        surah_name_partial = match.group(1).replace('-', ' ').strip().lower() 
        ayah_number = match.group(2) # Qdrant menyimpan 'ayah' sebagai string
        
        print(f"--- Kueri Filter Metadata Qdrant Terdeteksi: Surah {surah_name_partial}, Ayat {ayah_number} ---")
        
        return models.Filter(
            must=[
                # Pencarian yang lebih fleksibel pada 'surah_latin'
                models.FieldCondition(
                    key="surah_latin",
                    match=models.MatchText(text=surah_name_partial)
                ),
                models.FieldCondition(
                    key="ayah",
                    match=models.MatchValue(value=ayah_number)
                )
            ]
        )
    return None

def _parse_chroma_filter(prompt_text: str) -> Optional[Dict[str, Any]]:
    """
    Fungsi helper BARU untuk membuat filter ChromaDB 
    untuk pencarian eksak (misal: "Baqarah 255").
    """
    # Mencari pola: (Surat) [Nama Surah] (Ayat|:) [Nomor Ayat]
    match = re.search(r"(?:surat\s)?([\w\-]+)\s*(?:ayat|:)\s*(\d+)", prompt_text, re.IGNORECASE)
    
    if match:
        surah_name = match.group(1).replace('-', ' ').strip()
        ayah_number = int(match.group(2))
        
        print(f"--- Kueri Filter Metadata Terdeteksi: Surah {surah_name}, Ayat {ayah_number} ---")
        
        # Filter ChromaDB menggunakan sintaks Mongo-like
        return {
            "$and": [
    {"surah": {"$text_contains": surah_name}}, # Cari nama surah yang mengandung teks
                {"ayah_number": {"$eq": ayah_number}},
            ]
        }
    return None

async def retrieve_relevant_context(prompt_text: str) -> List[str]:
    """
    Fungsi retrieval yang digabung (Qdrant + Chroma) dengan HyDE secara paralel.
    Sekarang mendukung flag QDRANT_ENABLED / CHROMA_ENABLED.
    """
    contexts = []
    
    try:
        # --- TAHAP 1: FILTER METADATA QDRANT (Pencarian Eksak) ---
        if QDRANT_ENABLED:
            qdrant_filter = _parse_qdrant_filter(prompt_text)
            
            if qdrant_filter:
                try:
                    # Run blocking search in a threadpool
                    results_exact = await asyncio.to_thread(
                        qdrant_client.search,
                        collection_name=QDRANT_COLLECTION_NAME, # Pastikan ini "islamic_ai_vc"
                        query_vector=None, 
                        query_filter=qdrant_filter,
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                    )
                    
                    if results_exact:
                        payload = results_exact[0].payload
                        ctx_str = f"[Konteks Quran (Eksak Qdrant)] {payload.get('source', 'N/A')}: {payload.get('page_content', 'N/A')}"
                        contexts.append(ctx_str)
                        print(f"Total {len(contexts)} konteks (Eksak Qdrant) ditemukan. Melewati pencarian semantik.")
                        return contexts # Langsung kembalikan jika ditemukan eksak
                except Exception as qe:
                    print(f"WARNING: Gagal melakukan filter Qdrant: {qe}")
            
        # --- TAHAP 2: TRANSFORMASI HyDE (Pencarian Semantik) ---
        hypothetical_answer = await _generate_hypothetical_answer(prompt_text)
        
        # --- TAHAP 3: PENCARIAN VEKTOR (Semantik) ---
        embedding_response = await openai_client.embeddings.create(
            input=hypothetical_answer, 
            model=EMBEDDING_MODEL
        )
        prompt_vector = embedding_response.data[0].embedding
        
        all_hits = []

        # 3a. Definisikan fungsi pencarian async pembungkus I/O sinkron
        async def search_qdrant():
            if not QDRANT_ENABLED:
                return []
            try:
                res = await asyncio.to_thread(
                    qdrant_client.search,
                    collection_name=QDRANT_COLLECTION_NAME,
                    query_vector=prompt_vector,
                    limit=5,
                    with_payload=True,
                    score_threshold=0.5 
                )
                return res
            except Exception as qe:
                print(f"WARNING: Gagal melakukan search Qdrant: {qe}")
                return []

        async def search_chroma_quran():
            if not CHROMA_ENABLED:
                return {}
            try:
                res = await asyncio.to_thread(
                    collection_quran.query,
                    query_embeddings=[prompt_vector],
                    n_results=3, 
                    include=['metadatas', 'documents', 'distances']
                )
                return res
            except Exception as ce:
                print(f"WARNING: Gagal melakukan query Chroma Quran: {ce}")
                return {}

        async def search_chroma_hadith():
            if not CHROMA_ENABLED:
                return {}
            try:
                res = await asyncio.to_thread(
                    collection_hadith.query,
                    query_embeddings=[prompt_vector],
                    n_results=3, 
                    include=['metadatas', 'documents', 'distances']
                )
                return res
            except Exception as ce:
                print(f"WARNING: Gagal melakukan query Chroma Hadits: {ce}")
                return {}

        # Eksekusi ketiga pencarian secara PARALEL
        results_qdrant, results_quran_chroma, results_hadith_chroma = await asyncio.gather(
            search_qdrant(),
            search_chroma_quran(),
            search_chroma_hadith()
        )

        # Proses hasil Qdrant
        for hit in results_qdrant:
            payload = hit.payload
            all_hits.append({
                "score": hit.score,
                "source_type": payload.get('data_type', 'N/A'),
                "source_info": payload.get('source', 'N/A'),
                "text": payload.get('page_content', 'N/A'),
            })

        # Proses hasil Quran Chroma
        if results_quran_chroma.get('metadatas'):
            for meta, doc, dist in zip(results_quran_chroma['metadatas'][0], results_quran_chroma['documents'][0], results_quran_chroma['distances'][0]):
                arabic_text = meta.get('arabic_text', '[Teks Arab tidak tersedia]')
                translation_text = meta.get('original_text', doc)
                combined_text = f"Teks Arab: {arabic_text}\nTerjemahan: {translation_text}"
                all_hits.append({
                    "score": 1 - dist, 
                    "source_type": 'Quran (Chroma)',
                    "source_info": f"QS {meta.get('surah', 'N/A')}:{meta.get('ayah_number', 'N/A')}",
                    "text": combined_text,
                })
                
        # Proses hasil Hadith Chroma
        if results_hadith_chroma.get('metadatas'):
            for meta, doc, dist in zip(results_hadith_chroma['metadatas'][0], results_hadith_chroma['documents'][0], results_hadith_chroma['distances'][0]):
                arabic_text = meta.get('Arab', '[Teks Arab tidak tersedia]')
                translation_text = meta.get('original_text', doc)
                combined_text = f"Teks Arab: {arabic_text}\nTerjemahan: {translation_text}"
                all_hits.append({
                    "score": 1 - dist,
                    "source_type": 'Hadith (Chroma)',
                    "source_info": f"HR {meta.get('Perawi', 'N/A')}",
                    "text": combined_text,
                })
        
        # --- TAHAP 4: FORMAT HASIL & SELEKSI AKHIR ---
        all_hits.sort(key=lambda x: x['score'], reverse=True)
        
        for hit in all_hits[:7]: 
            ctx_str = f"[Konteks {hit['source_type']} - {hit['source_info']} (Skor: {hit['score']:.2f})]\n{hit['text']}"
            if ctx_str not in contexts:
                contexts.append(ctx_str)
        
        print(f"Total {len(contexts)} konteks (Gabungan Qdrant/Chroma) ditemukan.")
        return contexts

    except Exception as e:
        print(f"Error saat retrieval gabungan Qdrant/Chroma: {e}")
        return []

# Pastikan fungsi ini mengembalikan DICT, bukan objek Pydantic
async def generate_smart_answer(conversation_id: str, prompt_id: str, prompt_text: str, history: List[MessageHistory]) -> dict:
    
    relevant_contexts = await retrieve_relevant_context(prompt_text)
    openai_history_messages = [msg.model_dump() for msg in history]

        # 2. Ambil beberapa pesan terakhir (misal, 6 pesan: 3 pasang user/bot)
        # Ini penting agar tidak melebihi token limit.
    recent_history = openai_history_messages[-6:] # Ambil 6 terakhir

    if relevant_contexts:
        print("--- RAG SUKSES: Menggunakan Konteks Lokal ---")
        context_string = "\n\n---\n\n".join(relevant_contexts)
        system_prompt_rag = f"""
        You are an intelligent, factual Islamic AI Assistant and expert in Al-Qur'an and Hadith.
        YOUR ROLE: Act as a **consultant/discussion partner** who is relaxed, deep, and has a bit of wise/friendly humor. Use communicative greetings/language styles, as if you are chatting.

        ### MULTILINGUAL & PERSONA INSTRUCTIONS (CRITICAL):
        1. **Language Detection:** You MUST detect the language of the user's prompt.
        2. **Language Consistency:** You MUST respond ENTIRELY in the SAME LANGUAGE as the user's prompt (e.g., English for English, Arabic for Arabic, Indonesian for Indonesian).
        
        ### ANSWER GENERATION INSTRUCTIONS:
        1.  **Intent Analysis:** Determine if the user is asking for a **deep discussion/explanation** or a **specific document/content format** (such as Kultum, Article, Letter, or Speech).
            * If requesting a **specific document** (e.g., "Create a kultum about patience"), focus your output on `long_form_content` using the appropriate format.
            * If requesting an **explanation/discussion** (e.g., "What is the wisdom of patience?"), focus your output on `long_form_content` as an informative structured article/essay.
        2.  **Discursive Style & Closing:** The `introductory_text` contains a warm response like confirmation. The `long_form_content` is the core answer. At the end of `long_form_content`, include a **rhetorical question**, an engaging affirmation, or an **invitation to reflect/discuss** relevant derivative topics.
        3.  **Continuous Discussion:** You will receive previous chat history. Ensure your answer connects and remembers that context.
        4.  **Mandatory Context (Multi-Evidence):** Your main answer (`long_form_content`) MUST summarize and integrate **ALL relevant evidence/context** (Qur'an and Hadith) available below.
        5.  **Specific Citations (Parsing) - MANDATORY COMPLETE:**
            * Select **maximum 3 Quranic Verses** that are most primary/clear to be **PARSED** into `quran_examples` objects.
            * Select **maximum 3 Hadiths** that are most primary/clear to be **PARSED** into `hadith_examples` objects.
            * **IMPORTANT (MANDATORY):** When parsing `quran_examples` and `hadith_examples`, **ENSURE** you copy the `arabic_text` from the context into the `arabic_text` field in JSON. The context will provide it (usually starting with "Teks Arab: ..."). **NEVER** leave the `arabic_text` field empty/null if Arabic text is available in the context. This is a top priority.
            * Use `letter_example` ONLY IF the user specifically requests a *formal/personal letter*.
        6.  **Persona-based Rejection:** If the context is completely irrelevant, respond in your signature style: "Oops, this is a good question, but my 'fresh evidences' aren't connecting there yet. Maybe we can 'chat' about another topic first that's closer to our references?" (Adapt this to the user's language).
        7.  **Final Format:** Provide your answer in valid JSON format according to this Pydantic schema:

        {SmartAnswerFormatV2.model_json_schema()}

        
        --- CONTEXT FROM DATABASE (QURAN & HADITH) ---
        {context_string}
        --- END OF CONTEXT ---
        """
        
        # 1. Ubah history Pydantic ke format dict standar OpenAI
        

        # 3. Bangun daftar pesan LENGKAP
        messages_for_api = [
            {"role": "system", "content": system_prompt_rag},
            # Masukkan riwayat SEBELUM prompt terakhir 
            *recent_history,
            # Masukkan prompt pengguna SAAT INI
            {"role": "user", "content": prompt_text}
        ]

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                response_format={ "type": "json_object" }, 
                messages=messages_for_api,
                temperature=0.1
            )
            
            json_response = response.choices[0].message.content
            # Validasi data, tapi kita akan gunakan sebagai dict
            parsed_data = SmartAnswerFormatV2.model_validate_json(json_response)
            
            # --- 櫨 PERBAIKAN UTAMA: Konversi ke dict SEBELUM return ---
            return_dict = parsed_data.model_dump(exclude_none=True) # Exclude_none=True lebih aman
            
            # Buat summary_text gabungan untuk frontend
            return_dict['summary_text'] = f"**{parsed_data.introductory_text}**\n\n{parsed_data.long_form_content or ''}"
            
            # Tambahkan ID (meskipun akan ditimpa nanti, ini untuk konsistensi)
            return_dict['conversation_id'] = conversation_id
            return_dict['prompt_id'] = prompt_id

            # Tambahkan fallback V1 untuk jaga-jaga (jika frontend masih pakai)
            # Pastikan ini juga dict
            return_dict['quran_example'] = parsed_data.quran_examples[0].model_dump() if parsed_data.quran_examples else None
            return_dict['hadith_example'] = parsed_data.hadith_examples[0].model_dump() if parsed_data.hadith_examples else None
            
            # Hapus return SmartAnswer(...) yang lama
            return return_dict # <-- HARUS MENGEMBALIKAN DICT

        except Exception as e:
            print(f"Error saat RAG Lapis 1: {e}")
            # Kembalikan dict
            return {
                "conversation_id": conversation_id, "prompt_id": prompt_id,
                "summary_text": f"Terjadi kesalahan saat memproses jawaban RAG: {str(e)}. (Konteks: {context_string[:100]}...)",
                "sources": ["OpenAI/RAG Error"],
                "quran_examples": [], "hadith_examples": [], "letter_example": None,
                "introductory_text": "Error", "long_form_content": str(e)
            }

    else:
        print("--- RAG GAGAL: Tidak ada konteks ditemukan. Tidak ada fallback. ---")
        # Kembalikan dict
        return {
            "conversation_id": conversation_id, 
            "prompt_id": prompt_id,
            "summary_text": "Maaf, saya tidak dapat menemukan informasi yang relevan dengan pertanyaan Anda di dalam database saya.",
            "sources": ["Local Database (Not Found)"],
            "quran_examples": [], "hadith_examples": [], "letter_example": None,
            "introductory_text": "Maaf", "long_form_content": "Tidak ada data."
        }
# --- 櫨 櫨 櫨 AKHIR PERBAIKAN 櫨 櫨 櫨 ---


def _bg_save_document(collection: str, doc_id: str, data: dict):
    """Fungsi pembantu untuk menulis dokumen ke Firestore di background thread."""
    try:
        db.collection(collection).document(doc_id).set(data)
    except Exception as e:
        print(f"Error menulis ke Firestore di background: {e}")

# --- ✨ 5. ENDPOINT PROMPT (TERMODIFIKASI) ---
# Fungsi ini sekarang akan menerima DICT dari generate_smart_answer
@router.post("/prompts", status_code=status.HTTP_201_CREATED)
async def post_prompt(prmt: Prompt, background_tasks: BackgroundTasks, user_data: dict = Depends(get_current_user_data)):
    
    # ... (Verifikasi sender_uid tetap di sini)

    try:
        # 0. ✨ LANGKAH BARU: GUARDRAIL KLASIFIKASI TOPIK (Menghemat RAG + GPT-4o) ✨
        is_relevant = await _is_islamic_topic_check(prmt.prompt_text, history=prmt.history)
        
        if not is_relevant:
            # Respon Penolakan Cepat dan Hangat (sekarang dinamis berdasarkan bahasa pengguna)
            answer_dict_fast_reject = await _generate_rejection_answer(
                conversation_id=prmt.conversation_id, 
                prompt_id="temp_reject",
                prompt_text=prmt.prompt_text
            )
            
            # Kita tetap harus menyimpan Prompt (langkah 1)
            data_prompt = prmt.model_dump()
            data_prompt.pop('history', None) 
            data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP 
            
            # Generate doc ID lokal (cepat & offline)
            doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
            prompt_id = doc_ref_prompt.id
            
            # Kita juga harus menyimpan Answer penolakan (sebagai 'answer')
            # Gunakan ID prompt yang sudah dibuat
            answer_dict_fast_reject['prompt_id'] = prompt_id
            data_answer_db = answer_dict_fast_reject.copy() 
            data_answer_db['timestamp'] = firestore.SERVER_TIMESTAMP
            
            doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
            answer_id = doc_ref_answer.id
            
            # Simpan ke Firestore di latar belakang (Background Tasks)
            background_tasks.add_task(_bg_save_document, PROMPTS_COLLECTION, prompt_id, data_prompt)
            background_tasks.add_task(_bg_save_document, ANSWERS_COLLECTION, answer_id, data_answer_db)

            # Format output yang konsisten
            answer_dict_fast_reject['answer_id'] = answer_id
            answer_dict_fast_reject['timestamp'] = time.time()
            data_prompt['prompt_id'] = prompt_id
            data_prompt['timestamp'] = time.time() # Perlu dikonversi untuk konsistensi
            
            return {
                "prompt": data_prompt,
                "answer": answer_dict_fast_reject
            }

        # 1. Simpan Prompt
        data_prompt = prmt.model_dump()
        data_prompt.pop('history', None) 
        data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP 
        
        doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
        prompt_id = doc_ref_prompt.id
        
        data_prompt['prompt_id'] = prompt_id
        data_prompt['timestamp'] = time.time() # Konversi ke float/epoch

        # 2. Proses Jawaban (Panggil fungsi ASYNC - HANYA JIKA RELEVAN)
        smart_answer_dict = await generate_smart_answer(
            conversation_id=prmt.conversation_id, 
            prompt_id=prompt_id, 
            prompt_text=prmt.prompt_text,
            history=prmt.history
        )
        
        # 3. Simpan SmartAnswer ke Firestore
        data_answer_db = smart_answer_dict.copy() 
        data_answer_db['timestamp'] = firestore.SERVER_TIMESTAMP
        
        doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
        answer_id = doc_ref_answer.id
        
        # Simpan ke Firestore di latar belakang (Background Tasks)
        background_tasks.add_task(_bg_save_document, PROMPTS_COLLECTION, prompt_id, data_prompt)
        background_tasks.add_task(_bg_save_document, ANSWERS_COLLECTION, answer_id, data_answer_db)
        
        # 4. ✨ PERUBAHAN: Kembalikan data lengkap, bukan status
        smart_answer_dict['answer_id'] = answer_id
        smart_answer_dict['timestamp'] = time.time()
        
        return {
            "prompt": data_prompt,
            "answer": smart_answer_dict
        }
        
    except Exception as e:
        # Ini adalah tempat error Anda ditangkap
        print(f"Detail Error di /prompts: {e}") # Tambahkan log
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error saat menyimpan data: {str(e)}")

# --- 櫨 ENDPOINT BARU UNTUK MODE TAMU (GUEST) ---
# users_chats.py (mulai sekitar baris 705)
@router.post("/prompts/guest", status_code=status.HTTP_201_CREATED)
async def post_prompt_guest(prmt: GuestPrompt):
    
    try:
        guest_conv_id = "guest_conversation"
        guest_prompt_id = f"guest_prompt_{int(time.time())}"
        
        # 0. 櫨 LANGKAH BARU: GUARDRAIL KLASIFIKASI TOPIK 櫨
        is_relevant = await _is_islamic_topic_check(prmt.prompt_text,history=prmt.history)
        
        if not is_relevant:
            smart_answer_dict = await _generate_rejection_answer(
                conversation_id=guest_conv_id, 
                prompt_id=guest_prompt_id,
                prompt_text=prmt.prompt_text
            )
        else:
             # 2. Proses Jawaban (Panggil fungsi ASYNC - HANYA JIKA RELEVAN)
            smart_answer_dict = await generate_smart_answer(
                conversation_id=guest_conv_id, 
                prompt_id=guest_prompt_id, 
                prompt_text=prmt.prompt_text,
                history=prmt.history 
            )

        # 3. Kembalikan data lengkap (Langkah ini sama untuk penolakan dan jawaban penuh)
        smart_answer_dict['answer_id'] = f"guest_answer_{int(time.time())}"
        smart_answer_dict['timestamp'] = time.time()
        
        data_prompt = {
            "conversation_id": guest_conv_id,
            "sender_uid": "guest",
            "prompt_text": prmt.prompt_text,
            "prompt_id": guest_prompt_id,
            "timestamp": time.time() - 1
        }

        return {
            "prompt": data_prompt,
            "answer": smart_answer_dict
        }
        
    except Exception as e:
        print(f"Error di endpoint /prompts/guest: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error saat memproses jawaban guest: {str(e)}")

# -----------------------------------------------------------------
# --- 6. ENDPOINT HISTORY (Tidak Berubah) ---
# -----------------------------------------------------------------

@router.get("/history/{conversation_id}", response_model=List[dict])
def get_conversation_history(conversation_id: str, user_data: dict = Depends(get_current_user_data)):
    # ... (kode lama Anda, tidak perlu diubah)
    # 1. Ambil semua Prompt
    prompts_stream = db.collection(PROMPTS_COLLECTION)\
                       .where('conversation_id', '==', conversation_id)\
                       .order_by('timestamp')\
                       .stream()
    
    prompts_map = {doc.id: doc.to_dict() for doc in prompts_stream}
    prompt_ids = list(prompts_map.keys())

    if not prompt_ids:
        return []

    # 2. Ambil semua Answer 
    answers_stream = db.collection(ANSWERS_COLLECTION)\
                       .where('conversation_id', '==', conversation_id)\
                       .order_by('timestamp')\
                       .stream()
    
    answers_map = {}
    for doc in answers_stream:
        data = doc.to_dict()
        # Pastikan data punya 'prompt_id' sebelum mapping
        if 'prompt_id' in data:
            answers_map[data['prompt_id']] = {"answer_id": doc.id, **data}
        
    # 3. Gabungkan Prompt dan Answer
    history = []
    
    for prompt_id in prompt_ids: 
        prompt_data = prompts_map.get(prompt_id)
        answer_data = answers_map.get(prompt_id) 
        
        if prompt_data:
            history.append({
                "type": "prompt",
                "id": prompt_id, 
                "text": prompt_data.get('prompt_text'),
                "timestamp": prompt_data.get('timestamp')
            })
        
        if answer_data:
            # Bersihkan metadata sebelum dikirim ke frontend
            answer_id = answer_data.pop('answer_id', 'unknown')
            answer_data.pop('conversation_id', None)
            answer_data.pop('prompt_id', None)
            
            history.append({
                "type": "answer",
                "id": answer_id,
                "content": answer_data, # KONTEN JAWABAN LENGKAP
                "timestamp": answer_data.get('timestamp')
            })
            
    # Urutkan ulang (Pengamanan)
    # Handle jika timestamp belum di-resolve oleh server
    history.sort(key=lambda x: x.get('timestamp') or 0 if isinstance(x.get('timestamp'), (int, float)) else (x.get('timestamp').timestamp() if x.get('timestamp') else 0))

    return history