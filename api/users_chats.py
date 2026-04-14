# --- IMPOR LAMA ANDA ---
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from firebase_config import db # Klien Firestore
from .auth import get_current_user_data # Dependency Auth
from typing import List, Optional, Dict, Any
from google.cloud import firestore # Untuk SERVER_TIMESTAMP
from qdrant_client import QdrantClient, models
from typing import List, Optional, Dict, Any
import time
import re # Anda mungkin masih membutuhkannya

# --- 櫨 IMPOR BARU UNTUK RAG + OPENAI ---
import openai
import chromadb
import os

QDRANT_URL = "http://194.233.85.152:6333"
QDRANT_API_KEY = ""
QDRANT_COLLECTION_NAME = "islamic_ai_vs" # Sesuai info Anda

try:
    # Hubungkan ke Qdrant di VPS Anda
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    
    # Cek apakah koleksi ada
    qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
    
    print(f"Berhasil terhubung ke Qdrant di {QDRANT_URL}")
    print(f"Menggunakan koleksi: {QDRANT_COLLECTION_NAME}")

except Exception as e:
    print(f"CRITICAL: Gagal terhubung ke Qdrant. {e}")
    print("Pastikan Qdrant server berjalan dan koleksi sudah ada.")

# -----------------------------------------------------------------
# --- 櫨 1. KLIEN OPENAI & VECTOR DB (KONEKSI "OTAK") ---
# -----------------------------------------------------------------

# Pastikan Anda mengatur Environment Variable ini di server Anda
# JANGAN hardcode API key di sini.

    # Di produksi, Anda mungkin ingin ini menghentikan startup server
    # exit(1) 

# Gunakan AsyncClient untuk FastAPI
openai_client = openai.AsyncOpenAI(api_key="")
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
        "Anda adalah sistem peninjau topik yang **CERMAT, TELITI, dan BIJAK** yang beroperasi dalam mode biner. "
        "Anda **HARUS** menjawab hanya dengan kata tunggal: **'YA'** atau **'TIDAK'**. \n\n"
        
        "### PRIORITAS UTAMA: ANALISIS KONTEKS DAN INTENSI ###\n"
        "1. **Analisis Konteks Berkelanjutan:** Periksa 'RIWAYAT (Context)' terlebih dahulu. JIKA prompt saat ini ('USER SAAT INI') ambigu (misal: 'lanjutkan', 'bagaimana?', 'apa itu') atau sangat singkat, keputusan Anda **WAJIB** mengikuti topik Islami dari RIWAYAT sebelumnya. Ini adalah cara paling bijak untuk menjaga alur percakapan.\n"
        "2. **Toleransi Typo/Bahasa:** Jika prompt memiliki kesalahan ketik (typo) atau tata bahasa yang buruk, namun **INTENSI** pengguna jelas merujuk pada Islam, Fiqih, atau Syariah, anggap sebagai **'YA'**.\n\n"

        "### KRITERIA 'YA' (Topik Islami) ###\n"
        "Kategorikan sebagai 'YA' jika pertanyaan berhubungan dengan **apapun** dari lingkup Islam, termasuk:\n"
        " - Al-Qur'an, Hadits, Sunnah, Tafsir, Sirah, Tauhid, Akhlak, Aqidah, atau Tasawuf.\n"
        " - Fiqih, Syariah, Hukum Islam (Ibadah, Muamalat, Ekonomi Syariah, Waris, Pernikahan).\n"
        " - Sejarah Islam, Kisah Nabi/Sahabat, atau Tokoh Ulama.\n"
        
        "### KRITERIA 'TIDAK' (Di Luar Topik) ###\n"
        "Kategorikan sebagai 'TIDAK' hanya jika topik tersebut secara **DEFINITIF dan MUTLAK** tidak memiliki referensi atau kaitan agama Islam sama sekali, contohnya:\n"
        " - Resep Masakan (kecuali ditanya tentang hukum halal/haram suatu bahan).\n"
        " - Berita umum, Politik praktis, Olahraga, Hiburan, atau Selebriti (tanpa ada kaitan Fiqih/Syariah).\n"
        " - Sains Murni (Fisika, Biologi), Matematika, atau Teknologi (tanpa konteks ajaran Islam tentang alam atau etika)."
    )

    prompt_for_classifier = history_str + "USER SAAT INI: " + prompt_text
    
    try:
        response = await openai_client.chat.completions.create(
            # GANTI MODEL DI SINI:
            model="gpt-4o", # <-- DIGANTI DARI gpt-3.5-turbo
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt_for_classifier}
            ],
            temperature=0.0,
            # Max tokens tetap 3, karena kita hanya ingin jawaban 'YA' atau 'TIDAK'
            max_tokens=3, 
            top_p=1
        )
        result = response.choices[0].message.content.strip().upper()
        print(f"--- Klasifikasi Topik Cepat (GPT-4o): {result} ---")
        return result == 'YA'
    except Exception as e:
        print(f"Error saat klasifikasi topik: {e}. Menganggap YA (Topik Islam) secara default.")
        return True
async def _generate_hypothetical_answer(prompt_text: str) -> str:
    """
    Fungsi BARU (HyDE): Mengambil prompt pengguna dan menghasilkan draf
    jawaban/tafsir hipotetis untuk digunakan sebagai vektor pencarian.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": (
                    "Anda adalah seorang ahli tafsir dan hadits yang cerdas. "
                    "Seorang pengguna akan bertanya sesuatu. Tugas Anda adalah menulis "
                    "jawaban umum yang ringkas (satu paragraf) untuk pertanyaan mereka, "
                    "seolah-olah Anda sedang menjelaskan konsepnya. "
                    "Cukup jelaskan konsepnya saja."    
                )},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,
            max_tokens=75
        )
        hypothetical_answer = response.choices[0].message.content.strip()
        print(f"--- Jawaban Hipotetis (HyDE) Dibuat: '{hypothetical_answer}' ---")
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
# Ganti seluruh fungsi lama Anda dengan yang ini
async def retrieve_relevant_context(prompt_text: str) -> List[str]:
    """
    Fungsi retrieval yang digabung (Qdrant + Chroma) dengan HyDE.
    Sekarang mendukung flag QDRANT_ENABLED / CHROMA_ENABLED.
    """
    contexts = []
    
    try:
        # --- TAHAP 1: FILTER METADATA QDRANT (Pencarian Eksak) ---
        if QDRANT_ENABLED:
            qdrant_filter = _parse_qdrant_filter(prompt_text)
            
            if qdrant_filter:
                try:
                    results_exact = qdrant_client.search(
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
                    print(f"WARNING: Gagal melakukan filter Qdrant (mungkin koleksi salah?): {qe}")
                    # Jangan hentikan proses, biarkan lanjut ke semantik
            
        # --- TAHAP 2: TRANSFORMASI HyDE (Pencarian Semantik) ---
        # (Ini berjalan untuk keduanya, tidak apa-apa)
        hypothetical_answer = await _generate_hypothetical_answer(prompt_text)
        
        # --- TAHAP 3: PENCARIAN VEKTOR (Semantik) ---
        embedding_response = await openai_client.embeddings.create(
            input=hypothetical_answer, 
            model=EMBEDDING_MODEL
        )
        prompt_vector = embedding_response.data[0].embedding
        
        all_hits = []

        # 3a. PENCARIAN DI QDRANT (Hanya jika diaktifkan)
        if QDRANT_ENABLED:
            try:
                results_qdrant = qdrant_client.search(
                    collection_name=QDRANT_COLLECTION_NAME, # Pastikan ini "islamic_ai_vc"
                    query_vector=prompt_vector,
                    limit=5,
                    with_payload=True,
                    score_threshold=0.5 
                )
                
                for hit in results_qdrant:
                    payload = hit.payload
                    all_hits.append({
                        "score": hit.score,
                        "source_type": payload.get('data_type', 'N/A'),
                        "source_info": payload.get('source', 'N/A'),
                        "text": payload.get('page_content', 'N/A'),
                    })
            except Exception as qe:
                 print(f"WARNING: Gagal melakukan search Qdrant (mungkin koleksi salah?): {qe}")
                 # Lanjutkan ke Chroma


        # 3b. PENCARIAN DI CHROMA (Hanya jika diaktifkan)
        if CHROMA_ENABLED:
            results_quran_chroma = collection_quran.query(
                query_embeddings=[prompt_vector],
                n_results=3, 
                include=['metadatas', 'documents', 'distances']
            )
            results_hadith_chroma = collection_hadith.query(
                query_embeddings=[prompt_vector],
                n_results=3, 
                include=['metadatas', 'documents', 'distances']
            )

            # 3b.i. Proses hasil Quran Chroma
            if results_quran_chroma.get('metadatas'):
                for meta, doc, dist in zip(results_quran_chroma['metadatas'][0], results_quran_chroma['documents'][0], results_quran_chroma['distances'][0]):
                    
                    # --- 櫨 PERBAIKAN DI SINI 櫨 ---
                    # Ambil kedua teks dari metadata
                    arabic_text = meta.get('arabic_text', '[Teks Arab tidak tersedia]')
                    translation_text = meta.get('original_text', doc)
                    
                    # Gabungkan keduanya untuk "text" yang akan dikirim ke AI
                    combined_text = f"Teks Arab: {arabic_text}\nTerjemahan: {translation_text}"
                    # --- 櫨 AKHIR PERBAIKAN 櫨 ---

                    all_hits.append({
                        "score": 1 - dist, 
                        "source_type": 'Quran (Chroma)',
                        "source_info": f"QS {meta.get('surah', 'N/A')}:{meta.get('ayah_number', 'N/A')}",
                        "text": combined_text, # <-- Gunakan teks gabungan
                    })
                    
            # 3b.ii. Proses hasil Hadith Chroma
            if results_hadith_chroma.get('metadatas'):
                for meta, doc, dist in zip(results_hadith_chroma['metadatas'][0], results_hadith_chroma['documents'][0], results_hadith_chroma['distances'][0]):
                    
                    # --- 櫨 PERBAIKAN DI SINI (untuk Hadits) 櫨 ---
                    arabic_text = meta.get('Arab', '[Teks Arab tidak tersedia]')
                    translation_text = meta.get('original_text', doc)
                    combined_text = f"Teks Arab: {arabic_text}\nTerjemahan: {translation_text}"
                    # --- 櫨 AKHIR PERBAIKAN 櫨 ---

                    all_hits.append({
                        "score": 1 - dist,
                        "source_type": 'Hadith (Chroma)',
                        "source_info": f"HR {meta.get('Perawi', 'N/A')}",
                        "text": combined_text, # <-- Gunakan teks gabungan
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
        Anda adalah Asisten AI Islami yang cerdas, faktual, dan ahli dalam Al-Qur'an serta Hadits.
        PERAN ANDA: Bertindak sebagai **konsultan/teman diskusi Islami** yang santai, mendalam, dan memiliki sedikit humor yang bijak/ramah. Gunakan sapaan/gaya bahasa yang komunikatif, seolah Anda sedang ngobrol.

        ### INSTRUKSI GENERASI JAWABAN:
        1.  **Analisis Inten:** Tentukan apakah pengguna meminta **diskusi/penjelasan mendalam** atau meminta **format dokumen/konten spesifik** (seperti Kultum, Artikel, Surat, atau Pidato).
            * Jika meminta **dokumen spesifik** (misal: "Buatkan kultum tentang sabar"), fokuskan output Anda pada `long_form_content` dan gunakan format Kultum/Pidato/Surat yang tepat.
            * Jika meminta **penjelasan/diskusi** (misal: "Apa hikmah sabar?"), fokuskan output Anda pada `long_form_content` sebagai artikel/esai informatif yang terstruktur.
        2.  **Gaya Diskusif & Penutup:** Bagian `introductory_text` berisi respon yang hangat seperti konfirmasi. Bagian `long_form_content` adalah inti jawaban. Di akhir `long_form_content`, sertakan **pertanyaan retoris**, penegasan yang *engaging*, atau **ajakan untuk merenung/diskusi** topik turunan yang relevan.
        3.  **Diskusi berkelanjutan**: Anda akan menerima riwayat obrolan sebelumnya. Pastikan jawaban Anda nyambung dan mengingat konteks tersebut.
        4.  **Konteks Wajib (Multi-Dalil):** Jawaban utama Anda (`long_form_content`) HARUS merangkum dan mengintegrasikan **SEMUA dalil/konteks yang relevan** (Qur'an dan Hadits) yang tersedia di bawah.
        5.  **Kutipan Spesifik (Parsing) - WAJIB LENGKAP:**
            * Pilih **maksimal 3 Ayar Al-Qur'an** yang paling utama/jelas untuk di-**PARSE** ke dalam objek `quran_examples`.
            * Pilih **maksimal 3 Hadits** yang paling utama/jelas untuk di-**PARSE** ke dalam objek `hadith_examples`.
            * **PENTING (WAJIB):** Saat mem-parsing `quran_examples` dan `hadith_examples`, **PASTIKAN** Anda menyalin `arabic_text` dari konteks ke dalam bidang `arabic_text` di JSON. Konteks akan menyediakannya (biasanya diawali "Teks Arab: ..."). **JANGAN PERNAH** biarkan bidang `arabic_text` kosong/null jika teks Arab tersedia dalam konteks. Ini adalah prioritas utama.
            * Gunakan `letter_example` HANYA JIKA pengguna secara spesifik meminta *surat resmi* (misal: "Buatkan surat izin tidak masuk sekolah").
        6.  **Penolakan dengan Persona:** Jika konteks tidak relevan sama sekali, balas dengan gaya khas Anda: "Waduh, pertanyaan ini bagus, tapi 'dalil-dalil segar' saya belum nyambung ke sana nih. Mungkin kita coba 'ngobrolin' topik lain dulu yang lebih dekat dengan referensi kita?"
        7.  **Format Akhir:** Berikan jawaban Anda dalam format JSON yang valid sesuai skema Pydantic ini:

        {SmartAnswerFormatV2.model_json_schema()}

        
        --- KONTEKS DARI DATABASE (QURAN & HADITS) ---
        {context_string}
        --- AKHIR KONTEKS ---
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


# --- 櫨 5. ENDPOINT PROMPT (TERMODIFIKASI) ---
# Fungsi ini sekarang akan menerima DICT dari generate_smart_answer
# users_chats.py (mulai sekitar baris 632)
@router.post("/prompts", status_code=status.HTTP_201_CREATED)
async def post_prompt(prmt: Prompt, user_data: dict = Depends(get_current_user_data)):
    
    # ... (Verifikasi sender_uid tetap di sini)

    try:
        # 0. 櫨 LANGKAH BARU: GUARDRAIL KLASIFIKASI TOPIK (Menghemat RAG + GPT-4o) 櫨
        is_relevant = await _is_islamic_topic_check(prmt.prompt_text, history=prmt.history)
        
        if not is_relevant:
            # Respon Penolakan Cepat dan Hangat (tanpa perlu RAG/GPT-4o)
            answer_dict_fast_reject = {
                "conversation_id": prmt.conversation_id, 
                "prompt_id": "temp_reject",
                "summary_text": "Maaf, sepertinya pertanyaan ini di luar fokus utama kami.",
                "sources": ["Topic Guardrail"],
                "quran_examples": [], "hadith_examples": [], "letter_example": None,
                "introductory_text": "Masya Allah, pertanyaan yang menarik! Tapi mohon maaf, sepertinya pertanyaan ini di luar 'bidang fokus' saya, yaitu Al-Qur'an, Hadits, Fiqih, dan Syariah. Bagaimana kalau kita coba 'ngobrolin' hikmah puasa atau kisah sahabat saja dulu?",
                "long_form_content": None
            }
            
            # Kita tetap harus menyimpan Prompt (langkah 1)
            data_prompt = prmt.model_dump()
            data_prompt.pop('history', None) 
            data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP 
            doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
            doc_ref_prompt.set(data_prompt)
            prompt_id = doc_ref_prompt.id
            
            # Kita juga harus menyimpan Answer penolakan (sebagai 'answer')
            # Gunakan ID prompt yang sudah dibuat
            answer_dict_fast_reject['prompt_id'] = prompt_id
            data_answer_db = answer_dict_fast_reject.copy() 
            data_answer_db['timestamp'] = firestore.SERVER_TIMESTAMP
            doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
            doc_ref_answer.set(data_answer_db)

            # Format output yang konsisten
            answer_dict_fast_reject['answer_id'] = doc_ref_answer.id
            answer_dict_fast_reject['timestamp'] = time.time()
            data_prompt['prompt_id'] = prompt_id
            data_prompt['timestamp'] = time.time() # Perlu dikonversi untuk konsistensi
            
            return {
                "prompt": data_prompt,
                "answer": answer_dict_fast_reject
            }

        # 1. Simpan Prompt (sekarang menjadi langkah yang dibagi, disimpan di dalam IF dan ELSE)
        data_prompt = prmt.model_dump()
        # 櫨 Hapus history dari data yang disimpan di DB (tidak perlu disimpan)
        data_prompt.pop('history', None) 
        
        data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP 
        doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
        doc_ref_prompt.set(data_prompt)
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
        
        # ... (Langkah 3 dan 4 Simpan dan Kembalikan seperti semula)
        # ...
        
        # 3. Simpan SmartAnswer ke Firestore
        # 櫨 data_answer_db adalah salinan dari dict
        data_answer_db = smart_answer_dict.copy() 
        data_answer_db['timestamp'] = firestore.SERVER_TIMESTAMP
        
        doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
        doc_ref_answer.set(data_answer_db)
        answer_id = doc_ref_answer.id
        
        # 4. 櫨 PERUBAHAN: Kembalikan data lengkap, bukan status
        # 櫨 Menambahkan key ke dict
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
            smart_answer_dict = {
                "conversation_id": guest_conv_id, 
                "prompt_id": guest_prompt_id,
                "summary_text": "Maaf, sepertinya pertanyaan ini di luar fokus utama kami.",
                "sources": ["Topic Guardrail"],
                "quran_examples": [], "hadith_examples": [], "letter_example": None,
                "introductory_text": "Masya Allah, pertanyaan yang menarik! Tapi mohon maaf, sepertinya pertanyaan ini di luar 'bidang fokus' saya, yaitu Al-Qur'an, Hadits, Fiqih, dan Syariah. Bagaimana kalau kita coba 'ngobrolin' hikmah puasa atau kisah sahabat saja dulu?",
                "long_form_content": None
            }
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