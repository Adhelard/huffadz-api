from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.documents import Document 
from qdrant_client import QdrantClient
from dotenv import load_dotenv 
import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from firebase_config import db # Asumsi file ini ada dan berisi inisialisasi Firestore
from .auth import get_current_user_data # Asumsi fungsi ini ada untuk otentikasi
from typing import List, Optional, Dict, Any
from google.cloud import firestore 
import time

# --- Setup Awal ---
load_dotenv()
router = APIRouter(tags=["Users, Conversations & Messages (Prompt/Answer)"])

RAG_CHAIN = None # Global variable untuk menyimpan RAG Chain LCEL

# -----------------------------------------------------------------
# --- DEFENISI MODEL PYDANTIC ---
# -----------------------------------------------------------------

class RAGQuery(BaseModel):
    query: str = Field(..., description="Pertanyaan yang diajukan ke RAG.")
    conversation_id: str = Field(..., description="ID percakapan saat ini.")

class UserProfile(BaseModel):
    username: str
    photo_url: Optional[str] = None

class Conversation(BaseModel):
    user_id: str
    title: str

class Prompt(BaseModel):
    conversation_id: str 
    sender_uid: str
    prompt_text: str 

# --- SUB-MODEL UNTUK KONTEN JAWABAN KOMPLEKS (Dipertahankan untuk masa depan) ---
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

class SmartAnswer(BaseModel):
    conversation_id: str
    prompt_id: str 
    summary_text: str = Field(..., description="Ringkasan atau teks utama dari jawaban AI.")
    letter_example: Optional[LetterContent] = None
    hadith_example: Optional[HadithContent] = None
    quran_example: Optional[QuranicContent] = None 
    sources: List[str] = Field(default_factory=list, description="Daftar sumber yang digunakan.")

# --- KONSTANTA NAMA KOLEKSI FIREBASE ---
USERS_COLLECTION = "users"
CONVERSATIONS_COLLECTION = "conversations"
PROMPTS_COLLECTION = "prompts"
ANSWERS_COLLECTION = "answers"


# -----------------------------------------------------------------
# --- INISIALISASI RAG CHAIN (LCEL) ---
# -----------------------------------------------------------------

def initialize_rag_chain():
    """Inisialisasi RAG dari dua koleksi (Qur'an dan Hadis) menggunakan LCEL."""
    global RAG_CHAIN

    try:
        # 1. Inisialisasi Klien Qdrant dan Embeddings
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        embeddings = OpenAIEmbeddings(api_key="")

        # 2. Inisialisasi Retriever
        quran_vs = Qdrant(client=client, collection_name="quran_data", embeddings=embeddings)
        quran_retriever = quran_vs.as_retriever(search_kwargs={"k": 3})
        hadith_vs = Qdrant(client=client, collection_name="hadith_data", embeddings=embeddings)
        hadith_retriever = hadith_vs.as_retriever(search_kwargs={"k": 3})

        def combined_retriever(query: str) -> List[Document]:
            """Menggabungkan dokumen dari retriever Qur'an dan Hadis."""
            quran_docs = quran_retriever.invoke(query)
            hadith_docs = hadith_retriever.invoke(query)
            return quran_docs + hadith_docs
        
        # 3. Inisialisasi LLM dan Prompt
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )

        system_prompt = (
            "Anda adalah asisten AI yang ahli dalam Al-Qur'an dan Hadis.\n"
            "Jawab pertanyaan berdasarkan **semua** konteks yang diberikan dari kedua sumber.\n"
            "Jika jawaban tidak ditemukan, katakan bahwa informasi tidak tersedia dalam data Qur'an dan Hadis.\n"
            "Sertakan sumber (Surah dan Ayat atau Perawi/Kitab Hadis) jika memungkinkan.\n"
            "Pastikan jawaban Anda terstruktur dan informatif.\n\n"
            "Konteks:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}"),])

        # 4. LCEL Chain
        generation_chain = (qa_prompt | llm | StrOutputParser())

        RAG_CHAIN = RunnableParallel({
            "context": RunnableLambda(lambda x: combined_retriever(x["input"])),
            "input": RunnablePassthrough(),
        }).assign(
            answer=generation_chain
        )
        
        print("✅ RAG Chain Qur’an + Hadis berhasil diinisialisasi menggunakan LCEL!")

    except Exception as e:
        print(f"❌ Error inisialisasi RAG Chain: {e}")
        RAG_CHAIN = None

# Jalankan inisialisasi saat startup
initialize_rag_chain()

# -----------------------------------------------------------------
# --- ENDPOINT UTAMA (RAG DAN CRUD) ---
# -----------------------------------------------------------------

@router.get("")
def api_root():
    return {"message": "API v1 OK"}

@router.get("/debug/me")
def debug_me(user_data: dict = Depends(get_current_user_data)):
    return user_data

# --- USERS ---
@router.post("/users/register", status_code=status.HTTP_201_CREATED)
def register_user_profile(profile: UserProfile, user_data: dict = Depends(get_current_user_data)):
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
    user_uid = user_data.get('uid')
    doc = db.collection(USERS_COLLECTION).document(user_uid).get()
    if not doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profil pengguna belum terdaftar di Firestore.")
    return {"id": doc.id, **doc.to_dict()}

# --- CONVERSATIONS ---
@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(conv: Conversation, user_data: dict = Depends(get_current_user_data)):
    if conv.user_id != user_data.get('uid'):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Anda tidak diizinkan membuat obrolan atas nama pengguna lain.")
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
# --- ENDPOINT PROMPTS (RAG Live dan Penyimpanan Database) ---
# -----------------------------------------------------------------

@router.post("/prompts", status_code=status.HTTP_201_CREATED)
async def post_prompt(prmt: Prompt, user_data: dict = Depends(get_current_user_data)):
    """Mengirim Prompt baru, memicu RAG, dan menyimpan Prompt + SmartAnswer ke Firestore."""
    sender_uid = user_data.get('uid')
    
    if prmt.sender_uid != sender_uid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ID pengirim prompt tidak cocok dengan pengguna yang terotentikasi.")
    
    if RAG_CHAIN is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG Service belum siap.")

    # 1. Simpan Prompt ke Firestore (Pesan Pengguna)
    try:
        data_prompt = prmt.model_dump()
        data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP 
        
        doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
        doc_ref_prompt.set(data_prompt)
        prompt_id = doc_ref_prompt.id
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error menyimpan prompt: {str(e)}")

    # 2. Proses RAG (Live)
    try:
        rag_response = await RAG_CHAIN.ainvoke({"input": prmt.prompt_text})
        
        answer_text = rag_response["answer"]
        source_documents = rag_response.get("context", []) 
        
        # 3. Format Sumber untuk SmartAnswer
        sources_list = []
        for doc in source_documents:
            metadata_ref = (
                f"Q.S. {doc.metadata.get('surah_latin')} ayat {doc.metadata.get('ayah')}" 
                if doc.metadata.get('surah_latin') else 
                f"Hadis: {doc.metadata.get('Perawi')} - {doc.metadata.get('book')}" if doc.metadata.get('Perawi') else "Sumber Tidak Diketahui"
            )
            sources_list.append(metadata_ref)
        
        # Buat objek SmartAnswer
        smart_answer_obj = SmartAnswer(
            conversation_id=prmt.conversation_id, 
            prompt_id=prompt_id,
            summary_text=answer_text,
            sources=list(set(sources_list)) # Menghilangkan duplikasi
        )
        
    except Exception as e:
        print(f"Error RAG processing: {e}")
        # Tangani error RAG
        smart_answer_obj = SmartAnswer(
            conversation_id=prmt.conversation_id, 
            prompt_id=prompt_id,
            summary_text="Maaf, terjadi kesalahan saat memproses jawaban AI (RAG Error). Silakan coba lagi.",
            sources=["Internal Server Error"]
        )

    # 4. Simpan SmartAnswer (Jawaban AI) ke Firestore
    try:
        data_answer = smart_answer_obj.model_dump(exclude_none=True)
        data_answer['timestamp'] = firestore.SERVER_TIMESTAMP
        
        doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
        doc_ref_answer.set(data_answer)
        answer_id = doc_ref_answer.id
        
        return {
            "prompt_id": prompt_id, 
            "answer_id": answer_id,
            "conversation_id": prmt.conversation_id, 
            "status": "prompt_and_answer_recorded",
            "ai_summary": smart_answer_obj.summary_text # Feedback langsung ke klien
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error menyimpan jawaban: {str(e)}")


# -----------------------------------------------------------------
# --- ENDPOINT HISTORY DAN DEBUG RAG ---
# -----------------------------------------------------------------

@router.post("/rag/chat", status_code=status.HTTP_200_OK)
async def rag_chat(data: RAGQuery, user_data: dict = Depends(get_current_user_data)):
    """Endpoint untuk mengajukan pertanyaan ke RAG berbasis data Quran/Hadis (TESTING SAJA)."""
    if RAG_CHAIN is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG Service belum siap.")
    try:
        response = await RAG_CHAIN.ainvoke({"input": data.query})
        answer = response["answer"]
        source_documents = response.get("context", []) 
        sources = []
        for doc in source_documents:
            source_info = {
                "source_type": doc.metadata.get('data_type', 'N/A'),
                "content_snippet": doc.page_content[:150] + "...",
                "metadata_ref": (
                    f"Q.S. {doc.metadata.get('surah_latin')} ayat {doc.metadata.get('ayah')}" 
                    if doc.metadata.get('surah_latin') else 
                    f"Hadis, Perawi: {doc.metadata.get('Perawi')}" if doc.metadata.get('Perawi') else "Sumber Data Tidak Diketahui"
                ),
            }
            sources.append({k: v for k, v in source_info.items() if v is not None})
        return {"answer": answer, "sources": sources, "model": "RAG (Qdrant + OpenAI LLM / LCEL)"}
    except Exception as e:
        print(f"Error RAG Chat: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Terjadi kesalahan saat memproses kueri: {e}")


@router.get("/history/{conversation_id}", response_model=List[dict])
def get_conversation_history(conversation_id: str, user_data: dict = Depends(get_current_user_data)):
    """Mengambil semua pasangan Prompt dan SmartAnswer untuk conversation_id tertentu."""
    
    prompts_stream = db.collection(PROMPTS_COLLECTION)\
                       .where('conversation_id', '==', conversation_id)\
                       .order_by('timestamp')\
                       .stream()
    
    prompts_map = {doc.id: doc.to_dict() for doc in prompts_stream}
    prompt_ids = list(prompts_map.keys())

    if not prompt_ids: return []

    # Ambil 10 prompt_id pertama (limit Firestore 'in')
    prompt_ids_to_query = prompt_ids[:10]
    
    answers_stream = db.collection(ANSWERS_COLLECTION)\
                       .where('conversation_id', '==', conversation_id)\
                       .where('prompt_id', 'in', prompt_ids_to_query)\
                       .order_by('timestamp')\
                       .stream()
    
    answers_map = {}
    for doc in answers_stream:
        data = doc.to_dict()
        answers_map[data['prompt_id']] = {"answer_id": doc.id, **data}
    
    history = []
    
    for prompt_id in prompt_ids_to_query:
        prompt_data = prompts_map.get(prompt_id)
        answer_data = answers_map.get(prompt_id)
        
        if prompt_data:
            history.append({"type": "prompt", "id": prompt_id, "text": prompt_data.get('prompt_text'), "timestamp": prompt_data.get('timestamp')})
            
        if answer_data:
            if 'conversation_id' in answer_data: del answer_data['conversation_id']
            if 'prompt_id' in answer_data: del answer_data['prompt_id']
            
            history.append({"type": "answer", "id": answer_data['answer_id'], "content": answer_data, "timestamp": answer_data.get('timestamp')})
            
    history.sort(key=lambda x: x['timestamp'])

    return history