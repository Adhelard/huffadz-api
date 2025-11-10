from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from firebase_config import db  # Klien Firestore
from .auth import get_current_user_data  # Dependency Auth
from typing import List, Optional
from google.cloud import firestore  # Untuk SERVER_TIMESTAMP

from rag_engine.loader import load_quran_csv, load_hadith_csv
from rag_engine.vector_store import build_vector_store, load_vector_store
from rag_engine.qa_chain import get_rag_chain

import os

router = APIRouter(tags=["Users, Conversations & Messages (Prompt/Answer)"])

# -----------------------------------------------------------------
# --- MODEL PYDANTIC ---
# -----------------------------------------------------------------

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


class LetterContent(BaseModel):
    letter_type: str
    recipient: str
    sender: str
    date: str
    salutation: str
    body_paragraphs: List[str]
    closing: str


class QuranicContent(BaseModel):
    surah_name: str
    surah_number: int
    ayah_number: str
    arabic_text: Optional[str] = None
    translation: str
    tafsir_summary: Optional[str] = None


class HadithContent(BaseModel):
    book: Optional[str] = None
    number: Optional[str] = None
    narrator: Optional[str] = None
    arabic_text: Optional[str] = None
    translation: str
    details: Optional[str] = None


class SmartAnswer(BaseModel):
    conversation_id: str
    prompt_id: str
    summary_text: str
    letter_example: Optional[LetterContent] = None
    hadith_example: Optional[HadithContent] = None
    quran_example: Optional[QuranicContent] = None
    sources: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------
# --- FIRESTORE COLLECTIONS ---
# -----------------------------------------------------------------

USERS_COLLECTION = "users"
CONVERSATIONS_COLLECTION = "conversations"
PROMPTS_COLLECTION = "prompts"
ANSWERS_COLLECTION = "answers"


# -----------------------------------------------------------------
# --- INISIALISASI RAG ---
# -----------------------------------------------------------------

VECTOR_PATH = "./rag_db"

try:
    vector_store = load_vector_store(VECTOR_PATH)
    print("✅ Vector store loaded from disk.")
except Exception:
    print("⚙️ Building new vector store from CSV files...")
    quran_docs = load_quran_csv("./data/quran.csv")
    hadith_docs = load_hadith_csv("./data/hadis.csv")
    vector_store = build_vector_store(quran_docs, hadith_docs, VECTOR_PATH)
    print("✅ Vector store built and saved.")

rag_chain = get_rag_chain(vector_store)


def generate_smart_answer_rag(conversation_id: str, prompt_id: str, prompt_text: str) -> SmartAnswer:
    """Menjawab pertanyaan menggunakan LangChain Core & LangGraph (RAG)."""
    try:
        answer_text = rag_chain.invoke(prompt_text)
        return SmartAnswer(
            conversation_id=conversation_id,
            prompt_id=prompt_id,
            summary_text=answer_text,
            sources=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghasilkan jawaban RAG: {str(e)}")


# -----------------------------------------------------------------
# --- ENDPOINTS ---
# -----------------------------------------------------------------

@router.get("")
def api_root():
    return {"message": "API v1 OK"}


# 🧍‍♂️ USER PROFILE ------------------------------------------------

@router.post("/users/register", status_code=status.HTTP_201_CREATED)
def register_user_profile(profile: UserProfile, user_data: dict = Depends(get_current_user_data)):
    user_uid = user_data.get('uid')
    try:
        data_to_save = profile.model_dump()
        data_to_save['email'] = user_data.get('email')
        db.collection(USERS_COLLECTION).document(user_uid).set(data_to_save)
        return {"id": user_uid, **data_to_save}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/me")
def get_my_profile(user_data: dict = Depends(get_current_user_data)):
    user_uid = user_data.get('uid')
    doc = db.collection(USERS_COLLECTION).document(user_uid).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Profil belum terdaftar.")
    return {"id": doc.id, **doc.to_dict()}


# 💬 CONVERSATIONS --------------------------------------------------

@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(conv: Conversation, user_data: dict = Depends(get_current_user_data)):
    if conv.user_id != user_data.get('uid'):
        raise HTTPException(status_code=403, detail="Tidak diizinkan membuat percakapan untuk pengguna lain.")
    try:
        data = conv.model_dump()
        data['created_at'] = firestore.SERVER_TIMESTAMP
        doc_ref = db.collection(CONVERSATIONS_COLLECTION).document()
        doc_ref.set(data)
        return {"conversation_id": doc_ref.id, **data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=List[dict])
def get_my_conversations(user_data: dict = Depends(get_current_user_data)):
    user_uid = user_data.get('uid')
    convs = db.collection(CONVERSATIONS_COLLECTION).where('user_id', '==', user_uid).stream()
    return [{"conversation_id": d.id, **d.to_dict()} for d in convs]


# 🤖 PROMPTS & RAG ANSWERS -----------------------------------------




@router.post("/prompts", status_code=status.HTTP_201_CREATED)
def post_prompt(prmt: Prompt, user_data: dict = Depends(get_current_user_data)):
    sender_uid = user_data.get('uid')
    if prmt.sender_uid != sender_uid:
        raise HTTPException(status_code=403, detail="UID pengirim tidak cocok.")

    try:
        # Simpan Prompt ke Firestore
        data_prompt = prmt.model_dump()
        data_prompt['timestamp'] = firestore.SERVER_TIMESTAMP
        doc_ref_prompt = db.collection(PROMPTS_COLLECTION).document()
        doc_ref_prompt.set(data_prompt)
        prompt_id = doc_ref_prompt.id

        # Hasilkan jawaban dengan RAG
        smart_answer_obj = generate_smart_answer_rag(
            conversation_id=prmt.conversation_id,
            prompt_id=prompt_id,
            prompt_text=prmt.prompt_text
        )

        # Simpan jawaban
        data_answer = smart_answer_obj.model_dump(exclude_none=True)
        data_answer['timestamp'] = firestore.SERVER_TIMESTAMP
        doc_ref_answer = db.collection(ANSWERS_COLLECTION).document()
        doc_ref_answer.set(data_answer)

        return {
            "prompt_id": prompt_id,
            "answer_id": doc_ref_answer.id,
            "conversation_id": prmt.conversation_id,
            "status": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error menyimpan prompt/answer: {str(e)}")


# 📜 HISTORY ---------------------------------------------------------

@router.get("/history/{conversation_id}", response_model=List[dict])
def get_conversation_history(conversation_id: str, user_data: dict = Depends(get_current_user_data)):
    prompts_stream = db.collection(PROMPTS_COLLECTION).where('conversation_id', '==', conversation_id).order_by('timestamp').stream()
    prompts_map = {d.id: d.to_dict() for d in prompts_stream}
    if not prompts_map:
        return []

    answers_stream = db.collection(ANSWERS_COLLECTION).where('conversation_id', '==', conversation_id).order_by('timestamp').stream()
    answers_map = {d.to_dict()['prompt_id']: {"answer_id": d.id, **d.to_dict()} for d in answers_stream}

    history = []
    for pid, p in prompts_map.items():
        history.append({"type": "prompt", "id": pid, "text": p['prompt_text'], "timestamp": p.get('timestamp')})
        if pid in answers_map:
            ans = answers_map[pid]
            del ans['conversation_id']
            del ans['prompt_id']
            history.append({"type": "answer", "id": ans['answer_id'], "content": ans, "timestamp": ans.get('timestamp')})
    history.sort(key=lambda x: x['timestamp'])
    return history
