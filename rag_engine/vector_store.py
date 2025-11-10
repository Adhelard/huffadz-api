from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(quran_docs, hadith_docs, persist_path="./rag_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    all_docs = quran_docs + hadith_docs
    db = FAISS.from_documents(all_docs, embedding=embeddings)
    db.save_local(persist_path)
    return db

def load_vector_store(persist_path="./rag_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
