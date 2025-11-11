import os
import pandas as pd
from dotenv import load_dotenv
from typing import List

# --- IMPORT HUGGINGFACE & LANGCHAIN CORE ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
# --- IMPORT QDRANT CLIENT ---
from qdrant_client import QdrantClient 

import time 

load_dotenv()

# Konfigurasi dari .env
COLLECTION_NAME = "quran_hadith_data"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- KONSTANTA OPTIMAL UNTUK INDEXING ---
QDRANT_TIMEOUT = 300 # 5 menit timeout untuk upsert data besar
QDRANT_BATCH_SIZE = 250 # Ukuran batch untuk mengunggah vektor

# 1. Custom Document Creation
def create_documents_from_csv(file_path: str, data_type: str) -> List[Document]:
    """Memuat dan memformat data CSV menjadi objek Document LangChain."""
    df = pd.read_csv(file_path).fillna('') # Mengganti NaN dengan string kosong
    documents = []

    for _, row in df.iterrows():
        # Tambahkan data_type ke metadata untuk memudahkan identifikasi sumber
        metadata = row.to_dict()
        metadata["data_type"] = data_type 
        
        page_content = ""
        
        # Logika Penggabungan Kolom untuk Konten yang Kaya
        if data_type == "hadith":
            # Kolom: Perawi, Arab, Terjemahan
            page_content = (
                f"Hadis: {metadata.get('Terjemahan', 'N/A')}. "
                f"Teks Arab: {metadata.get('Arab', 'N/A')}. "
                f"Perawi: {metadata.get('Perawi', 'N/A')}."
            )
        elif data_type == "quran":
            # Kolom: ..., translation, tafsir_wajiz, ...
            page_content = (
                f"Ayat Al-Qur'an (Q.S. {metadata.get('surah_latin')} ayat {metadata.get('ayah')}, Juz {metadata.get('juz')}): "
                f"Terjemahan: {metadata.get('translation', 'N/A')}. "
                f"Tafsir Wajiz: {metadata.get('tafsir_wajiz', 'N/A')}."
            )

        # Buat dokumen LangChain
        documents.append(
            Document(page_content=page_content, metadata=metadata)
        )
        
    return documents

def index_data(quran_csv_path: str, hadith_csv_path: str):
    """Mengindeks data ke Qdrant."""
    print("Mulai Indexing data...")
    
    # Memuat dan Memformat Data
    quran_docs = create_documents_from_csv(quran_csv_path, "quran")
    hadith_docs = create_documents_from_csv(hadith_csv_path, "hadith")
    all_docs = quran_docs + hadith_docs
    print(f"Total dokumen yang dimuat: {len(all_docs)}")

    # Text Splitting 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(all_docs) 
    print(f"Total chunks yang dibuat: {len(splits)}")

    # --- INISIALISASI EMBEDDINGS HUGGINGFACE (GRATIS) ---
    print("Inisialisasi Embeddings Open-Source (HuggingFace BGE)...")
    model_name = "BAAI/bge-small-en-v1.5"
    # Ganti 'cpu' dengan 'cuda' jika VPS Anda punya GPU
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Inisialisasi Qdrant Client dengan TIMEOUT TINGGI
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT # Mengatasi WriteTimeout
    )
    
    # Cek dan Hapus koleksi lama
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Koleksi lama '{COLLECTION_NAME}' berhasil dihapus.")
    except Exception:
        print(f"Koleksi '{COLLECTION_NAME}' tidak ada, akan dibuat baru.")
        
    # Membuat Vector Store dan mengunggah data
    print(f"Mengunggah embeddings ({len(splits)} chunks) ke Qdrant...")
    time.sleep(2) # Tunda sebentar

    vectorstore = Qdrant.from_documents(
        splits,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        batch_size=QDRANT_BATCH_SIZE # Mengunggah dalam batch yang lebih kecil
    )
    print(f"Data berhasil diindeks dalam koleksi Qdrant: {COLLECTION_NAME}")
    return vectorstore

# Contoh penggunaan: Ganti dengan path file CSV Anda
if __name__ == "__main__":
    # PASTIKAN QDRANT SUDAH BERJALAN di http://localhost:6333
    index_data("./data/quran.csv", "./data/hadis.csv")
    print("Selesai. Vektor sudah ada di Qdrant.")