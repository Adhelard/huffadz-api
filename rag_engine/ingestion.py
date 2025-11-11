import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import Qdrant
from typing import List

# Muat variabel lingkungan
load_dotenv()

# --- KONFIGURASI ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "quran_hadith_data" # Harus sama dengan yang di users_chats.py

# File data Anda
QURAN_DATA_PATH = "data/quran.csv"
HADITH_DATA_PATH = "data/hadis.csv"

# Kolom yang akan dimasukkan dalam konten dokumen untuk Al-Qur'an
# Pilihlah kolom yang paling relevan untuk di-embed dan dicari.
QURAN_CONTENT_COLUMNS = ["surah_latin", "ayah", "translation", "tafsir_wajiz", "tafsir_tahlili"]
# Kolom yang akan dijadikan metadata tambahan (untuk ditampilkan sebagai sumber)
QURAN_METADATA_COLUMNS = ["surah_latin", "ayah", "surah_translation"]

# Kolom yang akan dimasukkan dalam konten dokumen untuk Hadis
HADITH_CONTENT_COLUMN = "Terjemahan" 
# Kolom yang akan dijadikan metadata tambahan
HADITH_METADATA_COLUMNS = ["Perawi"]

# --- FUNGSI UTAMA ---

def load_data_from_csv():
    """Memuat dan menggabungkan dokumen dari dua file CSV (Qur'an dan Hadis)."""
    
    all_documents = []
    
    # 1. Pemrosesan Data Al-Qur'an
    print(f"Memuat data dari {QURAN_DATA_PATH}...")
    # Gabungkan kolom konten menjadi satu string untuk Document content
    quran_metadata_to_use = [col for col in QURAN_METADATA_COLUMNS if col in pd.read_csv(QURAN_DATA_PATH).columns]
    
    # Gunakan CSVLoader dari LangChain
    quran_loader = CSVLoader(
        file_path=QURAN_DATA_PATH,
        encoding="utf-8",
        # Gabungkan kolom yang relevan ke dalam 'page_content'
        content_columns=QURAN_CONTENT_COLUMNS,
        # Kolom untuk metadata
        metadata_columns=quran_metadata_to_use
    )
    quran_docs = quran_loader.load()
    
    # Tambahkan metadata spesifik
    for doc in quran_docs:
        doc.metadata['data_type'] = 'Quran'
        # Buat snippet sumber ringkas untuk RAG
        doc.metadata['source'] = f"QS. {doc.metadata.get('surah_latin', 'N/A')}:{doc.metadata.get('ayah', 'N/A')}"
        all_documents.append(doc)

    # 2. Pemrosesan Data Hadis
    print(f"Memuat data dari {HADITH_DATA_PATH}...")
    hadith_metadata_to_use = [col for col in HADITH_METADATA_COLUMNS if col in pd.read_csv(HADITH_DATA_PATH).columns]

    hadith_loader = CSVLoader(
        file_path=HADITH_DATA_PATH,
        encoding="utf-8",
        # Konten utama adalah terjemahan
        content_columns=[HADITH_CONTENT_COLUMN],
        # Kolom untuk metadata
        metadata_columns=hadith_metadata_to_use
    )
    hadith_docs = hadith_loader.load()
    
    for doc in hadith_docs:
        doc.metadata['data_type'] = 'Hadith'
        doc.metadata['source'] = f"Perawi: {doc.metadata.get('Perawi', 'N/A')}"
        all_documents.append(doc)
        
    print(f"Total dokumen yang dimuat: {len(all_documents)}")
    return all_documents


def ingest_data_to_qdrant(documents: List):
    """Membuat vector store di Qdrant dari dokumen yang telah dimuat."""
    
    if not QDRANT_URL or not QDRANT_API_KEY :
        print("ERROR: Pastikan QDRANT_URL, QDRANT_API_KEY, dan OPENAI_API_KEY ada di .env")
        return

    # Inisialisasi Embeddings
    embeddings = OpenAIEmbeddings(
        api_key="",
        model="text-embedding-3-large",
        dimensions=1024
        )
    # Inisialisasi Qdrant Client (Langsung ke instance Qdrant Anda)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Optional: Hapus koleksi lama jika ada untuk membuat yang baru
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Koleksi lama '{COLLECTION_NAME}' dihapus.")
    except Exception:
        pass # Tidak masalah jika koleksi belum ada

    # Inisialisasi Text Splitter (Mungkin tidak diperlukan karena setiap baris CSV sudah menjadi satu dokumen)
    # Jika Anda ingin memecah konten Hadis/Tafsir yang sangat panjang, Anda bisa mengaktifkannya.
    # Namun, untuk data ayat/hadis yang relatif pendek, satu baris = satu dokumen lebih baik.
    
    # documents = RecursiveCharacterTextSplitter.split_documents(documents) # Non-aktifkan chunking untuk kasus ini
    
    # Buat Vector Store baru dan tambahkan dokumen
    print(f"Membuat Vector Store '{COLLECTION_NAME}' dan mengunggah {len(documents)} dokumen...")
    
    # Menggunakan LangChain Qdrant untuk menambahkan dokumen
    vectorstore = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True, # Memastikan koleksi dibuat
        # Optional: Konfigurasi Qdrant jika perlu, misal ukuran vektor (tergantung model OpenAI)
    )
    
    print("✅ Inisialisasi RAG Data (Ingestion) ke Qdrant Selesai!")


if __name__ == "__main__":
    # Pastikan direktori 'data' ada dan file CSV tersedia
    if not os.path.exists(QURAN_DATA_PATH) or not os.path.exists(HADITH_DATA_PATH):
        print(f"ERROR: File data tidak ditemukan. Pastikan ada di '{os.path.dirname(QURAN_DATA_PATH)}'.")
    else:
        # Step 1: Muat data dari CSV
        docs = load_data_from_csv()
        
        # Step 2: Ingest ke Qdrant
        ingest_data_to_qdrant(docs)