import os
import pandas as pd
import math
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from typing import List

# Muat variabel lingkungan
load_dotenv()

# --- KONFIGURASI ---
COLLECTION_NAME = "islamic_ai_vs" 

QDRANT_URL = ""
QDRANT_API_KEY = ""
QURAN_DATA_PATH = "data/quran.csv"
HADITH_DATA_PATH = "data/hadis.csv"

# Kolom konten dan metadata (tidak berubah)
QURAN_CONTENT_COLUMNS = ["surah_latin", "ayah", "translation", "tafsir_wajiz", "tafsir_tahlili"]
QURAN_METADATA_COLUMNS = ["surah_latin", "ayah", "surah_translation", "arabic", "juz", "page"]
HADITH_CONTENT_COLUMN = "Terjemahan" 
HADITH_METADATA_COLUMNS = ["Perawi","arab"]

# --- FUNGSI LOAD DATA (Tidak berubah, sudah bagus) ---

def load_data_from_csv() -> List[Document]:
    """
    Memuat dan menggabungkan dokumen dari dua file CSV (Qur'an dan Hadis)
    menggunakan pandas untuk kontrol penuh atas pembuatan Dokumen.
    """
    
    all_documents = []
    
    # 1. Pemrosesan Data Al-Qur'an
    print(f"Memuat data dari {QURAN_DATA_PATH}...")
    try:
        quran_df = pd.read_csv(QURAN_DATA_PATH, encoding="utf-8")
        quran_df = quran_df.fillna("") 
    except FileNotFoundError:
        print(f"ERROR: File {QURAN_DATA_PATH} tidak ditemukan.")
        return []
    except Exception as e:
        print(f"ERROR saat membaca {QURAN_DATA_PATH}: {e}")
        return []

    for _, row in quran_df.iterrows():
        content_parts = []
        if 'arabic' in row and row['arabic']:
            content_parts.append(f"Teks Arab: {row['arabic']}")

        for col in QURAN_CONTENT_COLUMNS:
            if col in row:
                content_parts.append(f"{col}: {row[col]}")
        page_content = "\n".join(content_parts)
        
        metadata = {}
        for col in QURAN_METADATA_COLUMNS:
            if col in row:
                metadata[col] = row[col]
                
        metadata['data_type'] = 'Quran'
        metadata['source'] = f"QS. {metadata.get('surah_latin', 'N/A')}:{metadata.get('ayah', 'N/A')}"
        
        all_documents.append(Document(page_content=page_content, metadata=metadata))

    # 2. Pemrosesan Data Hadis
    print(f"Memuat data dari {HADITH_DATA_PATH}...")
    try:
        hadith_df = pd.read_csv(HADITH_DATA_PATH, encoding="utf-8")
        hadith_df = hadith_df.fillna("")
    except FileNotFoundError:
        print(f"ERROR: File {HADITH_DATA_PATH} tidak ditemukan.")
        return all_documents 
    except Exception as e:
        print(f"ERROR saat membaca {HADITH_DATA_PATH}: {e}")
        return all_documents

    for _, row in hadith_df.iterrows():
        terjemahan = row.get(HADITH_CONTENT_COLUMN, "")
        arab = row.get("arab", "") # Ambil teks arab dari metadata
        
        # Gabungkan keduanya untuk page_content
        page_content = f"Teks Arab: {arab}\nTerjemahan: {terjemahan}"

        metadata = {}
        for col in HADITH_METADATA_COLUMNS:
            if col in row:
                metadata[col] = row[col]

        metadata['data_type'] = 'Hadith'
        metadata['source'] = f"Perawi: {metadata.get('Perawi', 'N/A')}"
        
        all_documents.append(Document(page_content=page_content, metadata=metadata))
        
    print(f"Total dokumen yang dimuat: {len(all_documents)}")
    return all_documents

# --- PERUBAHAN DI SINI ---
# BATCH_SIZE = 1000  # <--- INI TERLALU BESAR, MENYEBABKAN TIMEOUT
# Impor dan fungsi load_data_from_csv() biarkan sama...

BATCH_SIZE = 50     # <--- PERUBAHAN: Kita kecilkan lagi

def ingest_data_to_qdrant(documents: List[Document]):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(documents)
    total_docs_to_process = len(split_docs)
    print(f"Total dokumen setelah di-split: {total_docs_to_process}")
    
    if not split_docs:
        print("Tidak ada dokumen untuk di-ingest. Keluar.")
        return

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        timeout=60 
    )
    
    print(f"Mencoba koneksi ke Qdrant di {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        client.get_collections() 
        print("Koneksi Qdrant berhasil.")
    except Exception as e:
        print(f"GAGAL terhubung ke Qdrant: {e}")
        print("Pastikan Qdrant server berjalan dan bisa diakses dari script ini.")
        return 
    
    # --- LOGIKA BARU UNTUK 'RESUME' ---
    
    start_index = 0
    is_first_batch = True

    try:
        # 1. Cek apakah koleksi sudah ada
        client.get_collection(collection_name=COLLECTION_NAME)
        
        # 2. Jika ada, hitung dokumen yang ada
        print(f"Koleksi '{COLLECTION_NAME}' sudah ada. Menghitung dokumen...")
        # Kita gunakan exact=True untuk jumlah yang pasti
        count_result = client.count(collection_name=COLLECTION_NAME, exact=True)
        existing_docs = count_result.count
        
        if existing_docs > 0:
            print(f"Ditemukan {existing_docs} dokumen yang sudah ada.")
            
            if existing_docs >= total_docs_to_process:
                print("Semua dokumen tampaknya sudah di-ingest. Selesai.")
                return
            
            print(f"Melanjutkan proses dari dokumen ke-{existing_docs + 1}...")
            start_index = existing_docs # Mulai dari dokumen yang ada
            is_first_batch = False # PENTING! Jangan hapus koleksi yang ada

    except Exception as e:
        # 3. Jika koleksi tidak ada, kita mulai dari awal
        # Error 'Not found' atau sejenisnya berarti koleksi belum ada
        print(f"Info: Koleksi '{COLLECTION_NAME}' tidak ditemukan. Memulai dari awal.")
        # Hapus (jika ada sisa-sisa) untuk memastikan
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            print("Membersihkan koleksi lama (jika ada).")
        except:
            pass # Abaikan jika memang tidak ada
            
        start_index = 0
        is_first_batch = True
    
    # --- AKHIR LOGIKA BARU ---

    print(f"Memulai proses unggah dokumen...")

    # Hitung ulang total batch yang TERSISA
    remaining_docs = total_docs_to_process - start_index
    total_batches = math.ceil(remaining_docs / BATCH_SIZE)
    print(f"Total batch yang akan diproses: {total_batches}")

    # --- PERUBAHAN PADA LOOP ---
    # Kita mulai loop dari start_index, bukan dari 0
    for i in range(start_index, total_docs_to_process, BATCH_SIZE):
        # Ambil batch dokumen dari list utuh
        batch_docs = split_docs[i : i + BATCH_SIZE]
        
        # Logika print yang lebih baik
        current_batch_num = ((i - start_index) // BATCH_SIZE) + 1
        docs_processed_count = i + len(batch_docs)
        print(f"Memproses batch {current_batch_num} dari {total_batches} (Total: {docs_processed_count}/{total_docs_to_process} dokumen)...", end="\r")
        
        try:
            Qdrant.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=COLLECTION_NAME,
                force_recreate=is_first_batch, # Ini hanya akan True jika start_index = 0
                timeout=60
            )
        except Exception as e:
            print(f"\nERROR pada batch {current_batch_num}: {e}")
            print("Proses dihentikan. Anda bisa menjalankan ulang skrip ini untuk melanjutkan.")
            break
        
        # Sangat penting: setelah batch pertama (jika itu yg pertama), set ke False
        is_first_batch = False
    
    print(f"\nProses unggah selesai.")               # Tambah newline


if __name__ == "__main__":
    if not os.path.exists(QURAN_DATA_PATH) or not os.path.exists(HADITH_DATA_PATH):
        print(f"ERROR: File data tidak ditemukan. Pastikan ada di '{os.path.dirname(QURAN_DATA_PATH)}'.")
    else:
        docs = load_data_from_csv()
        
        if docs:
            print(f"Data berhasil dimuat (Total: {len(docs)} dokumen). Memulai ingest...")
            ingest_data_to_qdrant(docs)
            print("Proses indexing selesai.")
        else:
            print("Tidak ada dokumen yang dimuat. Proses dihentikan.")