import pandas as pd
import openai
import chromadb
from tqdm import tqdm
import os
import time

# --- 1. KONFIGURASI ---



# Inisialisasi Klien
# Gunakan OpenAI() (sync) karena ini adalah skrip satu kali
openai_client = openai.OpenAI(api_key="") 

# Gunakan PersistentClient agar database tersimpan di disk!
# Ini akan membuat folder './vector_db'
client_chroma = chromadb.PersistentClient(path="./vector_db")

# Buat (atau ambil jika sudah ada) koleksi Anda
collection_quran = client_chroma.get_or_create_collection(name="quran")
collection_hadith = client_chroma.get_or_create_collection(name="hadith")

# Nama file CSV Anda
QURAN_CSV_PATH = "data/quran.csv"
HADITH_CSV_PATH = "data/hadis.csv"

# Model Embedding
EMBEDDING_MODEL = "text-embedding-3-small"

# Ukuran batch untuk API (penting untuk efisiensi)
BATCH_SIZE = 25

# --- 2. FUNGSI UNTUK MEMPROSES CSV ---

def process_and_index(csv_path, collection, data_type):
    """Membaca CSV, membuat embedding, dan menyimpannya ke ChromaDB."""
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\nMemulai pemrosesan untuk: {csv_path} ({len(df)} baris)")
    except FileNotFoundError:
        print(f"File tidak ditemukan: {csv_path}. Silakan periksa nama file.")
        return

    # Siapkan list untuk batching
    batch_documents = []
    batch_metadatas = []
    batch_ids = []

    # Gunakan tqdm untuk progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Membaca {data_type}"):
        
        # --- SESUAIKAN KOLOM DI SINI ---
        try:
            if data_type == 'quran':
            # Penyesuaian berdasarkan kolom baru:
            # 'surah_name' -> 'surah_latin' (e.g., "Al-Fatihah")
            # 'ayah_number' -> 'ayah' (e.g., 1, 2, 3)
            # 'surah_number' -> 'surah_id' (e.g., 1)
            # 'translation' -> 'translation' (Nama kolom tetap)

            # Teks yang akan di-embed (hanya terjemahan, sesuai kode asli)
                text_to_embed = f"QS {row['surah_latin']} Ayat {row['ayah']}:\n" \
                                f"Teks Arab: {row['arabic']}\n" \
                                f"Terjemahan: {row['translation']}"
            # Metadata disesuaikan dan diperkaya dengan kolom baru
                metadata = {
                    "source": "quran",
                    "surah": row['surah_latin'],           # Sebelumnya: row['surah_name']
                    "surah_number": int(row['surah_id']),  # Sebelumnya: int(row['surah_number'])
                    "ayah_number": int(row['ayah']),       # Sebelumnya: int(row['ayah_number'])
                    "original_text": row['translation'],   # Teks terjemahan (sesuai kode asli)
                    
                    # (Rekomendasi tambahan dari kolom baru)
                    "arabic_text": row['arabic'],        # Simpan teks Arab
                    "juz": int(row['juz']),              # Simpan info Juz
                    "page": int(row['page'])             # Simpan info Halaman
                }
                
            # ID Dokumen disesuaikan
                doc_id = f"quran_{row['surah_id']}_{row['ayah']}"
            
            elif data_type == 'hadith':
                # Asumsi nama kolom: book_name, hadith_number, translation
                text_to_embed = f"HR {row['Perawi']} Arab: {row['Arab']} terjemahan: {row['Terjemahan']}"
                metadata = {
                    "source": "hadith",
                    "Perawi": row['Perawi'],
                    "Arab": row['Arab'],
                    "original_text": row['Terjemahan']
                }
                doc_id = f"hadith_{row['Perawi']}"
            
            else:
                continue
                
        except KeyError as e:
            print(f"\nError: Nama kolom {e} tidak ditemukan di {csv_path}.")
            print("Harap sesuaikan nama kolom di dalam skrip `index_data.py`.")
            return
        except Exception as e:
            print(f"\nError data pada baris {index}: {e}")
            continue # Lompati baris yang error

        # Tambahkan ke batch
        batch_documents.append(text_to_embed)
        batch_metadatas.append(metadata)
        batch_ids.append(doc_id)

        # Jika batch sudah penuh, proses
        if len(batch_documents) >= BATCH_SIZE:
            process_batch(collection, batch_documents, batch_metadatas, batch_ids)
            # Kosongkan batch
            batch_documents, batch_metadatas, batch_ids = [], [], []

    # Proses sisa batch terakhir
    if batch_documents:
        process_batch(collection, batch_documents, batch_metadatas, batch_ids)

    print(f"Selesai memproses {csv_path}.")

def process_batch(collection, documents, metadatas, ids):
    """Memanggil API OpenAI dan menyimpan ke ChromaDB."""
    try:
        # 1. Panggil API OpenAI untuk Embedding
        response = openai_client.embeddings.create(
            input=documents,
            model=EMBEDDING_MODEL
        )
        embeddings = [item.embedding for item in response.data]
        
        # 2. Simpan ke ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=documents, # Kita simpan juga teks yg di-embed
            metadatas=metadatas,
            ids=ids
        )
    except openai.RateLimitError:
        print("Terkena Rate Limit OpenAI. Menunggu 60 detik...")
        time.sleep(60)
        process_batch(collection, documents, metadatas, ids) # Coba lagi
    except Exception as e:
        print(f"Error saat memproses batch: {e}")

# --- 3. JALANKAN PROSES ---

if __name__ == "__main__":
    print("--- Memulai Proses Indexing Database Vektor ---")
    
    # Proses Al-Qur'an
    process_and_index(
        csv_path=QURAN_CSV_PATH,
        collection=collection_quran,
        data_type='quran'
    )
    
    # Proses Hadits
    process_and_index(
        csv_path=HADITH_CSV_PATH,
        collection=collection_hadith,
        data_type='hadith'
    )
    
    print("\n--- Proses Indexing Selesai ---")
    print(f"Database Vektor telah dibuat di folder: ./vector_db")
    print(f"Total data di koleksi Quran: {collection_quran.count()}")
    print(f"Total data di koleksi Hadits: {collection_hadith.count()}")