import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

# ================================================================
# KONFIGURASI DASAR
# ================================================================
load_dotenv()



COLLECTION_NAME = "quran_data"
QURAN_DATA_PATH = "data/quran.csv"

QURAN_CONTENT_COLUMNS = ["surah_latin", "ayah", "translation", "tafsir_wajiz", "tafsir_tahlili"]
QURAN_METADATA_COLUMNS = ["surah_latin", "ayah", "surah_translation"]

# ================================================================
# MUAT DATA DARI CSV
# ================================================================
def load_quran_docs():
    if not os.path.exists(QURAN_DATA_PATH):
        print(f"❌ File {QURAN_DATA_PATH} tidak ditemukan!")
        return []
    
    print(f"📖 Memuat data Qur'an dari {QURAN_DATA_PATH}...")
    df = pd.read_csv(QURAN_DATA_PATH)
    metadata_cols = [col for col in QURAN_METADATA_COLUMNS if col in df.columns]

    loader = CSVLoader(
        file_path=QURAN_DATA_PATH,
        encoding="utf-8",
        content_columns=[col for col in QURAN_CONTENT_COLUMNS if col in df.columns],
        metadata_columns=metadata_cols
    )
    docs = loader.load()

    for doc in docs:
        doc.metadata['data_type'] = 'Quran'
        doc.metadata['source'] = f"QS. {doc.metadata.get('surah_latin', 'N/A')}:{doc.metadata.get('ayah', 'N/A')}"
    print(f"✅ Total ayat dimuat: {len(docs)}")
    return docs

# ================================================================
# INGEST KE QDRANT
# ================================================================
def ingest_quran_to_qdrant(documents, batch_size=400):
    if not QDRANT_URL or not QDRANT_API_KEY or not OPENAI_API_KEY:
        print("❌ ERROR: Pastikan variabel lingkungan sudah diatur (.env)")
        return

    if not documents:
        print("⚠️ Tidak ada dokumen yang dimuat.")
        return

    print(f"🚀 Mengunggah data Qur'an ke koleksi '{COLLECTION_NAME}'...")

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🧹 Koleksi lama '{COLLECTION_NAME}' dihapus.")
    except Exception:
        print(f"ℹ️ Koleksi '{COLLECTION_NAME}' belum ada, membuat baru...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    print(f"📄 Total dokumen setelah split: {len(split_docs)}")

    total_batches = (len(split_docs) + batch_size - 1) // batch_size
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i:i + batch_size]
        batch_num = i // batch_size + 1

        Qdrant.from_documents(
            documents=batch,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            force_recreate=False
        )

        print(f"✅ Batch {batch_num}/{total_batches} selesai ({len(batch)} dokumen).")

    print("🎉 Ingest data Qur'an selesai!")

# ================================================================
# MAIN PROGRAM
# ================================================================
if __name__ == "__main__":
    docs = load_quran_docs()
    ingest_quran_to_qdrant(docs)
