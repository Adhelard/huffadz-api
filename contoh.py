import openai
import chromadb
import os
from dotenv import load_dotenv

# --- KONFIGURASI ULANG ---
# Pastikan konfigurasi ini sama dengan skrip indexing

load_dotenv() # Biasanya digunakan untuk memuat variabel lingkungan dari file .env
openai_client = openai.OpenAI(api_key="") # Ambil API Key dari environment variable

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo" # Atau "gpt-4-turbo" untuk hasil yang lebih baik

# Inisialisasi Klien ChromaDB (PersistentClient untuk membaca data yang sudah tersimpan)
client_chroma = chromadb.PersistentClient(path="./vector_db")

# Ambil kembali koleksi yang sudah dibuat
collection_quran = client_chroma.get_collection(name="quran")
collection_hadith = client_chroma.get_collection(name="hadith")

def get_relevant_documents(query: str, k: int = 3):
    """
    Mengubah pertanyaan menjadi embedding dan mencarinya di koleksi Quran dan Hadith.
    """
    # 1. Buat Embedding untuk Pertanyaan (Query)
    response = openai_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding
    
    # 2. Cari di Koleksi Quran
    results_quran = collection_quran.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['metadatas', 'documents', 'distances']
    )
    
    # 3. Cari di Koleksi Hadith
    results_hadith = collection_hadith.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['metadatas', 'documents', 'distances']
    )
    
    # Gabungkan hasil pencarian
    all_results = []
    
    # Memproses hasil Quran
    if results_quran['metadatas']:
        for meta, doc, dist in zip(results_quran['metadatas'][0], results_quran['documents'][0], results_quran['distances'][0]):
            # Format output untuk Quran
            source_info = f"QS {meta['surah']} Ayat {meta['ayah_number']} (Juz {meta['juz']})"
            all_results.append({
                "source": source_info,
                "text": meta['original_text'], # Ambil terjemahan bersih
                "distance": dist
            })
            
    # Memproses hasil Hadith
    if results_hadith['metadatas']:
        for meta, doc, dist in zip(results_hadith['metadatas'][0], results_hadith['documents'][0], results_hadith['distances'][0]):
            # Format output untuk Hadith
            source_info = f"HR {meta['Perawi']}"
            all_results.append({
                "source": source_info,
                "text": meta['original_text'], # Ambil terjemahan bersih
                "distance": dist
            })

    # Urutkan berdasarkan jarak (similarity)
    all_results.sort(key=lambda x: x['distance'])
    
    # Ambil 3 dokumen terbaik secara keseluruhan (Anda bisa menyesuaikan k)
    return all_results[:k]

def answer_question_with_rag(query: str):
    """
    Melakukan RAG: mencari dokumen dan menggunakannya untuk menghasilkan jawaban.
    """
    print(f"-> Mencari dokumen relevan untuk: '{query}'...")
    relevant_docs = get_relevant_documents(query, k=5) # Ambil 5 dokumen teratas
    
    if not relevant_docs:
        # Jika tidak ada dokumen yang ditemukan, coba jawab tanpa konteks atau beri tahu pengguna
        print("!! Tidak ditemukan dokumen relevan dari database. Mencoba menjawab tanpa konteks.")
        context = ""
        source_citations = "Tidak ada sumber data yang ditemukan di database."
    else:
        # 1. Buat String Konteks dari Dokumen yang Ditemukan
        context_parts = []
        source_citations = "\n\n**Sumber Data yang Ditemukan:**\n"
        
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"[{i+1}] {doc['source']}: {doc['text']}")
            source_citations += f"* [{i+1}] {doc['source']}\n"
            
        context = "\n---\n".join(context_parts)
        print("-> Dokumen relevan berhasil diambil. Mengirim ke LLM...")

    # 2. Susun Prompt ke LLM
    system_prompt = (
        "Anda adalah asisten Islami yang berpengetahuan luas. "
        "Gunakan informasi yang disediakan di bawah ini (KONTEKS) untuk menjawab "
        "pertanyaan pengguna. Jika KONTEKS tidak memuat jawaban yang relevan, "
        "jawablah dengan jujur bahwa Anda tidak dapat menemukan informasi yang "
        "sesuai di database Anda, lalu coba jawab berdasarkan pengetahuan umum Anda. "
        "Sebutkan nomor referensi dari KONTEKS di dalam jawaban Anda, misalnya [1] atau [2]."
    )
    
    full_prompt = (
        f"PERTANYAAN PENGGUNA:\n{query}\n\n"
        "KONTEKS DARI DATABASE (Al-Qur'an dan Hadits):\n"
        "===========================================\n"
        f"{context}\n"
        "===========================================\n\n"
        "Jawaban Anda (Gunakan bahasa Indonesia yang baku dan informatif):"
    )

    # 3. Panggil API OpenAI (LLM)
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        return f"## ✅ Jawaban RAG Anda\n\n{answer}\n\n---\n{source_citations}"
        
    except Exception as e:
        return f"Terjadi kesalahan saat memanggil LLM: {e}"
    

if __name__ == "__main__":
    
    # PENTING: Anda harus memastikan bahwa skrip indexing sudah dijalankan
    # dan folder './vector_db' sudah terisi.
    
    print("\n\n--- Memulai Pengujian RAG ---")
    
    # Contoh Pertanyaan
    question_1 = "Apa perintah untuk menafkahkan harta di jalan Allah?"
    print(answer_question_with_rag(question_1))
    
    print("\n" + "="*50 + "\n")
    
    question_2 = "Apa yang Rasulullah katakan tentang amal yang paling dicintai Allah?"
    print(answer_question_with_rag(question_2))
    
    print("\n" + "="*50 + "\n")
    
    question_3 = "Apakah ada ayat yang menjelaskan mengenai larangan berbuat syirik?"
    print(answer_question_with_rag(question_3))