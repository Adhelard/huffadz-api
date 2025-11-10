import pandas as pd
from langchain_core.documents import Document

def load_quran_csv(path: str):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        text = (
            f"{row['arabic']} ({row['latin']})\n\n"
            f"Terjemahan: {row['translation']}\n\n"
            f"Tafsir: {row.get('tafsir_wajiz', '')}"
        )
        meta = {
            "source": "Quran",
            "surah": row.get("surah_latin", ""),
            "ayah": row.get("ayah", "")
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def load_hadith_csv(path: str):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        text = f"{row['Arab']}\n\nTerjemahan: {row['Terjemahan']}"
        meta = {"source": "Hadis", "perawi": row.get("Perawi", "")}
        docs.append(Document(page_content=text, metadata=meta))
    return docs
