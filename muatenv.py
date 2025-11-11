import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

QDRANT_HOST = os.getenv("QDRANT_URL")
QDRANT_API = os.getenv("QDRANT_API")