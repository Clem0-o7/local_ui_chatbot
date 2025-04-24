# rag_server.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from datetime import datetime


app = FastAPI()

# 1. Load your embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize a persistent Chroma client (stores data in ./rag_store)
#    This replaces the old Settings-based constructor with PersistentClient. :contentReference[oaicite:0]{index=0}
client = chromadb.PersistentClient(
    path="./rag_store",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# 3. Create or get the 'chat_memory' collection for embeddings & texts
collection = client.get_or_create_collection(name="chat_memory")

# 4. Define request schema
class Query(BaseModel):
    text: str

@app.post("/rag/query")
async def query_rag(req: Query):
    # Embed the incoming text
    embedding = embedder.encode([req.text])[0].tolist()
    # Query the top-3 most similar stored documents
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
    )
    # Return the list of document strings (or empty list)
    return {"context": results.get("documents", [[]])[0]}

@app.post("/rag/store")
async def store_rag(req: Query):
    # 1. Encode the text
    embedding = embedder.encode([req.text])[0].tolist()

    # 2. Prepare a non-empty metadata dict
    metadata = {
        "source": "user",                # identifies who spoke
        "timestamp": datetime.utcnow().isoformat()  # optional timestamp
    }

    # 3. Add to ChromaDB with metadata
    collection.add(
        documents=[req.text],
        embeddings=[embedding],
        ids=[str(hash(req.text))],
        metadatas=[metadata],
    )
    return {"status": "stored"}
