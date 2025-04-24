from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from datetime import datetime

app = FastAPI()

# 1. Load your embedding model
print("[RAG Server] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize persistent Chroma client
print("[RAG Server] Initializing ChromaDB...")
client = chromadb.PersistentClient(
    path="./rag_store",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# 3. Create or get the 'chat_memory' collection
collection = client.get_or_create_collection(name="chat_memory")
print("[RAG Server] ChromaDB collection ready: chat_memory")

# 4. Define schema
class Query(BaseModel):
    text: str

@app.post("/rag/query")
async def query_rag(req: Query):
    print(f"[RAG] /query received: {req.text}")
    try:
        embedding = embedder.encode([req.text])[0].tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
        )
        context = results.get("documents", [[]])[0]
        print(f"[RAG] Query results: {context}")
        return {"context": context}
    except Exception as e:
        print(f"[RAG] Error in /query: {e}")
        return {"context": []}

@app.post("/rag/store")
async def store_rag(req: Query):
    print(f"[RAG] /store received: {req.text}")
    try:
        embedding = embedder.encode([req.text])[0].tolist()
        metadata = {
            "source": "user",
            "timestamp": datetime.utcnow().isoformat()
        }
        collection.add(
            documents=[req.text],
            embeddings=[embedding],
            ids=[str(hash(req.text))],
            metadatas=[metadata],
        )
        print("[RAG] Document stored successfully.")
        return {"status": "stored"}
    except Exception as e:
        print(f"[RAG] Error in /store: {e}")
        return {"status": "error", "detail": str(e)}
