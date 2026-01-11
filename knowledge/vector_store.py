import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "knowledge/faiss.index"
DOCS_PATH = "knowledge/docs.pkl"

model = SentenceTransformer(MODEL_NAME)

def build_vector_store(chunks):
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    return index, docs
