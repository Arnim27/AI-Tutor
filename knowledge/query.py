from transformers import pipeline
from knowledge.vector_store import load_vector_store
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

def ask_pdf(question, teaching_style):
    greetings = ["hi", "hello", "hey", "hii"]

    if question.lower().strip() in greetings:
        return (
            "Hi! 👋 I’m your AI tutor.\n"
            "Ask me questions based on the uploaded PDF."
        )

    index, docs = load_vector_store()
    if index is None:
        return "Please upload a PDF first."

    q_embedding = embedder.encode([question])
    D, I = index.search(np.array(q_embedding), k=3)

    context = "\n".join([docs[i] for i in I[0]])

    prompt = f"""
You are an AI tutor.
Teaching style: {teaching_style}

Answer ONLY from the content below.
If the answer is not present, say:
"This is not covered in the document."

Content:
{context}

Question:
{question}
"""

    result = llm(prompt)[0]["generated_text"]
    return result
