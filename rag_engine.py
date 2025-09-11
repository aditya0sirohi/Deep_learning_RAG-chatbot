# rag_engine.py
import os
import json
import re
from dotenv import load_dotenv

# LangChain official imports (no langchain_community)
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from openai import OpenAI

# ---------------- Utilities ----------------
def improved_clean(text: str) -> str:
    text = re.sub(r'(\b\w+\b)(?:\s*\1\b)+', r'\1', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_chunks_file(input_file="doc_chunks.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- Env & client ----------------
load_dotenv()
# Accept either OPENROUTER_API_KEY or OPENAI_API_KEY (so it works both ways)
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    # not a fatal crash: we will let the caller see a clear message later
    print("WARNING: No OPENROUTER_API_KEY or OPENAI_API_KEY found in env/secrets.")

client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# ---------------- Embeddings & Vectorstore ----------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Try to load FAISS index from repo folder ./faiss_index
VECTORSTORE_DIR = "faiss_index"

vectorstore = None
try:
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS index from", VECTORSTORE_DIR)
except Exception as e:
    print("FAISS.load_local failed:", e)
    # Fallback: attempt to rebuild from doc_chunks.json (if present)
    try:
        chunks = load_chunks_file("doc_chunks.json")
        documents = [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in chunks]
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTORSTORE_DIR)
        print("Rebuilt FAISS index and saved to", VECTORSTORE_DIR)
    except FileNotFoundError:
        print("FATAL: doc_chunks.json not found and faiss_index load failed. Please upload faiss_index/ or doc_chunks.json to the Space.")
        # leave vectorstore as None

# ---------------- Prompt Template ----------------
prompt_template = """
You are a knowledgeable assistant.
Always provide an in-depth answer (at least 5-8 sentences) and, when useful, include examples and brief reasoning.
If the context doesn't contain an answer, respond: "Sorry, answer not found."

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---------------- RAG Pipeline ----------------
def rag_pipeline(question: str, k: int = 3):
    if not vectorstore:
        return "Error: vectorstore not loaded. Please ensure faiss_index/ or doc_chunks.json is present in the Space."

    if "fuck you" in question.lower():
        return "Fuck you too :)."

    docs = vectorstore.similarity_search(question, k=k)
    if not docs:
        return "Sorry, no relevant documents found."

    # Keep context limited to avoid hitting token limits
    context = "\n\n".join([doc.page_content[:1000] for doc in docs])
    prompt_text = prompt.format(context=context, question=question)

    # call OpenRouter via the OpenAI-compatible client
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-4b:free",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.9,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM request failed: {e}"
