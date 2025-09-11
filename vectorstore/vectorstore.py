# from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
# import document_processor as dp
import json
from langchain.schema import Document

def load_chunks(input_file="doc_chunks.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)

chunks = load_chunks()
    
documents = [
    Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
    for chunk in chunks
    ]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings)

vectorstore.save_local("faiss_index")

# chunks = load_chunks()
# vectorstore = create_vectorstore(chunks)
print("FAISS index saved to ./faiss_index/")