import re
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
import json

# def clean(text):
#     text = re.sub(r'(\b\w+\b)(?:\s+\1\b)+', r'\1', text)
#     text = re.sub(r'(\b\w\b)(?:\s+\1\b)+', r'\1', text)
#     text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)
#     return text.strip()

def improved_clean(text):
    text = re.sub(r'(\b\w+\b)(?:\s*\1\b)+', r'\1', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

path = '/home/amantya/Desktop/task1/dl-rag-bot/data/lbdl.pdf'

def load_and_split(path, chunk_size=500, chunk_overlap=50):
    # loader = PyPDFLoader(path)
    # loader = PDFPlumberLoader(path)
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = improved_clean(chunk.page_content)
    return chunks

chunks = load_and_split(path)

def save_chunks(chunks, output_file="doc_chunks.json"):
    chunk_data = [
        {"id": i, "page_content": doc.page_content, "metadata": doc.metadata}
        for i, doc in enumerate(chunks)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=4)

    print(f" Saved {len(chunk_data)} chunks to {output_file}")


print(f"Total chunks: {len(chunks)}\n")
for idx, doc in enumerate(chunks[:50]):
    print(f"--- Document Chunk {idx+1} ---")
    print(doc.page_content)
    print(f"Chunk length: {len(doc.page_content)}\n")
    save_chunks(chunks)