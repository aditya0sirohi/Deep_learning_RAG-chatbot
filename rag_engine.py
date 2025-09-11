import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

prompt_template = prompt_template = """
You are a knowledgeable assistant. 
Always provide an **in-depth answer with at least 5-8 sentences**. 
When possible, include **examples, reasoning, and context expansion**.
Otherwise:
- Reply: "Sorry, answer not found."

Context:
{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# def rag_pipeline(question, k=3):
#     docs = vectorstore.similarity_search(question, k=k)
#     context = "\n\n".join([doc.page_content for doc in docs])

#     prompt_text = prompt.format(context=context, question=question)

#     response = llm.invoke(prompt_text)

#     return response

def rag_pipeline(question, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    if "fuck you" in question.lower():
        return "Fuck you too :)."
    if not docs:
        return "Sorry, no relevant documents found."
    
    context = "\n\n".join([doc.page_content[:1000] for doc in docs])          #->In summary: This line takes the list of retrieved documents, extracts the first 1000 characters of text from each one, and then joins them all into a single string, with each document's text separated by a blank line.
    prompt_text = prompt.format(context=context, question=question)

    response = client.chat.completions.create(
        model="qwen/qwen3-4b:free",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=1
    )

    return response.choices[0].message.content

