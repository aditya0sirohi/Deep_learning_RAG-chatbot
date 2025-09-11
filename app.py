# app.py
import gradio as gr
from rag_engine import rag_pipeline

def answer_question(user_input):
    if not user_input or user_input.strip() == "":
        return "Please ask a question."
    return rag_pipeline(user_input)

title = "📘 Deep Learning RAG Chatbot"
description = "Ask about deep learning. Data loaded from your FAISS index/doc_chunks.json."

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)

    with gr.Row():
        txt = gr.Textbox(label="Ask me about deep learning...", placeholder="Type your question here", lines=2)
        btn = gr.Button("Ask")

    output = gr.Textbox(label="Answer", lines=15)

    btn.click(answer_question, inputs=txt, outputs=output)
    txt.submit(answer_question, inputs=txt, outputs=output)

if __name__ == "__main__":
    demo.launch()
