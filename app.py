# import streamlit as st

# if "message_history" not in st.session_state:
#     st.session_state["message_history"] = []

# for message in st.session_state["message_history"]:
#     with st.chat_message(message["role"]):
#         st.text(message["content"])

# user_input = st.chat_input("Type your message here...")

# if user_input:
#     st.session_state["message_history"].append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.text(user_input)

#     st.session_state["message_history"].append({"role": "assistant", "content": "Hi there! How can I help you?"})
#     with st.chat_message("assistant"):
#         st.text("Hi there! How can I help you?")

import streamlit as st
from rag_engine import rag_pipeline

st.set_page_config(page_title="Deep Learning RAG Bot")
st.title("📘 Deep Learning Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask me about deep learning..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    response = rag_pipeline(user_query)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
