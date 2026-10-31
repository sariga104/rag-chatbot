"""Streamlit UI for the 'Chat with Your Documents' chatbot.

Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from src.rag import DocumentStore, answer_question

load_dotenv()  # loads ANTHROPIC_API_KEY from your .env file

st.set_page_config(page_title="Chat with Your Documents", page_icon="📄")
st.title("📄 Chat with Your Documents")
st.caption("Upload a PDF, then ask questions about it.")

# --- Sidebar: upload a PDF ---
uploaded = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded and st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading and indexing your document..."):
        try:
            store = DocumentStore()
            chunk_count = store.add_pdf(uploaded)
        except ValueError as error:
            st.sidebar.error(str(error))
            st.stop()
    st.session_state.store = store
    st.session_state.file_name = uploaded.name
    st.session_state.messages = []  # start a fresh chat for the new document
    st.sidebar.success(f"Indexed {chunk_count} chunks from {uploaded.name}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Show the chat so far ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle a new question ---
if question := st.chat_input("Ask a question about your document"):
    if "store" not in st.session_state:
        st.warning("Please upload a PDF in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                sources = st.session_state.store.search(question)
                answer = answer_question(
                    question, sources, st.session_state.messages
                )
            except Exception as error:
                st.error(f"Something went wrong: {error}")
                st.stop()

        st.markdown(answer)
        with st.expander("Sources used"):
            for source in sources:
                st.markdown(f"**Page {source['page']}**")
                st.caption(source["text"])

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
