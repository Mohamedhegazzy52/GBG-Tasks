# app.py — Streamlit UI for CV RAG Chatbot
# Run: streamlit run app.py
import os
from dotenv import load_dotenv
import streamlit as st
from rag import build_pipeline



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CV RAG Chatbot", page_icon="🤖", layout="centered")

st.title("Chat with CVs")
st.caption("Powered by Gemini 2.5 Flash")

# ── Load pipeline once ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CVs...")
def load_pipeline():
    return build_pipeline()

try:
    rag_chain, cv_files, num_chunks = load_pipeline()
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# ── Sidebar — info only ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Loaded CVs")
    for f in cv_files:
        st.markdown(f"- `{f}`")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat state ────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Display history ───────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about the CVs...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = rag_chain.invoke(user_input)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
