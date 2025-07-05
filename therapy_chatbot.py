import os
import openai
import streamlit as st
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

PERSIST_DIR = "./storage/VectorStoreIndex/"

# Streamlit UI
st.title("💬 Trauma-Sensitive Chatbot")
st.caption("Supporting emotional reflection with curated counselor responses")

# Sidebar API input
with st.sidebar:
    openai_api_key = st.text_input("🔐 OpenAI API Key", key="api_key", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        st.success("API key set!")
    else:
        st.warning("Please add your OpenAI API key to continue.")
        st.stop()

# Load index (from disk)
try:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "You are a trauma-sensitive counseling assistant.\n"
            "Use calm, empathetic, and supportive language.\n"
            "Refer to retrieved counselor responses as guidance, not templates."
        ),
        verbose=True
    )
except Exception as e:
    st.error(f"Could not load index: {e}")
    st.stop()

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I'm here to support your emotional journey. How can I help you today?"}
    ]

# User Input
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking empathetically..."):
            try:
                reply = chat_engine.chat(st.session_state.messages[-1]["content"])
                st.write(reply.response)
                st.session_state.messages.append({"role": "assistant", "content": reply.response})
            except Exception as e:
                st.error(f"Chat generation failed: {e}")


