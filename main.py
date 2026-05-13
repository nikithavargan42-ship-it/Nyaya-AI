import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from mtranslate import translate

# --- CONFIG ---
st.set_page_config(page_title="Nyaya-AI", page_icon="⚖️")
GROQ_API_KEY = st.secrets["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"]# Get from console.groq.com

# --- LOAD DATA ---
@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="vectorstore_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = get_retriever()
# Using ChatOpenAI as a bridge for Groq (very stable)
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# --- UI ---
st.title("⚖️ Nyaya-AI")
st.caption("Digital Legal Assistant for Indian Citizen")

langs = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta", "Gujarati": "gu"}
target_lang = st.sidebar.selectbox("Choose Language", list(langs.keys()))

if user_input := st.chat_input("Ask a legal question..."):
    st.chat_message("user").write(user_input)
    
    with st.spinner("Analyzing laws..."):
        # Translate to English
        eng_q = translate(user_input, "en")
        
        # Search Law
        docs = retriever.invoke(eng_q)
        context = "\n\n".join([d.page_content for d in docs])
        
        # AI Logic
        prompt = f"Using this law context: {context}\n\nQuestion: {eng_q}\nAnswer:"
        eng_response = llm.invoke(prompt).content
        
        # Translate back
        final_ans = translate(eng_response, langs[target_lang])
        
        with st.chat_message("assistant"):
            st.write(final_ans)
            if langs[target_lang] != "en":
                st.info(f"English: {eng_response[:200]}...")
