import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import google.generativeai as genai

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Bible Chatbot",
    page_icon="📖",
    layout="wide"
)

# ---------------------------
# Load API Key & Models
# ---------------------------
@st.cache_resource
def load_config():
    """Load API keys and configure Gemini."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback for local testing if env var is missing
        # api_key = st.secrets.get("GOOGLE_API_KEY") 
        pass
        
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Please add it to your Render Environment Variables.")
        st.stop()
        
    genai.configure(api_key=api_key)
    
    # Models
    generative_model = genai.GenerativeModel('gemini-1.5-flash')
    embedding_model = 'models/embedding-001'
    return generative_model, embedding_model

# ---------------------------
# Load Data & Compressed Index
# ---------------------------
@st.cache_resource
def load_data():
    """Load the CSV and the pre-built compressed FAISS index."""
    
    index_file = "bible_compressed.index"
    csv_file = "KJV.csv" # Ensure this file is in your repo

    if not os.path.exists(index_file):
        st.error(f"Index file '{index_file}' not found. Please run build_compressed_index.py locally and upload the result.")
        st.stop()

    if not os.path.exists(csv_file):
        st.error(f"CSV file '{csv_file}' not found.")
        st.stop()
        
    try:
        # 1. Load Text Data
        # Optimization: Only load necessary columns to save RAM
        bible_df = pd.read_csv(csv_file, usecols=['Book', 'Chapter', 'Verse', 'Text'])
        bible_df['Text'] = bible_df['Text'].astype(str)
        
        # 2. Load FAISS Index (IVF-PQ)
        # This loads the tiny compressed file directly
        index = faiss.read_index(index_file)
        
        # Set nprobe (how many clusters to search). 
        # Higher = more accurate but slower. 10 is a sweet spot.
        index.nprobe = 10
        
        return bible_df, index
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# --- Initialize ---
generative_model, embedding_model = load_config()
bible_df, faiss_index = load_data()

# ---------------------------
# Logic
# ---------------------------
def search_bible(query, top_k=5):
    # Embed the query
    response = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )
    query_emb = np.array([response['embedding']]).astype('float32')
    
    # Search the compressed index
    distances, indices = faiss_index.search(query_emb, top_k)
    
    # Fetch results from DataFrame
    results = bible_df.iloc[indices[0]]
    return results

def get_chatbot_response(question):
    top_verses = search_bible(question, top_k=5)

    context = "\n".join(
        f"{row['Book']} {row['Chapter']}:{row['Verse']} - {row['Text']}"
        for _, row in top_verses.iterrows()
    )

    prompt = f"""You are a helpful and knowledgeable Bible assistant.
Your goal is to answer the user's question using ONLY the context provided below.
If the context doesn't contain the answer, gently say so.

Context:
{context}

Question: {question}
Answer:"""

    try:
        response = generative_model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# ---------------------------
# UI
# ---------------------------
st.title("📖 Bible Chatbot")
st.caption("Powered by Gemini Flash & FAISS IVF-PQ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your Bible question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_stream = get_chatbot_response(prompt)
        if response_stream:
            full_response = st.write_stream(response_stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.write("Sorry, I had trouble getting a response.")
