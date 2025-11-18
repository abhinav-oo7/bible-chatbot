import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import google.generativeai as genai
import urllib.request

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Bible Chatbot",
    page_icon="📖",
    layout="wide"
)

# ---------------------------
# Load API Key & Models (Cached)
# ---------------------------
@st.cache_resource
def load_models():
    """Load and configure API keys and models once."""
    
    # On Render, we use Environment Variables directly
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Please add it to your Render Environment Variables.")
        st.stop()
        
    genai.configure(api_key=api_key)
    
    # Use the fast "flash" model for the chatbot
    generative_model = genai.GenerativeModel('gemini-flash-latest')
    
    # Use the embedding model
    embedding_model = 'models/embedding-001'
    return generative_model, embedding_model

# ---------------------------
# Load Data & FAISS Index (Cached)
# ---------------------------
@st.cache_resource
def load_faiss_index():
    """Load the Bible data and the FAISS index from disk."""
    
    # This matches the filename defined in render_build.sh
    embeddings_file = "gemini_bible_embeddings_v2.npy"
    csv_file = "KJV.csv"
    
    # Fallback URL (Safety net if build script fails)
    FILE_URL = "https://github.com/abhinav-oo7/bible-chatbot/releases/download/v1/gemini_bible_embeddings.npy"

    # Check if file exists (It should be there from render_build.sh)
    if not os.path.exists(embeddings_file):
        with st.spinner(f"Downloading database (184MB)..."):
            try:
                urllib.request.urlretrieve(FILE_URL, embeddings_file)
                st.success("Download complete!")
            except Exception as e:
                st.error(f"Error downloading file: {e}")
                st.stop()

    if not os.path.exists(csv_file):
        st.error(f"Missing required file: {csv_file}")
        st.stop()
        
    try:
        # Load data
        bible_df = pd.read_csv(csv_file)
        bible_df['Text'] = bible_df['Text'].astype(str)
        embeddings = np.load(embeddings_file)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        return bible_df, index
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# --- Load everything ---
generative_model, embedding_model = load_models()
bible_df, faiss_index = load_faiss_index()

# ---------------------------
# Semantic Search Function
# ---------------------------
def search_bible(query, top_k=5):
    response = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )
    query_emb = np.array([response['embedding']]).astype('float32')
    _, idx = faiss_index.search(query_emb, top_k)
    results = bible_df.iloc[idx[0]]
    return results

# ---------------------------
# Generate Answer Function
# ---------------------------
def get_chatbot_response(question):
    top_verses = search_bible(question, top_k=5)

    context = "\n".join(
        top_verses['Book'] + " " +
        top_verses['Chapter'].astype(str) + ":" +
        top_verses['Verse'].astype(str) + " - " +
        top_verses['Text']
    )

    prompt = f"""You are a helpful and knowledgeable Bible assistant.
Your goal is to answer the user's question using the context below.

Context:
{context}

Question: {question}
Answer:"""

    try:
        response_iterator = generative_model.generate_content(prompt, stream=True)
        
        # Generator to yield text chunks for Streamlit
        def stream_generator(iterator):
            for chunk in iterator:
                if chunk.text:
                    yield chunk.text
                    
        return stream_generator(response_iterator)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("📖 Bible Chatbot")

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
