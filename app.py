import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
# load_dotenv is no longer needed for Vercel, but safe to keep
from dotenv import load_dotenv 
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
    
    # On Vercel, the key comes from Environment Variables
    # On your local machine, you can still use a .env file
    load_dotenv() 
    api_key = os.getenv("GOOGLE_API_KEY") 
    
    if not api_key:
        # Updated error message for Vercel
        st.error("GOOGLE_API_KEY not found. Please add it to your Vercel Project Environment Variables.")
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
    
    # This name is the local file name Vercel will create
    embeddings_file = "gemini_bible_embeddings_v2.npy"
    csv_file = "KJV.csv"
    
    # This is the public URL to your GitHub Release file
    FILE_URL = "https://github.com/abhinav-oo7/bible-chatbot/releases/download/v1/gemini_bible_embeddings.npy"

    # Check if the embedding file exists, if not, download it
    # This will run the first time the Vercel instance starts
    if not os.path.exists(embeddings_file):
        with st.spinner(f"Downloading {embeddings_file} (184MB)... This may take a moment."):
            try:
                urllib.request.urlretrieve(FILE_URL, embeddings_file)
                st.success("Download complete!")
            except Exception as e:
                st.error(f"Error downloading file: {e}")
                st.stop()
    # ----------------------------------------

    # Check if the CSV file exists (it should be in your Git repo)
    if not os.path.exists(csv_file):
        st.error(f"Missing required file: {csv_file}")
        st.stop()
        
    try:
        # Load the text data
        bible_df = pd.read_csv(csv_file)
        bible_df['Text'] = bible_df['Text'].astype(str)
        
        # Load the embeddings (which are now guaranteed to be downloaded)
        embeddings = np.load(embeddings_file)
        
        # Build the FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        print(f"FAISS index ready with {index.ntotal} verses.")
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
    """Embeds the query and searches the FAISS index."""
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
    """Searches for context and generates a streamed response."""
    top_verses = search_bible(question, top_k=5)

    # Prepare context
    context = "\n".join(
        top_verses['Book'] + " " +
        top_verses['Chapter'].astype(str) + ":" +
        top_verses['Verse'].astype(str) + " - " +
        top_verses['Text']
    )

    # --- THIS IS THE NEW, SMARTER PROMPT ---
    prompt = f"""You are a helpful and knowledgeable Bible assistant.
Your goal is to answer the user's question.

First, look at the 'Context' provided from the Bible.
- If the context is relevant and helps answer the question, please use it to form your answer.
- If the context is **not relevant** or **doesn't help**, you are allowed to use your own general Bible knowledge to provide a helpful response.
- For general conversation (like 'hi' or 'how are you'), just respond politely as an assistant.

Context:
{context}

Question: {question}
Answer:"""

    # --- THIS IS THE CHANGED PART ---
    try:
        # 1. Get the response iterator from Gemini
        response_iterator = generative_model.generate_content(prompt, stream=True)
        
        # 2. Define a new generator that yields *only* the text
        def stream_generator(iterator):
            for chunk in iterator:
                if chunk.text:  # Make sure the chunk has text
                    yield chunk.text
        
        # 3. Return the new, simplified generator
        return stream_generator(response_iterator)
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("📖 Bible Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask your Bible question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        # This creates the "streaming" effect
        response_stream = get_chatbot_response(prompt)
        if response_stream:
            # st.write_stream writes the chunks as they arrive
            full_response = st.write_stream(response_stream)
            # Add full response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.write("Sorry, I had trouble getting a response.")
