#!/bin/bash

# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download the Embedding Database (so it's ready on disk)
echo "Downloading Embedding Database..."
curl -L "https://github.com/abhinav-oo7/bible-chatbot/releases/download/v1/gemini_bible_embeddings.npy" -o gemini_bible_embeddings_v2.npy
echo "Download Complete!"
