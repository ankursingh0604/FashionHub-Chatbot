# FashionHub Chatbot 🛍️

A RAG-powered fashion assistant chatbot built with FAISS, Groq and Streamlit.

## Features
- Answers FAQ questions about shipping, returns etc.
- Suggests outfit combinations from product catalog
- Hallucination prevention — only suggests real products

## Tech Stack
- Python
- FAISS (vector search)
- Groq API (Llama 3.1 8B)
- Sentence Transformers
- Streamlit

## Setup
1. Get free API key from console.groq.com
2. Add key to Streamlit secrets
3. pip install faiss-cpu sentence-transformers groq streamlit
4. streamlit run app.py

## Live Demo
🚀 [Try it here](https://fashionapp-chatbot-8jxhhcbfatf56u7wqhfktl.streamlit.app/)