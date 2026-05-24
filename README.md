# FashionHub Chatbot 🛍️

A RAG-powered fashion assistant chatbot built with Weaviate and Groq.

## Features
- Answers FAQ questions about shipping, returns etc.
- Suggests outfit combinations from product catalog
- Hallucination prevention — only suggests real products

## Tech Stack
- Python
- Weaviate (vector database)
- Groq API (Llama 3.1 8B)
- Sentence Transformers

## Setup
1. Get free API keys from console.groq.com and huggingface.co
2. Add keys to the code
3. pip install weaviate-client groq
4. python fashionhub.py

## Live Demo
🚀 [Try it here](https://fashionapp-chatbot-8jxhhcbfatf56u7wqhfktl.streamlit.app/)