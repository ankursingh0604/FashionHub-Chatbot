import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="FashionHub Chatbot",
    page_icon="🛍️",
    layout="centered"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

fashion_hub_data = [
    "FAQ: What is the return policy? We offer a 30-day return policy for all unworn items with tags attached.",
    "FAQ: How long does shipping take? Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
    "FAQ: Do you ship internationally? Yes, globally! International shipping takes 7-14 business days.",
    "Product: Floral Summer Dress. A lightweight, breathable cotton dress with a vibrant floral print. Perfect for beach days and summer parties. Price: $45.",
    "Product: Classic Denim Jacket. A versatile, distressed blue denim jacket that pairs well with almost anything. Price: $60.",
    "Product: Wide-Brimmed Straw Hat. Offers excellent sun protection while keeping you stylish. Great for outdoor events. Price: $25.",
    "Product: Leather Ankle Boots. Sleek black leather boots with a slight heel. Great for evening wear or cooler weather. Price: $85.",
    "Product: White Linen Button-Down Shirt. A crisp, clean linen shirt that is both casual and elegant. Price: $40."
]

@st.cache_resource
def init_faiss():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(fashion_hub_data)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index

def ask_fashion_assistant(user_query, model, index):
    query_embedding = model.encode([user_query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = index.search(query_embedding, k=3)
    retrieved_chunks = [fashion_hub_data[i] for i in indices[0]]
    context_string = "\n".join(retrieved_chunks)
    
    system_instruction = (
        "You are a stylish and helpful fashion assistant for 'Fashion Forward Hub'.\n"
        "Use the provided context to answer the customer's question.\n"
        "If they ask for a look or style advice, combine the retrieved products to make an outfit.\n"
        "CRITICAL: Only suggest products that are explicitly named in the context. Do not invent products."
    )
    
    user_prompt = f"Context:\n{context_string}\n\nCustomer Question: {user_query}\n\nAssistant Answer:"
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.6
    )
    
    return completion.choices[0].message.content

st.title("🛍️ FashionHub Assistant")
st.markdown("*Your personal AI fashion advisor*")
st.divider()

model, index = init_faiss()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about our products or get outfit suggestions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_fashion_assistant(prompt, model, index)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})