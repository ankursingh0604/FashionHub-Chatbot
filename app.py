import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# Page Config

st.set_page_config(
    page_title="FashionHub Chatbot",
    page_icon="🛍️",
    layout="centered"
)

# API Keys

GROQ_API_KEY = "# Get free key from console.groq.com" 
 
# Fashion Data

fashion_hub_data = [
    {"content": "FAQ: What is the return policy? We offer a 30-day return policy for all unworn items with tags attached.", "type": "faq"},
    {"content": "FAQ: How long does shipping take? Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.", "type": "faq"},
    {"content": "FAQ: Do you ship internationally? Yes, globally! International shipping takes 7-14 business days.", "type": "faq"},
    {"content": "Product: Floral Summer Dress. A lightweight, breathable cotton dress with a vibrant floral print. Perfect for beach days and summer parties. Price: $45.", "type": "product"},
    {"content": "Product: Classic Denim Jacket. A versatile, distressed blue denim jacket that pairs well with almost anything. Price: $60.", "type": "product"},
    {"content": "Product: Wide-Brimmed Straw Hat. Offers excellent sun protection while keeping you stylish. Great for outdoor events. Price: $25.", "type": "product"},
    {"content": "Product: Leather Ankle Boots. Sleek black leather boots with a slight heel. Great for evening wear or cooler weather. Price: $85.", "type": "product"},
    {"content": "Product: White Linen Button-Down Shirt. A crisp, clean linen shirt that is both casual and elegant. Price: $40.", "type": "product"}
]

# Initialize ChromaDB + Load Data

@st.cache_resource
def init_chromadb():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.Client()
    
    collection = client.get_or_create_collection(name="fashionhub")
    
    # Load data if empty
    if collection.count() == 0:
        contents = [item["content"] for item in fashion_hub_data]
        embeddings = model.encode(contents).tolist()
        ids = [f"doc_{i}" for i in range(len(contents))]
        
        collection.add(
            documents=contents,
            embeddings=embeddings,
            ids=ids
        )
    
    return client, model

# RAG Function

def ask_fashion_assistant(user_query, client, model):
    collection = client.get_collection("fashionhub")
    
    # Encode query
    query_embedding = model.encode(user_query).tolist()
    
    # Retrieve top 3 chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    retrieved_chunks = results["documents"][0]
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

# Streamlit UI

st.title("🛍️ FashionHub Assistant")
st.markdown("*Your personal AI fashion advisor*")
st.divider()

# Initialize
client, model = init_chromadb()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about our products or get outfit suggestions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_fashion_assistant(prompt, client, model)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})