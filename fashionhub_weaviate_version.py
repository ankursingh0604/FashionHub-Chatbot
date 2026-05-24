import os
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from groq import Groq

GROQ_API_KEY = ""  # Get free API key from console.groq.com
HF_API_KEY = ""    # Get free API key from huggingface.co/settings/tokens

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

# Connecting to WeAviate
client = weaviate.connect_to_embedded(
    headers={
        "X-Huggingface-Api-Key": HF_API_KEY
    }
)

try:
    # Creating Collection
    if client.collections.exists("FashionHub"):
        client.collections.delete("FashionHub")

    collection = client.collections.create(
        name="FashionHub",
        # Configures Weaviate to use a free, fast open-source embedding model hosted by Hugging Face
        vectorizer_config=Configure.Vectorizer.text2vec_huggingface(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="type", data_type=DataType.TEXT)
        ]
    )

    # Loading and indexing the data
    print("Indexing mock fashion data into local vector store...")
    collection.data.insert_many(fashion_hub_data)
    print("Data indexed successfully!")

    # RAG PIPELINE FUNCTION
    def ask_fashion_assistant(user_query: str):
        # Performing vector similarity search
        search_response = collection.query.near_text(
            query=user_query,
            limit=3
        )

        # Consolidating the context texts
        retrieved_chunks = [obj.properties["content"] for obj in search_response.objects]
        context_string = "\n".join(retrieved_chunks)

        # Setting up system behavior instruction rules
        system_instruction = (
            "You are a stylish and helpful fashion assistant for 'Fashion Forward Hub'.\n"
            "Use the provided context to answer the customer's question.\n"
            "If they ask for a look or style advice, combine the retrieved products to make an outfit.\n"
            "CRITICAL: Only suggest products that are explicitly named in the context. Do not invent products."
        )

        user_prompt = f"Context:\n{context_string}\n\nCustomer Question: {user_query}\n\nAssistant Answer:"

        # Routing to Groq's API for text completion
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



    query_faq = "What is your return policy if I wear something once?"
    print(f"User: {query_faq}")
    print(f"Bot: {ask_fashion_assistant(query_faq)}\n")

    query_style = "I need an outfit setup for an afternoon outdoor party, any suggestions?"
    print(f"User: {query_style}")
    print(f"Bot: {ask_fashion_assistant(query_style)}")

finally:

    client.close()