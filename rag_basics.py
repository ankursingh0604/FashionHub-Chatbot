def retrieve_documents(query, corpus):
    query_words = set(query.lower().split())
    retrieved_docs = []

    for document in corpus:
        doc_words = set(document.lower().split())
        for word in query_words:
            if word in doc_words:
                retrieved_docs.append(document)
                break
    return retrieved_docs

def augmented_prompt(query,context):
    text_for_llm = f"Context:{context}, \nQuestion:{query}"
    return text_for_llm

def generate_message(augment_prompt, system_instruction="You are amazing"):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": augment_prompt}
    ]
    return messages

corpus = [
    "The capital of France is Paris.",
    "Quantum computing utilizes qubits.",
    "The Eiffel Tower is 330 meters tall."
]
user_question = "How tall is the Eiffel Tower?"

docs = retrieve_documents(user_question, corpus)
final_prompt = augmented_prompt(user_question, docs)
llm_input = generate_message(final_prompt, "You are an expert geographer.")

import json
print(json.dumps(llm_input, indent=2))