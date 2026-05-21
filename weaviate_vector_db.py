import weaviate
import weaviate.classes as wvc
import numpy as np
from sentence_transformers import SentenceTransformer

bbc_news_dataset = [
    {
        "title": "UK Economy Sees Unexpected Growth Boost",
        "body": "The UK economy expanded by 0.4% this quarter, outperforming initial central bank forecasts. Tech sector investments and increased consumer spending on services drove the surge. Policymakers caution that inflation risks remain persistent despite the positive gross domestic product numbers."
    },
    {
        "title": "Breakthrough in Quantum Computing Scalability",
        "body": "Researchers have successfully demonstrated a new error-correction method for silicon-based qubits. This breakthrough allows quantum processors to maintain coherence for significantly longer periods, bringing commercial quantum applications closer to reality over the next decade."
    }
]
def chunk_text(text, chunk_size=20, overlap=5):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
client = weaviate.connect_to_embedded() # Connect to weaviate

try:
  collection_name = "BBC"
  if client.collections.exists(collection_name):
    client.collections.delete(collection_name)
  
  collection = client.collections.create(name=collection_name)

  # Configuration done, now proceeding with loading and indexing data with batch process
  with collection.batch.dynamic() as batch:
    for doc in bbc_news_dataset:
      chunks=chunk_text(doc["body"])

      embeddings = model.encode(chunks)
      for chunk_str, vector in zip(chunks, embeddings):
                # Insert both textual metadata and the mathematical vector
                batch.add_object(
                    properties={
                        "content": chunk_str,
                        "source_title": doc["title"]
                    },
                    vector=vector.tolist()
                )
  user_query = "Tell me about quantum error correction scalability"
  query_vector = model.encode(user_query).tolist() # It converts user_query into same vector space
  
  results = collection.query.near_vector(
        near_vector=query_vector,
        limit=2,
        return_metadata=wvc.query.MetadataQuery(distance=True)
    ) #HNSW indexing

  for i, obj in enumerate(results.objects):
        print(f"\nMatch #{i+1} (Distance: {obj.metadata.distance:.4f})")
        print(f"Source Document: {obj.properties['source_title']}")
        print(f"Retrieved Content: \"{obj.properties['content']}\"")
finally:
  client.close()