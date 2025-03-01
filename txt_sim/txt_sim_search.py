import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Step 1: Load the dataset from CSV
df = pd.read_csv('data.csv')  # Ensure 'data.csv' is in your working directory
texts = df['text'].tolist()    # Replace 'text' with your column name

# Step 2: Extract embeddings using a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Step 3: Build a FAISS index for similarity search
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance is common for cosine similarity (after normalization) or Euclidean similarity
index.add(embeddings)  # Add our text embeddings to the index

# Step 4: Define a function to search for similar texts
def search_similar(query, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = [texts[i] for i in indices[0]]
    return results

# Example query
query = "Your sample query text here"
similar_texts = search_similar(query)
print("Top similar texts:")
for i, text in enumerate(similar_texts):
    print(f"{i+1}. {text}")
