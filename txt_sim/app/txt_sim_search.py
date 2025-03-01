import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_data(csv_path: str) -> pd.DataFrame:
    """Loads the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    return df

def build_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Converts a list of texts into embeddings using a pre-trained model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype(np.float32)  # FAISS expects float32

def build_index(embeddings: np.ndarray):
    """Builds a FAISS index from the embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Using L2 distance
    #pylint: disable-next=no-value-for-parameter
    index.add(embeddings)
    return index

def search_similar(index, query_text, model_name="all-MiniLM-L6-v2", texts=None, k=5):
    """Given a query text, returns top k similar texts along with their distances."""
    model = SentenceTransformer(model_name)
    query_emb = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_emb, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        distance = distances[0][i]
        text_match = texts[idx] if texts is not None else f"ID: {idx}"
        results.append((text_match, distance))
    return results
