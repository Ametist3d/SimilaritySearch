import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_data(csv_path: str) -> pd.DataFrame:
    
    """Loads the CSV file into a pandas DataFrame using a fallback encoding."""
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        try:
            print("Default encoding failed. Trying 'latin-1' encoding...")
            df = pd.read_csv(csv_path, encoding="latin-1")
        except UnicodeDecodeError:
            print("latin-1 encoding failed. Trying 'cp1252' encoding...")
            df = pd.read_csv(csv_path, encoding="cp1252")

    return df

def combine_all_columns(row):
    # Convert each value to string, ignoring None or NaN
    vals = []
    for val in row.values:
        if pd.notnull(val):
            vals.append(str(val))
    return " ".join(vals)



def build_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """Converts a list of texts into embeddings using a pre-trained model on GPU if available."""
    device = get_device()
    print(f"Using device for embedding extraction: {device}")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype(np.float32)  # FAISS expects float32

def build_index(embeddings: np.ndarray):
    """Builds a FAISS index from the embeddings and transfers it to GPU if available."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Using L2 distance
    #pylint: disable-next=no-value-for-parameter
    index.add(embeddings)
    return index

@torch.inference_mode()
def search_similar(index, query_text, model=None, model_name="all-MiniLM-L6-v2", texts=None, k=5):
    """Search with inference mode decorator."""
    if model is None:
        device = get_device()
        model = SentenceTransformer(model_name, device=device)
        model.eval()
    
    query_emb = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_emb, k)
    
    return [
        (texts[idx] if texts else f"ID: {idx}", distances[0][i])
        for i, idx in enumerate(indices[0])
    ]
