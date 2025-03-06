import os
import pandas as pd
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class TextSimSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        self.index: faiss.Index = None
        self.texts: List[str] = None

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Loads the CSV file using fallback encodings if needed."""
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

    @staticmethod
    def combine_all_columns(row) -> str:
        """Combines all non-null column values into a single string."""
        vals = []
        for val in row.values:
            if pd.notnull(val):
                vals.append(str(val))
        return " ".join(vals)

    def build_embeddings(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into embeddings using the SentenceTransformer model."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.astype(np.float32)

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Creates a FAISS index from the embeddings."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def setup(self, csv_path: str, index_file_path: str) -> None:
        """Loads the CSV, builds or loads the FAISS index, and sets the text list."""
        df = self.load_data(csv_path)
        df["combined_text"] = df.apply(TextSimSearch.combine_all_columns, axis=1)
        self.texts = df["combined_text"].tolist()

        if os.path.exists(index_file_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(index_file_path)
            if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
                print("Moving index to GPU...")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            print("Building new FAISS index...")
            embeddings = self.build_embeddings(self.texts)
            self.index = self.build_index(embeddings)
            # Save the index as CPU index even if built on GPU.
            if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_file_path)
            else:
                faiss.write_index(self.index, index_file_path)

    def search_similar(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Searches the FAISS index for texts similar to the query."""
        query_emb = self.model.encode([query_text], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_emb, k)
        results = []
        for i, idx in enumerate(indices[0]):
            text = self.texts[idx] if self.texts is not None else f"ID: {idx}"
            results.append((text, distances[0][i]))
        return results


# Global instance for shared usage
TEXT_SIMSEARCH_INSTANCE: TextSimSearch = None

def get_text_sim_search() -> TextSimSearch:
    return TEXT_SIMSEARCH_INSTANCE

def setup_text_sim_search(csv_path: str, index_file_path: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    global TEXT_SIMSEARCH_INSTANCE
    TEXT_SIMSEARCH_INSTANCE = TextSimSearch(model_name=model_name)
    TEXT_SIMSEARCH_INSTANCE.setup(csv_path, index_file_path)
