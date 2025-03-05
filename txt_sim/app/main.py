from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
import logging
import torch
import faiss
from txt_sim_search import (
    load_data,
    combine_all_columns,
    build_embeddings,
    build_index,
    search_similar,
    get_device,
)
from request import SearchRequest 
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

app_logger = logging.getLogger("api")
index: faiss.Index = None
texts: List[str] = None
model: SentenceTransformer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for FastAPI resources"""
    global index, texts, model
    
    # Initialize resources
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "..", "..", "_DS", "war-news.csv")
    index_filename = "war-news.index"
    index_file_path = os.path.join(BASE_DIR, "..", "..", index_filename)

    try:
        # Load data
        app_logger.info("Loading dataset...")
        df = load_data(csv_path)
        df["combined_text"] = df.apply(combine_all_columns, axis=1)
        texts = df["combined_text"].tolist()

        # Load/build index
        if os.path.exists(index_file_path):
            app_logger.info("Loading existing FAISS index...")
            index = faiss.read_index(index_file_path)
            if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
                app_logger.info("Moving index to GPU...")
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            app_logger.info("Building new index...")
            embeddings = build_embeddings(texts)
            index = build_index(embeddings)
            if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, index_file_path)
            else:
                faiss.write_index(index, index_file_path)

        # Initialize model
        app_logger.info("Loading sentence transformer model...")
        device = get_device()
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        model.eval()

    except Exception as e:
        app_logger.error(f"Initialization failed: {str(e)}")
        raise

    yield  # App is running

    # Cleanup
    app_logger.info("Cleaning up resources...")
    del model
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = search_similar(
            index=index,
            query_text=request.query,
            model=model,
            texts=texts,
            k=min(request.k, 20)
        )
        return {
            "query": request.query,
            "results": [
                {"text": text, "score": float(score)}
                for text, score in results
            ]
        }
    except Exception as e:
        app_logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search processing failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)