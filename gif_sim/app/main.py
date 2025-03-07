import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
import faiss
import torch

from request import ImageSearchRequest  # Reuse your request model
from gif_sim_search import get_gif_sim_search, setup_gif_sim_search
from config import DATASET_DIR, EMBEDDINGS_FILE

app_logger = logging.getLogger("api")
index: faiss.Index = None
gif_files: list = None

# Initialize the GIF similarity search instance.
setup_gif_sim_search()
gif_sim_search = get_gif_sim_search()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, gif_files
    try:
        # List GIF files from the dataset directory.
        gif_files = [
            os.path.join(DATASET_DIR, f)
            for f in os.listdir(DATASET_DIR)
            if f.lower().endswith(".gif")
        ]

        # Load or create embeddings.
        if os.path.exists(EMBEDDINGS_FILE):
            app_logger.info("Loading existing GIF embeddings...")
            data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
            embeddings = data["embeddings"]
            gif_files = data["gif_files"]
        else:
            app_logger.info("Creating new GIF embeddings...")
            embeddings, gif_files = gif_sim_search.process_gif_directory(
                DATASET_DIR, max_frames=30, num_frames=3
            )
            np.savez(EMBEDDINGS_FILE, embeddings=embeddings, gif_files=gif_files)

        # Build FAISS index.
        app_logger.info("Building FAISS index for GIFs...")
        index = gif_sim_search.create_faiss_index(embeddings)

        # Assign the index and file list to the instance.
        gif_sim_search.index = index
        gif_sim_search.gif_files = gif_files
    except Exception as e:
        app_logger.error(f"Initialization failed: {str(e)}")
        raise

    yield
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def gif_search(request: ImageSearchRequest):
    if not request.gif:
        raise HTTPException(status_code=400, detail="GIF cannot be empty")
    temp_path = None
    try:
        # Process the input GIF (from URL or base64) and get temporary file path.
        temp_path = gif_sim_search.process_input_gif(request.gif, request.is_base64)
        # Compute the aggregated embedding for the query GIF.
        query_embedding = gif_sim_search.gif_to_embedding(
            temp_path, max_frames=30, num_frames=3
        ).astype(np.float32)
        # Search for similar GIFs.
        results = gif_sim_search.search_similar(
            np.expand_dims(query_embedding, axis=0), request.k
        )
        return {"results": results}
    except Exception as e:
        app_logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search processing failed")
    finally:
        # Clean up temporary file.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
