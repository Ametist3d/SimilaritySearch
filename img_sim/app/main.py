from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager
import os
import logging
import numpy as np
import faiss
import torch

from typing import List
from request import ImageSearchRequest

from img_sim_search import get_image_sim_search, setup_image_sim_search

# Import shared configuration for image dataset
from config import DATASET_DIR, EMBEDDINGS_FILE

app_logger = logging.getLogger("api")
index: faiss.Index = None
image_files: List[str] = None

# Initialize the image similarity search instance.
setup_image_sim_search()
image_sim_search = get_image_sim_search()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, image_files
    try:
        # List image files from the dataset directory.
        image_files = [
            os.path.join(DATASET_DIR, f)
            for f in os.listdir(DATASET_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Load or create embeddings.
        if os.path.exists(EMBEDDINGS_FILE):
            app_logger.info("Loading existing embeddings...")
            data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
            embeddings = data["embeddings"]
            image_files = data["image_files"]
        else:
            app_logger.info("Creating new embeddings...")
            embeddings, image_files = image_sim_search.process_image_directory(
                DATASET_DIR
            )
            np.savez(EMBEDDINGS_FILE, embeddings=embeddings, image_files=image_files)

        # Build FAISS index.
        app_logger.info("Building FAISS index...")
        index = image_sim_search.create_faiss_index(embeddings)

        # Assign the index and image_files to the instance.
        image_sim_search.index = index
        image_sim_search.image_files = image_files

    except Exception as e:
        app_logger.error(f"Initialization failed: {str(e)}")
        raise

    yield

    # Cleanup.
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def image_search(request: ImageSearchRequest):
    if not request.image:
        raise HTTPException(status_code=400, detail="Image cannot be empty")

    try:
        # Process the input image (from URL or base64) using the helper method.
        pil_image = image_sim_search.process_input_image(
            request.image, request.is_base64
        )
        # Extract the embedding from the processed PIL image.
        query_embedding = image_sim_search.get_embedding_from_image(pil_image).astype(
            np.float32
        )
        # Search for similar images.
        results = image_sim_search.search_similar(
            np.expand_dims(query_embedding, axis=0), request.k
        )
        return {"results": results}
    except Exception as e:
        app_logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search processing failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
