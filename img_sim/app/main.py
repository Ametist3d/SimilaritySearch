from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager
import os
import logging
import numpy as np
import faiss

from typing import List
import torch

from request import ImageSearchRequest

from img_sim_search import (
    get_embedding,
    embedding_model,
    preprocess,
    process_image,
    device,
)

app_logger = logging.getLogger("api")
index: faiss.Index = None
image_files: List[str] = None

# Configuration
CLASSIFICATION_THRESHOLD = 0.7
EMBEDDINGS_FILE = "embeddings_dataset.npz"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, image_files

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        images_folder = os.path.join(BASE_DIR, "..", "..", "_DS", "dataset")
        image_files = [
            os.path.join(images_folder, f)
            for f in os.listdir(images_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Load or create embeddings
        if os.path.exists(EMBEDDINGS_FILE):
            app_logger.info("Loading existing embeddings...")
            data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
            embeddings = data["embeddings"]
            image_files = data["image_files"]
        else:
            app_logger.info("Creating new embeddings...")
            embeddings = []
            for path in image_files:
                embeddings.append(get_embedding(path))
            embeddings = np.stack(embeddings)
            np.savez(EMBEDDINGS_FILE, embeddings=embeddings, image_files=image_files)

        # Build FAISS index
        app_logger.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))

    except Exception as e:
        app_logger.error(f"Initialization failed: {str(e)}")
        raise

    yield

    # Cleanup
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def image_search(request: ImageSearchRequest):
    if not request.image:
        raise HTTPException(status_code=400, detail="Image cannot be empty")

    try:
        # Process input image
        pil_image = process_image(request.image, request.is_base64)
        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = embedding_model(image_tensor)
        query_embedding = embedding.cpu().view(-1).numpy().astype(np.float32)

        # Search index
        distances, indices = index.search(
            np.expand_dims(query_embedding, axis=0), request.k
        )

        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(image_files):
                continue  # Skip invalid indices
            results.append(
                {
                    "image_path": image_files[idx],
                    "similarity_score": float(
                        1 / (1 + distances[0][i])
                    ),  # Convert distance to similarity
                }
            )

        return {"results": results[: request.k]}

    except Exception as e:
        app_logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search processing failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
