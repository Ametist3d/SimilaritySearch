from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager
import os
import logging
import numpy as np
import faiss

from typing import List
import torch
from request import AudioSearchRequest

# from audio_sim_search import (
#     wav_to_embedding,
#     process_audio_directory,
#     create_faiss_index,
#     search_similar,
#     model,
#     feature_extractor,
#     device,
# )
from audio_sim_search import get_audio_sim_search, setup_audio_sim_search

# Import shared configuration
from config import DATASET_DIR, EMBEDDINGS_FILE

app_logger = logging.getLogger("api")
index: faiss.Index = None
audio_files: List[str] = None
setup_audio_sim_search()
audio_sim_search = get_audio_sim_search()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, audio_files

    try:
        # Get audio files from the dataset directory
        audio_files = [
            os.path.join(DATASET_DIR, f)
            for f in os.listdir(DATASET_DIR)
            if f.lower().endswith(".wav")
        ]

        # Load or create embeddings
        if os.path.exists(EMBEDDINGS_FILE):
            app_logger.info("Loading existing embeddings...")
            data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
            embeddings = data["embeddings"]
            audio_files = data["audio_files"]
        else:
            app_logger.info("Creating new embeddings...")
            embeddings, audio_files = audio_sim_search.process_audio_directory(
                DATASET_DIR
            )
            np.savez(EMBEDDINGS_FILE, embeddings=embeddings, audio_files=audio_files)

        # Build FAISS index
        app_logger.info("Building FAISS index...")
        index = audio_sim_search.create_faiss_index(embeddings)

    except Exception as e:
        app_logger.error(f"Initialization failed: {str(e)}")
        raise

    yield

    # Cleanup
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def audio_search(request: AudioSearchRequest):
    if not request.audio:
        raise HTTPException(status_code=400, detail="Audio cannot be empty")

    try:
        # Process input audio
        if request.is_base64:
            # Handle base64 audio (not implemented in this example)
            raise HTTPException(
                status_code=501, detail="Base64 audio not yet supported"
            )
        else:
            # Handle URL or file path
            query_embedding = audio_sim_search.wav_to_embedding(request.audio)
            if query_embedding is None:
                raise HTTPException(status_code=400, detail="Failed to process audio")

        # Assign the index and audio_files to the instance so they are used in search_similar
        audio_sim_search.index = index
        audio_sim_search.audio_files = audio_files

        # Search index
        results = audio_sim_search.search_similar(np.expand_dims(query_embedding, axis=0), request.k)

        return {"results": results}

    except Exception as e:
        app_logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search processing failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
