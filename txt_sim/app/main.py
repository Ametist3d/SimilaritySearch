from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
import uvicorn
import logging
import torch
from request import SearchRequest
from txt_sim_search import get_text_sim_search, setup_text_sim_search
import config

app_logger = logging.getLogger("api")
text_sim_search = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_sim_search
    try:
        # Use dataset directory and embeddings file from config
        dataset_dir = config.DATASET_DIR
        embeddings_file = config.EMBEDDINGS_FILE

        # Find any CSV file in the dataset directory (Txt_DS)
        csv_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
        if not csv_files:
            raise Exception("No CSV file found in the dataset folder")
        csv_file = csv_files[0]
        csv_path = os.path.join(dataset_dir, csv_file)

        app_logger.info(f"Using CSV file: {csv_path}")
        app_logger.info(f"Index file path: {embeddings_file}")

        # Setup the text similarity search instance using the CSV and index file from config
        setup_text_sim_search(csv_path, embeddings_file)
        text_sim_search = get_text_sim_search()
    except Exception as e:
        app_logger.error(f"Initialization failed: {str(e)}")
        raise

    yield  # App is running

    app_logger.info("Cleaning up resources...")
    del text_sim_search
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        results = text_sim_search.search_similar(
            query_text=request.query, k=min(request.k, 20)
        )
        return {
            "query": request.query,
            "results": [
                {"text": text, "score": float(score)} for text, score in results
            ],
        }
    except Exception as e:
        app_logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Search processing failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
