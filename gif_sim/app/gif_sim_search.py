import os
import base64
from io import BytesIO
from typing import List, Tuple
import logging
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from PIL import Image, ImageSequence
import requests
import tempfile
from config import DATASET_DIR, CACHE_DIR
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GifSimSearch:
    def __init__(self):
        """Initialize the GIF similarity search class using CLIP with cached weights."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the CLIP model and processor using CACHE_DIR.
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=CACHE_DIR
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=CACHE_DIR
        )
        self.clip_model.eval()
        # These will be set during dataset processing.
        self.index: faiss.Index = None
        self.gif_files: List[str] = None

    def extract_frames(
        self, gif_path: str, max_frames: int = 30, num_frames: int = 3
    ) -> List[Image.Image]:
        """
        Extract frames from a GIF file. First, load up to max_frames and then select
        num_frames evenly spaced frames.
        """
        frames = []
        try:
            with Image.open(gif_path) as gif:
                for idx, frame in enumerate(ImageSequence.Iterator(gif)):
                    if idx >= max_frames:
                        break
                    frames.append(frame.convert("RGB"))
        except Exception as e:
            logger.error(f"Failed to extract frames from {gif_path}: {str(e)}")
        # If more frames than needed, select evenly spaced frames.
        if len(frames) > num_frames:
            indices = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames

    def frame_to_embedding(self, frame: Image.Image) -> np.ndarray:
        """
        Compute an embedding for a single frame using CLIP.
        """
        inputs = self.clip_processor(
            images=frame, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        return outputs.squeeze().cpu().numpy()

    def aggregate_embeddings(self, frame_embeddings: List[np.ndarray]) -> np.ndarray:
        """Aggregate a list of embeddings by averaging."""
        return np.mean(frame_embeddings, axis=0)

    def gif_to_embedding(
        self, gif_path: str, max_frames: int = 30, num_frames: int = 3
    ) -> np.ndarray:
        """
        Extract frames from a GIF, compute CLIP embeddings for each, and return the aggregated embedding.
        """
        logger.info(f"Processing GIF: {gif_path}")
        frames = self.extract_frames(
            gif_path, max_frames=max_frames, num_frames=num_frames
        )
        if not frames:
            raise ValueError(f"No frames extracted from {gif_path}")
        frame_embeddings = []
        for frame in frames:
            try:
                emb = self.frame_to_embedding(frame)
                frame_embeddings.append(emb)
            except Exception as e:
                logger.error(
                    f"Error extracting embedding for a frame in {gif_path}: {str(e)}"
                )
        gif_embedding = self.aggregate_embeddings(frame_embeddings)
        return gif_embedding

    def process_gif_directory(
        self, gif_folder: str, max_frames: int = 30, num_frames: int = 3
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process all GIF files in a directory and return their embeddings and filenames.
        A progress bar is displayed for visual feedback.
        """
        gif_files = [
            os.path.join(gif_folder, f)
            for f in os.listdir(gif_folder)
            if f.lower().endswith(".gif")
        ]
        logger.info(f"Found {len(gif_files)} GIF files in {gif_folder}")
        embeddings = []
        valid_files = []
        for gif_file in tqdm(gif_files, desc="Processing GIFs"):
            try:
                emb = self.gif_to_embedding(
                    gif_file, max_frames=max_frames, num_frames=num_frames
                )
                embeddings.append(emb)
                valid_files.append(os.path.basename(gif_file))
            except Exception as e:
                logger.error(f"Error processing {gif_file}: {str(e)}")
        embeddings = np.array(embeddings)
        return embeddings, valid_files

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a FAISS index from the given embeddings."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        # pylint: disable-next = no-value-for-parameter
        index.add(embeddings.astype(np.float32))
        return index

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """Search the FAISS index for GIFs similar to the query embedding."""
        if self.index is None or self.gif_files is None:
            raise ValueError(
                "FAISS index and GIF files must be initialized before searching."
            )
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.gif_files):
                continue
            # pylint: disable = unsubscriptable-object
            results.append(
                {
                    "gif_path": os.path.normpath(
                        os.path.join(DATASET_DIR, self.gif_files[idx])
                    ),
                    "similarity_score": float(1 / (1 + distances[0][i])),
                }
            )
            # pylint: enable = unsubscriptable-object
        return results

    def process_input_gif(self, input_data: str, is_base64: bool) -> str:
        """
        Process an input GIF from a URL or base64 string.
        Save it to a uniquely named temporary file and return the file path.
        """
        # Create a unique temporary file. The file is not automatically deleted.
        temp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Close the file so we can write to it.

        if is_base64:
            if "base64," in input_data:
                input_data = input_data.split("base64,")[1]
            gif_data = base64.b64decode(input_data)
            with open(temp_path, "wb") as f:
                f.write(gif_data)
        else:
            response = requests.get(input_data, timeout=10)
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(response.content)
        return temp_path


# Global instance for shared usage.
GIF_SIMSEARCH_INSTANCE = None


def get_gif_sim_search():
    return GIF_SIMSEARCH_INSTANCE


def setup_gif_sim_search():
    global GIF_SIMSEARCH_INSTANCE
    GIF_SIMSEARCH_INSTANCE = GifSimSearch()
