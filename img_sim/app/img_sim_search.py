import os
import base64
from io import BytesIO
from typing import List, Tuple
import logging
import torch
from torchvision import models, transforms
import faiss
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

from config import DATASET_DIR

logger = logging.getLogger(__name__)


class ImageSimSearch:
    def __init__(self):
        """Initialize the image similarity search class."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        # Remove the final classification layer to obtain embeddings.
        self.embedding_model = torch.nn.Sequential(
            *(list(self.model.children())[:-1])
        ).to(self.device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # These will be set during dataset processing.
        self.index: faiss.Index = None
        self.image_files: List[str] = None

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Extract embedding from an image file given by its path."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(image_tensor)
        return embedding.cpu().view(-1).numpy()

    def get_embedding_from_image(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from a PIL image."""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(image_tensor)
        return embedding.cpu().view(-1).numpy()

    def process_image_directory(
        self, image_folder: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Process all image files in a directory and return embeddings plus filenames."""
        image_files = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        embeddings = []
        valid_files = []
        for img_file in tqdm(image_files, desc="Processing Images"):
            try:
                emb = self.get_embedding(img_file)
                embeddings.append(emb)
                valid_files.append(os.path.basename(img_file))
            except Exception as e:
                logger.error(f"Error processing {img_file}: {str(e)}")
        embeddings = np.array(embeddings)
        return embeddings, valid_files

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create and return a FAISS index from the embeddings."""
        # pylint: disable = no-value-for-parameter
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        # pylint: enable = no-value-for-parameter
        return index

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """Search for similar images in the FAISS index."""
        if self.index is None or self.image_files is None:
            raise ValueError(
                "FAISS index and image files must be initialized before searching."
            )
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.image_files):
                continue  # Skip invalid indices
            # pylint: disable = unsubscriptable-object
            results.append(
                {
                    "image_path": os.path.normpath(
                        os.path.join(DATASET_DIR, self.image_files[idx])
                    ),
                    "similarity_score": float(1 / (1 + distances[0][i])),
                }
                # pylint: enable = unsubscriptable-object
            )
        return results

    def process_input_image(self, input_data: str, is_base64: bool) -> Image.Image:
        """Process an input image from a URL or base64 string and return a PIL image."""
        if is_base64:
            # Handle base64 input (strip data URI header if present)
            if "base64," in input_data:
                input_data = input_data.split("base64,")[1]
            img_data = base64.b64decode(input_data)
            return Image.open(BytesIO(img_data)).convert("RGB")
        else:
            response = requests.get(input_data, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")


# Global instance for shared usage.
IMAGE_SIMSEARCH_INSTANCE = None


def get_image_sim_search():
    return IMAGE_SIMSEARCH_INSTANCE


def setup_image_sim_search():
    global IMAGE_SIMSEARCH_INSTANCE
    IMAGE_SIMSEARCH_INSTANCE = ImageSimSearch()
