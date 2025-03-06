import os

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the cache directory
CACHE_DIR = os.path.join(BASE_DIR, "..", "..", "_CACHE")
os.makedirs(CACHE_DIR, exist_ok=True)

# Define the dataset directory
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "_DS", "Txt_DS")

# Define the embeddings file path
DATASET_NAME = os.path.basename(DATASET_DIR)  # e.g., "Audio_DS"
EMBEDDINGS_FILE = os.path.join(DATASET_DIR, f"{DATASET_NAME}_embeddings.index")