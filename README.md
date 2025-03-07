# SimilaritySearch

SimilaritySearch is a multi-modal similarity search project that enables efficient retrieval of similar items from various data types. The currently supported modalities include:

- **Text** – Search and compare textual data using SentenceTransformers.
- **Audio** – Compare audio files using an Audio Spectrogram Transformer (AST) model.
- **Image** – Find similar images using ResNet50 for embedding extraction.
- **GIF** – Search for similar GIFs using the CLIP model.

All services are built with FastAPI and utilize FAISS for fast similarity search.

## Features

- **Text Similarity Search:** Leverage state-of-the-art sentence embeddings to search and compare texts.
- **Audio Similarity Search:** Process and compare WAV files using AST-based embeddings.
- **Image Similarity Search:** Extract image embeddings using a pre-trained ResNet50 network.
- **GIF Similarity Search:** Compute GIF embeddings by extracting and aggregating frame features using CLIP.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/SimilaritySearch.git
   cd SimilaritySearch
   ```
2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   The project includes multiple requirements.txt files (each tailored for different modalities). You can install a consolidated set of dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```
   Make sure your Python version and environment support libraries such as PyTorch, FAISS, and FastAPI.

## Configuration

Each modality uses its own config.py file to set up key parameters and paths. The common configuration parameters include:

 - BASE_DIR: The root directory for the module.
 - CACHE_DIR: Directory where model weights and caches are stored.
 - DATASET_DIR: Directory containing the dataset. Organize your datasets as follows:
    - Text: Place CSV files under _DS/Txt_DS
    - Audio: Place WAV files under _DS/Audio_DS
    - Image: Place image files (PNG, JPG, JPEG) under _DS/Img_DS
    - GIF: Place GIF files under _DS/Gif_DS
- EMBEDDINGS_FILE: The path where computed embeddings or FAISS indexes will be saved.

**Example datasets**
 - **Text:**   - https://www.kaggle.com/code/ayushithakre12/starter-task-finding-semantic-textual-57d22b08-2
 - **Images:** - https://academictorrents.com/details/df0aad374e63b3214ef9e92e178580ce27570e59
 - **Audio:**  - https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
 - **GIF:**    - https://www.kaggle.com/datasets/electron0zero/giphy-dataset
 
Before running any service, verify that the paths in the corresponding config.py file match your local directory structure and dataset locations.

## Running the Services

Each modality has its own FastAPI application implemented in a main.py file. To run a service (for example, the GIF similarity search), execute:
```bash
   python main.py
```
By default, the service runs on http://0.0.0.0:5000. To run multiple services simultaneously, adjust the port number in each main.py file.

## Sending Requests
All services expose a /search endpoint that accepts POST requests with JSON payloads. Below are example payloads for each modality:

1. *Text Similarity Search*

Endpoint: /search
Payload:
```json
   {
   "query": "your search text",
   "k": 3
   }
```
2. *Audio Similarity Search*

Endpoint: /search
Payload:
```json
   {
   "audio": "URL or base64 encoded audio",
   "is_base64": false,
   "k": 3
   }
```
3. *Image Similarity Search*

Endpoint: /search
Payload:
```json
   {
  "image": "URL or base64 encoded image",
  "is_base64": false,
  "k": 3
   }
```
4. *GIF Similarity Search*

Endpoint: /search
Payload:
```json
   {
  "gif": "URL or base64 encoded GIF",
  "is_base64": false,
  "k": 3
   }
```
You can test these endpoints using tools like curl or Postman.

## Project Structure

 - README.md: Project overview and detailed instructions.
 - config.py: Contains configuration settings (dataset paths, cache directories, embedding file locations) for each modality.
 - main.py: FastAPI application entry point for each service.
 - request.py: Defines request data schemas using Pydantic.
 - *_sim_search.py: Core modules for computing embeddings and performing similarity searches (for text, audio, image, and GIF).
 - requirements.txt: Lists required dependencies for the project.