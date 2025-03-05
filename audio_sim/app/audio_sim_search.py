import os
import requests
import tempfile
import numpy as np
import librosa
import torch
import faiss
from transformers import ASTFeatureExtractor, ASTModel
from typing import List

# Load AST model and feature extractor
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
model = ASTModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def download_audio_from_url(url: str) -> str:
    """Download an audio file from a URL and return the local file path."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    except Exception as e:
        print(f"Error downloading audio from URL: {str(e)}")
        return None


def get_audio_paths(audio_folder: str) -> List[str]:
    """Get all WAV files from the audio directory"""
    audio_files = []
    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files


def wav_to_embedding(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Convert WAV file (local path or URL) to AST embedding."""
    try:
        # If the input is a URL, download the file first
        if wav_path.startswith(("http://", "https://")):
            local_path = download_audio_from_url(wav_path)
            if local_path is None:
                return None
            wav_path = local_path  # Use the downloaded file
            is_temp_file = True
        else:
            is_temp_file = False

        # Load and process the audio file
        waveform, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        inputs = feature_extractor(
            waveform,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

        # Clean up temporary file if it was created
        if is_temp_file:
            os.remove(wav_path)

        return embedding
    except Exception as e:
        print(f"Error processing {wav_path}: {str(e)}")
        return None


def process_audio_directory(audio_folder: str) -> tuple:
    """Process all audio files and return embeddings + filenames"""
    audio_files = get_audio_paths(audio_folder)
    embeddings = []
    valid_files = []

    for audio_file in audio_files:
        emb = wav_to_embedding(audio_file)
        if emb is not None:
            embeddings.append(emb)
            valid_files.append(os.path.basename(audio_file))

    return np.array(embeddings), valid_files


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create and return FAISS index"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # For Euclidean distance
    index.add(embeddings.astype(np.float32))
    return index


def search_similar(
    index: faiss.Index,
    filenames: List[str],
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> List[dict]:
    """Search for similar items in the FAISS index"""
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(filenames):
            continue  # Skip invalid indices
        results.append(
            {
                "audio_path": filenames[idx],
                "similarity_score": float(
                    1 / (1 + distances[0][i])
                ),  # Convert distance to similarity
            }
        )
    return results
