import os
import numpy as np
import librosa
import torch
import faiss
from transformers import ASTFeatureExtractor, ASTModel

def get_audio_paths(audio_folder):
    """Get all WAV files from the audio directory"""
    audio_files = []
    for root, _, files in os.walk(audio_folder):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def wav_to_embedding(wav_path, feature_extractor, model, target_sr=16000):
    """Convert WAV file to AST embedding"""
    try:
        waveform, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        inputs = feature_extractor(
            waveform,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
    except Exception as e:
        print(f"Error processing {wav_path}: {str(e)}")
        return None

def process_audio_directory(audio_folder):
    """Process all audio files and return embeddings + filenames"""
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    model = ASTModel.from_pretrained(model_name)
    
    audio_files = get_audio_paths(audio_folder)
    embeddings = []
    valid_files = []
    
    for audio_file in audio_files:
        emb = wav_to_embedding(audio_file, feature_extractor, model)
        if emb is not None:
            embeddings.append(emb)
            valid_files.append(os.path.basename(audio_file))
    
    return np.array(embeddings), valid_files

def create_faiss_index(embeddings):
    """Create and return FAISS index"""
    # Normalize embeddings for cosine similarity
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # For Euclidean distance
    # index = faiss.IndexFlatIP(dim)  # For cosine similarity (use with normalized embeddings)
    #pylint: disable-next=no-value-for-parameter
    index.add(embeddings)
    return index

def search_similar(index, filenames, query_embedding, top_k=5):
    """Search for similar items in the FAISS index"""
    distances, indices = index.search(query_embedding, top_k)
    return [(filenames[i], distances[0][idx]) for idx, i in enumerate(indices[0])]