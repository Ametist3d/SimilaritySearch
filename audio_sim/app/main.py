import os
import numpy as np
from audio_sim_search import process_audio_directory, create_faiss_index, search_similar, wav_to_embedding
from transformers import ASTFeatureExtractor, ASTModel

def main():
    # Configure paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(BASE_DIR, "..", "..", "_DS", "Audio_DS")
    audio_folder = os.path.abspath(audio_folder)
    dataset_file = os.path.join(BASE_DIR, "..", "..", "audio_emb_dataset.npz")

    # Load/create embeddings
    if not os.path.exists(dataset_file):
        print("Creating new embeddings dataset...")
        embeddings, filenames = process_audio_directory(audio_folder)
        np.savez(dataset_file, embeddings=embeddings, filenames=filenames)
    else:
        print("Loading existing embeddings dataset...")
        data = np.load(dataset_file, allow_pickle=True)
        embeddings = data['embeddings']
        filenames = data['filenames']

    # Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Load model once for reuse
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # Interactive query loop
    while True:
        user_input = input("\nEnter query audio sample absolute path (or 'exit' to quit): ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
            
        if not os.path.isfile(user_input):
            print(f"Error: File '{user_input}' not found.")
            continue
            
        if not user_input.lower().endswith('.wav'):
            print("Error: Only .wav files are supported.")
            continue
            
        # Process query
        query_emb = wav_to_embedding(user_input, feature_extractor, model)
        if query_emb is not None:
            results = search_similar(index, filenames, np.array([query_emb]), 3)
            print("\nTop 3 similar results:")
            for idx, (filename, distance) in enumerate(results, 1):
                print(f"{idx}. {filename} (distance: {distance:.4f})")

if __name__ == "__main__":
    main()