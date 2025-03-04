import os
import logging
import torch
import faiss
from txt_sim_search import (
    load_data,
    build_embeddings,
    build_index,
    search_similar,
    combine_all_columns,
)

log = logging.getLogger("app")


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct CSV path
    # csv_path = os.path.join(BASE_DIR, "..", "..", "_DS", "Text_Similarity_Dataset.csv")
    csv_path = os.path.join(BASE_DIR, "..", "..", "_DS", "war-news.csv")
    log.info("CSV file path:" + os.path.abspath(csv_path))

    # Derive index filename from CSV filename
    csv_filename = os.path.basename(csv_path)
    index_filename = os.path.splitext(csv_filename)[0] + ".index"
    index_file_path = os.path.join(BASE_DIR, "..", "..", index_filename)

    df = load_data(csv_path)
    df["combined_text"] = df.apply(combine_all_columns, axis=1)
    texts = df["combined_text"].tolist()

    # Load or build index
    if os.path.exists(index_file_path):
        log.info("Loading existing FAISS index from disk...")
        index = faiss.read_index(index_file_path)
        if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
            log.info("Transferring loaded CPU index to GPU for faster search...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            log.info("Using CPU for FAISS index.")
    else:
        log.info("Building embeddings and FAISS index from dataset...")
        embeddings = build_embeddings(texts)
        index = build_index(embeddings)
        log.info("Saving FAISS index to disk...")
        # Convert GPU index to CPU index before saving if necessary.
        if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
            log.info("Converting GPU index to CPU index for serialization...")
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, index_file_path)


    log.info("\nWelcome to Text Similarity Search!")
    log.info("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your text query: ")
        if query.lower() == "exit":
            log.info("Exiting the application.")
            break

        results = search_similar(index, query, texts=texts, k=5)
        log.info("\nTop 5 Similar Results:")
        for i, (text_match, distance) in enumerate(results):
            log.info(f"{i+1}. {text_match} (Distance: {distance:.4f})\n\n")
        log.info("")


if __name__ == "__main__":
    main()
