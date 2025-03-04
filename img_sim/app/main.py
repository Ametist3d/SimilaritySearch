import os
import logging
import numpy as np
import faiss
from img_sim_search import get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the images folder
    images_folder = os.path.join(BASE_DIR, "..", "..", "_DS", "dataset")
    images_folder = os.path.abspath(images_folder)
    log.info("Image dataset path: " + images_folder)

    # List all image files in the folder (filter by common image extensions)
    image_files = [
        os.path.join(images_folder, file)
        for file in os.listdir(images_folder)
        if os.path.isfile(os.path.join(images_folder, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    log.info("Found {} images.".format(len(image_files)))

    # Define path to the embeddings file
    dataset_file = os.path.join(BASE_DIR, "..", "..", "embeddings_dataset.npz")

    # Check if embeddings file exists; if so, load from it
    if os.path.exists(dataset_file):
        log.info("Found existing embeddings dataset file: " + dataset_file)
        data = np.load(dataset_file, allow_pickle=True)
        embeddings = data["embeddings"]
        image_files = data["image_files"]
    else:
        # Process dataset images and extract embeddings
        embeddings = [get_embedding(path) for path in image_files]
        embeddings = np.stack(embeddings)
        # Save embeddings and corresponding image file paths to a file
        np.savez(dataset_file, embeddings=embeddings, image_files=image_files)
        log.info("Saved embeddings to file: " + dataset_file)

    # Build a FAISS index for efficient similarity search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance
    #pylint: disable-next=no-value-for-parameter
    index.add(embeddings)  # Add dataset embeddings to the index

    # Querying: Let the user input an absolute path for the query image
    user_input = input("Enter query image absolute path: ").strip()
    log.info("Query image path: " + user_input)
    
    # Compute the query image's embedding
    query_embedding = get_embedding(user_input)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Retrieve the k nearest neighbors
    k = 3  # Number of similar images to retrieve
    #pylint: disable-next=no-value-for-parameter
    distances, indices = index.search(query_embedding, k)
    print("Indices of similar images:", indices)

    # Display the file paths of the similar images
    print("Similar images:")
    for idx in indices[0]:
        print(image_files[idx])

if __name__ == "__main__":
    main()
