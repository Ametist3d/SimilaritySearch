from txt_sim_search import load_data, build_embeddings, build_index, search_similar
import os

def main():
    # CSV path relative to main.py (located in txt_sim/app)
    # Get the absolute path of the current file (main.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the CSV file relative to the project root
    csv_path = os.path.join(BASE_DIR, "..", "..", "_DS", "Text_Similarity_Dataset.csv")

    print("CSV file path:", os.path.abspath(csv_path))
        
    # Load data from CSV
    df = load_data(csv_path)
    # Adjust the column name if needed (assuming the text column is named 'text')
    texts = df['text1'].tolist()

    print("Building embeddings...")
    embeddings = build_embeddings(texts)
    print("Building FAISS index...")
    index = build_index(embeddings)

    print("\nWelcome to Text Similarity Search!")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Enter your text query: ")
        if query.lower() == "exit":
            print("Exiting the application.")
            break
        
        results = search_similar(index, query, texts=texts, k=5)
        print("\nTop 5 Similar Results:")
        for i, (text_match, distance) in enumerate(results):
            print(f"{i+1}. {text_match} (Distance: {distance:.4f})")
        print("")

if __name__ == "__main__":
    main()
