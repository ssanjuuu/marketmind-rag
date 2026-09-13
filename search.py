import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model used to build the index
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to your existing local ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="news_articles")

def search(query, n_results=5):
    # Convert the question into an embedding
    query_embedding = model.encode([query]).tolist()

    # Find the most similar chunks in the database
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    return results

def print_results(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"\n--- Result {i+1} (similarity score: {1 - dist:.3f}) ---")
        print(f"Source: {meta['source']}")
        print(f"Title: {meta['title']}")
        print(f"Link: {meta['link']}")
        print(f"Excerpt: {doc[:200]}...")

if __name__ == "__main__":
    print("\nMarketMind Search - type a question, or 'quit' to exit\n")

    while True:
        query = input("Ask something: ")
        if query.lower() in ["quit", "exit"]:
            break

        results = search(query)
        print_results(results)
        print("\n")