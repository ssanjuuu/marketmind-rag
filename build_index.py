import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load the articles we fetched earlier
with open("articles.json", "r") as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")

# Load a free, local embedding model (downloads once, then runs offline)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up a local, persistent vector database (saves to disk in a "chroma_db" folder)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="news_articles")

def chunk_text(text, chunk_size=300):
    """Split text into rough word-based chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Build chunks from all articles
all_chunks = []
all_metadata = []
all_ids = []

for idx, article in enumerate(articles):
    # Combine title and summary as the text to chunk
    full_text = f"{article['title']}. {article['summary']}"
    chunks = chunk_text(full_text)

    for chunk_idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({
            "source": article["source"],
            "title": article["title"],
            "link": article["link"],
            "published": article["published"],
        })
        all_ids.append(f"article_{idx}_chunk_{chunk_idx}")

print(f"Created {len(all_chunks)} chunks from {len(articles)} articles")

# Generate embeddings for all chunks
print("Generating embeddings (this may take a minute)...")
embeddings = model.encode(all_chunks, show_progress_bar=True)

# Store everything in ChromaDB
print("Storing in ChromaDB...")
collection.upsert(
    ids=all_ids,
    embeddings=embeddings.tolist(),
    documents=all_chunks,
    metadatas=all_metadata,
)

print(f"Done! Indexed {len(all_chunks)} chunks into ChromaDB.")
print("You can now search these articles semantically.")