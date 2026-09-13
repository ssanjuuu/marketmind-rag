import json
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Common words to ignore during keyword matching - they're too generic to be useful
STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "of", "to", "in",
    "for", "with", "as", "by", "this", "that", "it", "are", "was", "be", "have",
    "has", "market", "markets", "economy", "economic", "news", "today", "says"
}

def tokenize(text):
    """Split text into words, removing generic stopwords."""
    return [word for word in text.lower().split() if word not in STOPWORDS]

# Load the same embedding model used before
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to your existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="news_articles")

# Load all chunks from ChromaDB to build the BM25 index
print("Building BM25 keyword index...")
all_data = collection.get()
all_documents = all_data["documents"]
all_metadatas = all_data["metadatas"]
all_ids = all_data["ids"]

tokenized_corpus = [tokenize(doc) for doc in all_documents]
bm25 = BM25Okapi(tokenized_corpus)

print(f"Indexed {len(all_documents)} chunks for keyword search\n")


def vector_search(query, n_results=10):
    """Semantic search using embeddings."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    scored = {}
    for doc, meta, dist, doc_id in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]
    ):
        similarity = 1 - dist
        scored[doc_id] = {"doc": doc, "meta": meta, "vector_score": similarity}
    return scored


def keyword_search(query, n_results=10):
    """BM25 keyword search."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Only keep documents with an actual keyword match (score > 0)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    top_indices = [idx for idx in top_indices if scores[idx] > 0]

    scored = {}
    for idx in top_indices:
        doc_id = all_ids[idx]
        scored[doc_id] = {
            "doc": all_documents[idx],
            "meta": all_metadatas[idx],
            "keyword_score": scores[idx],
        }
    return scored


def hybrid_search(query, n_results=5, vector_weight=0.6, keyword_weight=0.4):
    """Combine vector and keyword search with weighted scores."""
    vector_results = vector_search(query, n_results=15)
    keyword_results = keyword_search(query, n_results=15)

    max_keyword_score = max([r["keyword_score"] for r in keyword_results.values()], default=1)
    if max_keyword_score == 0:
        max_keyword_score = 1

    combined = {}
    all_ids_seen = set(vector_results.keys()) | set(keyword_results.keys())

    for doc_id in all_ids_seen:
        v_score = vector_results.get(doc_id, {}).get("vector_score", 0)
        k_score_raw = keyword_results.get(doc_id, {}).get("keyword_score", 0)
        k_score = k_score_raw / max_keyword_score

        final_score = (vector_weight * v_score) + (keyword_weight * k_score)

        meta = vector_results.get(doc_id, keyword_results.get(doc_id))["meta"]
        doc = vector_results.get(doc_id, keyword_results.get(doc_id))["doc"]

        combined[doc_id] = {
            "doc": doc,
            "meta": meta,
            "final_score": final_score,
            "vector_score": v_score,
            "keyword_score": k_score,
        }

    ranked = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)
    return ranked[:n_results]


if __name__ == "__main__":
    print("MarketMind Hybrid Search - type a question, or 'quit' to exit\n")

    while True:
        query = input("Ask something: ")
        if query.lower() in ["quit", "exit"]:
            break

        results = hybrid_search(query)

        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} (combined score: {r['final_score']:.3f}, "
                  f"vector: {r['vector_score']:.3f}, keyword: {r['keyword_score']:.3f}) ---")
            print(f"Source: {r['meta']['source']}")
            print(f"Title: {r['meta']['title']}")
            print(f"Link: {r['meta']['link']}")

        print("\n")