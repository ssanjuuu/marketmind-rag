import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="MarketMind", page_icon="📈", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="news_articles")

model = load_model()
collection = load_db()

st.title("📈 MarketMind")
st.caption("Semantic search over live financial and crypto news")

query = st.text_input("Ask a question about markets, crypto, or finance:", placeholder="e.g. What's happening with Bitcoin?")

if query:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    st.subheader(f"Top {len(documents)} results")

    for doc, meta, dist in zip(documents, metadatas, distances):
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{meta['title']}**")
                st.caption(f"Source: {meta['source']} | Published: {meta['published']}")
                st.write(doc[:250] + "...")
                st.markdown(f"[Read full article]({meta['link']})")
            with col2:
                st.metric("Relevance", f"{1 - dist:.2f}")