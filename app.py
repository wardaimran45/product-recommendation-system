"""
Product Recommendation System — Streamlit + free local embeddings (Hugging Face)

No API key required. Embeddings are computed locally using a small
open-source sentence-transformer model.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from products import PRODUCTS

st.set_page_config(page_title="Product Recommender", page_icon="🛍️", layout="centered")


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_model():
    # Small, fast, fully local model — no API key, no cost.
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Embedding product catalog...")
def embed_products(_model, products):
    texts = [f"{p['title']}. {p['description']}" for p in products]
    embeddings = _model.encode(texts)
    return embeddings


def get_recommendations(selected_index, embeddings, products, top_n=3):
    target_embedding = embeddings[selected_index].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, embeddings)[0]

    # Rank by similarity, excluding the selected product itself
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_indices = [i for i in ranked_indices if i != selected_index]

    top_indices = ranked_indices[:top_n]
    return [(products[i], similarities[i]) for i in top_indices]


def main():
    st.title("🛍️ Product Recommendation System")
    st.caption(
        "Pick a product a user just viewed — this recommends similar items "
        "using semantic embeddings, no purchase data required."
    )

    model = load_model()
    embeddings = embed_products(model, PRODUCTS)

    titles = [p["title"] for p in PRODUCTS]
    selected_title = st.selectbox("Last product viewed by the user:", titles)
    selected_index = titles.index(selected_title)
    selected_product = PRODUCTS[selected_index]

    st.divider()
    st.subheader("Viewed product")
    render_product_card(selected_product)

    st.divider()
    st.subheader("Recommended for this user")

    recommendations = get_recommendations(selected_index, embeddings, PRODUCTS, top_n=3)
    cols = st.columns(3)
    for col, (product, score) in zip(cols, recommendations):
        with col:
            render_product_card(product, score=score)

    with st.expander("How this works"):
        st.markdown(
            """
            1. Each product's title + description is turned into a numeric
               **embedding** using a free, local Hugging Face model
               (`all-MiniLM-L6-v2` — no API key needed).
            2. When a user views a product, we compare its embedding to every
               other product's embedding using **cosine similarity**.
            3. The three most similar products are shown as recommendations —
               the same retrieval pattern used in production recommendation
               and RAG systems, just without an API cost.
            """
        )


def render_product_card(product, score=None):
    st.markdown(f"**{product['title']}**")
    st.caption(f"{product['category']} · €{product['price']:.2f}")
    st.write(product["description"])
    if score is not None:
        st.progress(min(max(float(score), 0.0), 1.0), text=f"Similarity: {score:.2f}")


if __name__ == "__main__":
    main()
