"""
build_index.py
---------------
Optional standalone script to pre-build the FAISS index and embedding cache
before launching the Streamlit app, so the first app load is instant.

Usage:
    python build_index.py
"""

from retrieval import build_index

if __name__ == "__main__":
    print("Building FAISS index over the CS FAQ knowledge base...")
    index, faq_list = build_index(force_rebuild=True)
    print(f"Done. Indexed {len(faq_list)} FAQ entries.")
    print("Cached index: data/faq_index.faiss")
    print("Cached metadata: data/faq_meta.json")
