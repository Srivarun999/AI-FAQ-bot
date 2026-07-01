"""
retrieval.py
------------
Implements the full retrieval pipeline taught in this project:

    Question -> Embedding -> FAISS Search -> Best Match -> Answer

- Embeddings are generated with a local Sentence-Transformers model
  (all-MiniLM-L6-v2), so no API key is required for retrieval itself.
- FAISS (Facebook AI Similarity Search) is used for fast vector search
  over the FAQ knowledge base.
- The index is built once and cached to disk (data/faq_index.faiss +
  data/faq_meta.json) so subsequent app launches are instant.
"""

import json
import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "faq_dataset.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "faq_index.faiss")
META_PATH = os.path.join(os.path.dirname(__file__), "data", "faq_meta.json")

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_tokenizer = None


def get_model():
    """Lazily load the sentence-transformer embedding model (cached)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_tokenizer():
    """Get the tokenizer from the sentence-transformers model."""
    global _tokenizer
    if _tokenizer is None:
        model = get_model()
        _tokenizer = model.tokenizer
    return _tokenizer


def tokenize_query(query: str):
    """
    Tokenize a query and return token information.
    Returns: {tokenizer_name, tokens, token_ids}
    """
    tokenizer = get_tokenizer()
    encoded = tokenizer.encode(query, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    
    return {
        "tokenizer_name": tokenizer.__class__.__name__,
        "model_name": MODEL_NAME,
        "tokens": tokens,
        "token_ids": encoded,
    }


def load_faq_dataset():
    """Load the raw FAQ knowledge base from disk."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts):
    """Step 1-2: Turn a list of strings into embedding vectors."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def build_index(force_rebuild: bool = False):
    """
    Build (or load a cached) FAISS index over all FAQ questions.
    Returns (faiss_index, faq_list).
    """
    import faiss

    faq_list = load_faq_dataset()

    if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            faq_list = json.load(f)
        return index, faq_list

    questions = [item["question"] for item in faq_list]
    embeddings = embed_texts(questions)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalized inner product
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(faq_list, f, indent=2)

    return index, faq_list


def search(query: str, index, faq_list, top_k: int = 3):
    """
    Step 3-4: Embed the user's query and search the FAISS index for the
    closest matching FAQ entries.

    Returns a list of dicts: {question, answer, category, score}
    """
    query_vec = embed_texts([query])
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = faq_list[idx]
        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "category": item["category"],
            "score": float(score),  # cosine similarity, higher = better match (max 1.0)
        })
    return results
