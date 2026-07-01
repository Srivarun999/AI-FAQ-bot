# 🤖 FAQ Bot — Computer Science Knowledge Base

**PROJECT-3** — A single-page Streamlit app that answers predefined FAQs
intelligently using a full retrieval pipeline built on embeddings + FAISS.

## 🎓 Concepts covered
- **Knowledge Base**: a curated FAQ dataset (`data/faq_dataset.json`)
- **Retrieval Pipeline**:
  ```
  Question → Embedding → FAISS Search → Best Match → Answer
  ```
- **Retrieval-Augmented answering**: optionally rephrasing the retrieved
  answer with an LLM, using it strictly as grounding context

## 🗂️ Project structure
```
faq-bot/
├── app.py                  # Main Streamlit dashboard (single page UI)
├── retrieval.py             # Embedding + FAISS pipeline logic
├── build_index.py           # Optional CLI script to pre-build the index
├── data/
│   └── faq_dataset.json     # 120-entry Computer Science FAQ knowledge base
├── requirements.txt
├── .streamlit/
│   └── config.toml          # Premium dark theme
├── .gitignore
└── README.md
```

## 📚 About the knowledge base

`data/faq_dataset.json` contains **120 hand-written FAQ entries** covering
13 core Computer Science areas:

| Category | # Questions |
|---|---|
| Programming Fundamentals | 10 |
| Object-Oriented Programming | 10 |
| Data Structures | 10 |
| Algorithms | 10 |
| Database Management Systems | 10 |
| Operating Systems | 10 |
| Computer Networks | 10 |
| Software Engineering | 10 |
| Web Development | 10 |
| Artificial Intelligence & Machine Learning | 10 |
| Cybersecurity | 10 |
| Version Control & Git | 5 |
| Cloud Computing | 5 |

Each entry has the shape:
```json
{
  "id": 53,
  "category": "Operating Systems",
  "question": "What is the difference between a process and a thread?",
  "answer": "A process is an independent execution unit with its own memory space..."
}
```

Feel free to add more entries — the app will automatically re-embed and
re-index them the next time the cache is rebuilt (delete
`data/faq_index.faiss` and `data/faq_meta.json`, or run `build_index.py`).

## 🚀 Setup (VS Code / local)

1. Open this folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Pre-build the FAISS index so the first app launch is instant:
   ```bash
   python build_index.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. The app opens at `http://localhost:8501`.

> ℹ️ The first time you run the app (or `build_index.py`), the
> `sentence-transformers` library will download the small
> `all-MiniLM-L6-v2` embedding model (~80MB) — this only happens once.

## 🔑 API Key (optional)

Retrieval itself works **fully offline with no API key** — it uses a local
embedding model + FAISS. An API key is only needed if you enable the
optional **"Rephrase answer with an LLM"** toggle in the sidebar, which
demonstrates Retrieval-Augmented Generation (RAG): the retrieved FAQ answer
is passed to an LLM as grounding context, and the LLM rephrases it more
conversationally without inventing new facts.

- Get an Anthropic API key at https://console.anthropic.com/

## ☁️ Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io/ and create a new app pointing to
   `app.py`.
3. Deploy — the theme in `.streamlit/config.toml` applies automatically.
4. The FAISS index will build automatically on first load (cached via
   `st.cache_resource` for the app's lifetime); users optionally paste
   their own API key in the sidebar for the LLM rephrasing feature.

## 🖱️ How to use the app

1. Type any Computer Science question in the main input box.
2. Click **Get Answer**.
3. The app shows:
   - The **best matching FAQ** with its category and similarity score
   - The **retrieved answer** (or an LLM-rephrased version if enabled)
   - Other related matches in a collapsible section
4. Expand **"How the retrieval pipeline works"** to see the 5-step
   pipeline explained.
5. Expand **"Browse the full CS FAQ knowledge base"** to explore all 120
   entries, filterable by category.

## 🧩 How the pipeline works (see `retrieval.py`)

1. **Question** — user's raw text query.
2. **Embedding** — `all-MiniLM-L6-v2` (Sentence-Transformers) converts the
   question into a 384-dimensional vector.
3. **FAISS Search** — a `faiss.IndexFlatIP` index (inner product on
   normalized vectors = cosine similarity) compares the query vector
   against all 120 pre-computed FAQ embeddings.
4. **Best Match** — the FAQ entries with the highest similarity scores are
   returned (Top-K configurable in the sidebar).
5. **Answer** — the top match's answer is displayed, optionally passed
   through an LLM for a more natural, conversational rephrasing
   (Retrieval-Augmented Generation).

Try asking the same question with different phrasing to see how the
embedding-based search still finds the right FAQ, unlike simple keyword
matching!
