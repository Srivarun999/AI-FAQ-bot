"""
app.py
------
FAQ Bot — a single-page Streamlit dashboard that answers Computer Science
FAQs intelligently using a Retrieval pipeline:

    Question -> Embedding -> FAISS Search -> Best Match -> Answer

Run with:
    streamlit run app.py
"""

import streamlit as st
from retrieval import build_index, search, load_faq_dataset, tokenize_query

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FAQ Bot — CS Knowledge Base",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------
# Premium custom styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    h1 {
        background: linear-gradient(90deg, #06B6D4, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    h2 {
        color: #06B6D4;
        font-weight: 700;
        border-bottom: 2px solid #1E293B;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #A5B4FC;
    }
    .subtitle {
        color: #A1A1AA;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #06B6D4, #6366F1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(6, 182, 212, 0.3) !important;
    }
    .answer-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 2px solid #06B6D4;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
        line-height: 1.7;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.15);
    }
    .match-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #22303C;
        border-left: 4px solid #6366F1;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .match-card:hover {
        border-left-color: #06B6D4;
        box-shadow: 0 4px 8px rgba(6, 182, 212, 0.2);
    }
    .pipeline-step {
        display: inline-block;
        background: linear-gradient(135deg, #1E293B, #0F172A);
        color: #7DD3FC;
        border: 1px solid #22303C;
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem;
        margin: 0.25rem;
        font-weight: 600;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #1E293B, #0F172A);
        color: #A5B4FC;
        border: 1px solid #3F46E1;
        border-radius: 999px;
        padding: 0.15rem 0.75rem;
        font-size: 0.75rem;
        margin-right: 0.5rem;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #22303C;
        border-radius: 10px;
        padding: 1rem;
    }
    .faq-item {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #22303C;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load / build index (cached across reruns in this session)
@st.cache_resource(show_spinner=False)
def get_index():
    return build_index()

index, faq_list = get_index()

st.title("🤖 FAQ Bot")
st.markdown(
    '<p class="subtitle">Ask any Computer Science question — answered '
    'intelligently from a 151-entry AI/ML FAQ knowledge base using embeddings + '
    'FAISS retrieval.</p>',
    unsafe_allow_html=True,
)

st.divider()

st.subheader("🔍 Ask Your Question")
question = st.text_input(
    "💬 Type a Computer Science question",
    placeholder="e.g. What is the difference between a process and a thread?",
    key="main_search"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    ask = st.button("🔎 Generate Answer", use_container_width=True)
with col2:
    show_kb = st.button("📚 Knowledge Base", use_container_width=True)

st.divider()

if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "show_knowledge_base" not in st.session_state:
    st.session_state.show_knowledge_base = False

if show_kb:
    st.session_state.show_knowledge_base = not st.session_state.show_knowledge_base

if st.session_state.show_knowledge_base:
    st.subheader("📚 Complete FAQ Knowledge Base")
    st.markdown(f"**Total FAQs:** {len(faq_list)}")
    
    for item in faq_list:
        st.markdown(
            f'<div class="faq-item">'
            f'<span class="badge">{item["category"]}</span><br>'
            f'<b>Q{item["id"]}: {item["question"]}</b><br><br>'
            f'<i>A: {item["answer"]}</i>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.divider()

if ask:
    if not question.strip():
        st.error("❌ Please type a question first.")
    else:
        with st.spinner("🔄 Embedding question & searching FAISS index..."):
            st.session_state.search_results = search(question, index, faq_list, top_k=3)

if st.session_state.search_results is not None:
    results = st.session_state.search_results
    
    if not results or results[0]["score"] < 0.35:
        st.warning("🤷 No confident match found. Try rephrasing your question.")
    else:
        st.success("✅ Found matches!")
        
        best = results[0]
        
        # Display answer with cosine similarity aside
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f'<div class="answer-box">{best["answer"]}</div>', unsafe_allow_html=True)
        with col2:
            similarity_pct = int(best["score"] * 100)
            st.metric("Cosine Similarity", f"{similarity_pct}%")
        
        st.divider()
        
        # Show tokenizer info
        tokenizer_info = tokenize_query(question)
        st.subheader("🔤 Tokenization Details")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**Tokenizer:** {tokenizer_info['tokenizer_name']}")
            st.markdown(f"**Model:** {tokenizer_info['model_name']}")
        with col2:
            st.markdown(f"**Input Query:** _{question}_")
        
        st.markdown("**Token IDs:**")
        token_df_data = []
        for token, token_id in zip(tokenizer_info['tokens'], tokenizer_info['token_ids']):
            token_df_data.append({"Token": token, "Token ID": token_id})
        
        st.dataframe(token_df_data, use_container_width=True, hide_index=True)
        
        st.divider()
        
        if len(results) > 1:
            st.subheader(f"🔗 Related Matches ({len(results) - 1})")
            for i, r in enumerate(results[1:], 1):
                match_col1, match_col2 = st.columns([5, 1])
                with match_col1:
                    st.markdown(
                        f'<div class="match-card">'
                        f'<span class="badge">{r["category"]}</span>'
                        f'<span class="badge">Score: {r["score"]:.4f}</span><br>'
                        f'<b>#{i+1} - {r["question"]}</b><br>'
                        f'{r["answer"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with match_col2:
                    sim_pct = int(r["score"] * 100)
                    st.metric(f"Similarity #{i+1}", f"{sim_pct}%")
        
        st.divider()
        with st.expander("📊 About FAISS & Cosine Similarity"):
            st.markdown("""
**FAISS (Facebook AI Similarity Search)** finds the closest FAQ matches using cosine similarity:
- **Cosine Similarity Score**: Ranges from 0 to 1 (higher = better match)
- **Embedding Model**: all-MiniLM-L6-v2 (captures semantic meaning)
- **How it works**: Your question is converted to a vector, then compared against all 120 FAQ embeddings
- **Score > 0.7**: Strong match | **0.5-0.7**: Moderate match | **< 0.5**: Weak match
            """)
