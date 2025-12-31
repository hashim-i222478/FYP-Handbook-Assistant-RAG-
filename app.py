import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import math
import google.generativeai as genai
import os

########################################
# CONFIG
########################################
INDEX_DIR = "handbook_index"
FAISS_INDEX_PATH = f"{INDEX_DIR}\\faiss.index"
CHUNKS_JSONL_PATH = f"{INDEX_DIR}\\chunks.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"

TOP_K = 7          # retrieve 7 candidates
FINAL_K = 5        # keep top 5 after MMR
SIM_THRESHOLD = 0.25  # assignment requirement

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key="AIzaSyA8cN50FzhTVbIzzoB-ASF-QekR_V_gl9A")

########################################
# LOAD INDEX, MODEL & CHUNKS
########################################
@st.cache_resource(show_spinner=True)
def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=True)
def load_chunks():
    chunks = []
    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

index = load_faiss_index()
model = load_model()
chunks_db = load_chunks()


########################################
# NORMALIZED QUERY EMBEDDING
########################################
def embed_query(text: str):
    emb = model.encode([text], convert_to_numpy=True)[0].astype("float32")
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb


########################################
# MMR RERANKING
########################################
def mmr(query_vec, doc_vecs, doc_ids, top_n=5, lambda_param=0.5):
    """
    Maximal Marginal Relevance:
    - query_vec: (dim,)
    - doc_vecs: (k, dim)
    - doc_ids:  indices in FAISS
    """
    selected = []
    remaining = list(range(len(doc_ids)))

    query_vec = query_vec.reshape(1, -1)
    doc_vecs_norm = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    for _ in range(top_n):
        if not remaining:
            break

        # Compute relevance scores
        sims_to_query = np.dot(doc_vecs_norm, query_vec.T).reshape(-1)

        if not selected:
            # pick the highest relevance first
            idx = remaining[np.argmax(sims_to_query[remaining])]
            selected.append(idx)
            remaining.remove(idx)
        else:
            # For each remaining doc, compute max similarity with selected docs
            selected_vecs = doc_vecs_norm[selected]
            sims_to_selected = np.dot(doc_vecs_norm, selected_vecs.T)  # shape (k, len(selected))
            max_sims = sims_to_selected.max(axis=1)

            # MMR score
            mmr_scores = lambda_param * sims_to_query - (1 - lambda_param) * max_sims

            idx = remaining[np.argmax(mmr_scores[remaining])]
            selected.append(idx)
            remaining.remove(idx)

    # Return document IDs corresponding to selected indices
    return [doc_ids[i] for i in selected]


########################################
# RETRIEVAL FUNCTION
########################################
def retrieve_chunks(query: str):
    q_emb = embed_query(query)

    # FAISS search
    scores, indices = index.search(q_emb.reshape(1, -1), TOP_K)
    scores = scores[0]
    indices = indices[0]

    # If all < threshold → reject
    if np.all(scores < SIM_THRESHOLD):
        return None, None, None

    # Filter invalid FAISS results (index returns -1 if fewer items)
    valid = [(i, s) for i, s in zip(indices, scores) if i != -1]
    if not valid:
        return None, None, None

    idxs = [v[0] for v in valid]
    vecs = []
    for i in idxs:
        # re-embed chunk for MMR or load embeddings.npy if you saved it
        vec = model.encode([chunks_db[i]["text"]], convert_to_numpy=True)[0]
        faiss.normalize_L2(vec.reshape(1, -1))
        vecs.append(vec)
    doc_vecs = np.vstack(vecs)

    # MMR reranking
    final_ids = mmr(q_emb, doc_vecs, idxs, top_n=FINAL_K)

    # Return final results
    selected_chunks = [chunks_db[i] for i in final_ids]
    selected_scores = [float(s) for (i, s) in valid if i in final_ids]

    return selected_chunks, final_ids, selected_scores


########################################
# NOW BUILD STREAMLIT UI
########################################
st.title("FYP Handbook Assistant (RAG)")
st.markdown("Ask questions strictly related to the **BS FYP Handbook 2023**.")

query = st.text_input("Ask a question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Retrieving from handbook..."):
        selected_chunks, final_ids, scores = retrieve_chunks(query)

    if selected_chunks is None:
        st.error("I don’t have that in the handbook.")
        st.stop()

    ########################################
    # Build Answer Using Prompt Template
    ########################################
    answer_context = "\n\n".join(
        f"[Chunk from p.{c['page']}]\n{c['text']}"
        for c in selected_chunks
    )

    prompt = f"""
You are a handbook assistant. Answer ONLY from the context.
Cite page numbers like "(p. X)". If unsure, say you don't know.

Question: {query}

Context:
{answer_context}
    """

    ########################################
    # Generate final answer using Gemini API
    ########################################
    try:
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = gemini_model.generate_content(prompt)
        extracted_answer = response.text
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        extracted_answer = "Unable to generate answer. Please check your API key and internet connection."

    st.subheader("Answer")
    st.write(extracted_answer)

    ########################################
    # Show Sources
    ########################################
    with st.expander("🔎 Sources (page references)"):
        for c in selected_chunks:
            st.markdown(f"**Chunk from page {c['page']}**")
            st.text(c['text'])
