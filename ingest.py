#!/usr/bin/env python3
"""
ingest.py

Ingest pipeline for the FAST-NUCES FYP Handbook (or any single PDF):
- Extracts text page-by-page (preserves page numbers)
- Cleans recurring headers (e.g., "FAST-NUCES XX")
- Detects section headings
- Chunks text ~250-400 words with ~30% overlap using sentence boundaries
- Creates embeddings with all-MiniLM-L6-v2
- Builds FAISS IndexFlatIP (cosine via normalized vectors)
- Persists index and chunk metadata.

Default PDF path: 3. FYP-Handbook-2023.pdf
"""

import os
import re
import json
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pdfplumber
from tqdm import tqdm

# embeddings
from sentence_transformers import SentenceTransformer
import faiss

# sentence tokenization
import nltk

# ensure punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# -----------------------
# dataclasses
# -----------------------
@dataclass
class Chunk:
    chunk_id: str
    page: int
    section_hint: str
    text: str
    word_start: int
    word_end: int
    # optionally include embedding when building
    # embedding: np.ndarray = None

# -----------------------
# Helpers: cleaning & heading detection
# -----------------------
HEADER_RE = re.compile(r'FAST-NUCES\s*\d{0,2}', flags=re.IGNORECASE)
# Patterns that indicate a section heading:
HEADING_PATTERNS = [
    re.compile(r'^[A-Z][A-Z\s\-\:]{3,}$'),        # ALL CAPS-like headings (>=4 chars)
    re.compile(r'^\d+(\.\d+)*\s+'),               # 1. 1.1 2.3 ...
    re.compile(r'^(Chapter|CHAPTER)\b', flags=re.IGNORECASE),
    re.compile(r'^(Executive Summary|Abstract|References|Appendix|Bibliography)\b', flags=re.IGNORECASE),
]


def clean_page_text(text: str) -> str:
    """
    Clean page text:
    - Remove repetitive headers/footers like FAST-NUCES XX
    - Normalize line breaks
    - Strip excessive whitespace
    """
    # remove header tokens
    text = HEADER_RE.sub('', text)

    # remove repeated page headers like "BS Final Year Project Handbook 2023"
    text = re.sub(r'BS\s*FINAL\s*YEAR\s*PROJECT\s*HANDBOOK\s*2023', '', text, flags=re.IGNORECASE)

    # normalize whitespace
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n\n', text)  # limit multiple newlines
    text = text.strip()
    return text


def is_heading(line: str) -> bool:
    if not line or len(line.strip()) < 3:
        return False
    l = line.strip()
    # check explicit patterns
    for pat in HEADING_PATTERNS:
        if pat.match(l):
            return True
    # also treat short uppercase headings (2-4 words) as heading if many uppercase chars
    if len(l.split()) <= 6 and sum(1 for ch in l if ch.isupper()) / max(1, len(l)) > 0.5:
        return True
    return False

# -----------------------
# Chunking logic
# -----------------------
def sentence_level_chunks_for_section(section_text: str,
                                      min_words: int = 250,
                                      max_words: int = 400,
                                      overlap_pct: float = 0.30) -> List[Tuple[str, int, int]]:
    """
    Create chunks from section_text based on sentence boundaries.
    Returns list of (chunk_text, word_start_idx, word_end_idx).
    """
    sentences = sent_tokenize(section_text)
    # convert sentences to lists of words to measure length
    sent_words = [s.split() for s in sentences]
    sent_lens = [len(sw) for sw in sent_words]

    chunks = []
    current = []
    current_len = 0
    word_idx = 0  # running index of words in the section
    sent_word_starts = []  # starting word index for each sentence
    acc = 0
    for l in sent_lens:
        sent_word_starts.append(acc)
        acc += l

    i = 0
    while i < len(sentences):
        # start a chunk at i
        chunk_words = []
        chunk_word_count = 0
        j = i
        while j < len(sentences) and chunk_word_count < min_words:
            chunk_words.append(sentences[j])
            chunk_word_count += sent_lens[j]
            j += 1
            # if exceeding max_words, we may stop earlier
            if chunk_word_count >= max_words:
                break
        # join sentences to make chunk
        chunk_text = ' '.join(chunk_words).strip()
        start_word = sent_word_starts[i]
        # compute end word index (exclusive)
        end_word = sent_word_starts[min(j, len(sent_word_starts) - 1)] + (sent_lens[j - 1] if j-1 < len(sent_lens) else 0) if j-1 >= 0 else start_word + chunk_word_count
        # simpler: compute approximate end as start + chunk_word_count
        end_word = start_word + chunk_word_count
        chunks.append((chunk_text, start_word, end_word))
        # advance i with overlap
        if chunk_word_count == 0:
            # fallback protection
            i = j
        else:
            # compute overlap in sentences to move i forward
            overlap_words = int(chunk_word_count * overlap_pct)
            # find sentence index to advance to such that approx overlap_words remain from end
            # we want next_i such that sent_word_starts[next_i] >= (start_word + chunk_word_count - overlap_words)
            desired_start_word = start_word + chunk_word_count - overlap_words
            # binary search for next_i
            next_i = j
            for k in range(i, j):
                if sent_word_starts[k] >= desired_start_word:
                    next_i = k
                    break
            if next_i <= i:
                next_i = i + 1
            i = next_i
    return chunks


def chunk_pages_to_chunks(pages: List[Tuple[int, str]],
                          min_words=250,
                          max_words=400,
                          overlap_pct=0.30) -> List[Chunk]:
    """
    pages: list of (page_number, page_text)
    Splits each page into section-aware sentence chunks and returns list of Chunk dataclasses.
    """
    all_chunks: List[Chunk] = []
    global_chunk_idx = 0

    for page_num, page_text in pages:
        cleaned = clean_page_text(page_text)
        if not cleaned:
            continue

        # Split page into rough lines to find headings, but preserve original paragraphs
        lines = [ln.strip() for ln in cleaned.split('\n') if ln.strip()]
        # Build sections: whenever we detect a heading, start a new section
        section_texts = []
        section_hints = []
        current_section_lines = []
        current_section_title = None

        for ln in lines:
            if is_heading(ln):
                # flush existing section
                if current_section_lines:
                    section_texts.append('\n'.join(current_section_lines))
                    section_hints.append(current_section_title or "")
                    current_section_lines = []
                # new section starts; treat this ln as the section title
                current_section_title = ln.strip()
                # continue - we will not drop the title from the section body; include it at start
                current_section_lines.append(current_section_title)
            else:
                current_section_lines.append(ln)
        # flush last
        if current_section_lines:
            section_texts.append('\n'.join(current_section_lines))
            section_hints.append(current_section_title or "")

        # If no headings detected at all, consider whole page one section with empty hint
        if not section_texts:
            section_texts = [cleaned]
            section_hints = [""]

        # For each section, produce sentence-boundary chunks with overlap
        for sec_text, sec_hint in zip(section_texts, section_hints):
            # make sure there is enough content
            # if very small, create a single chunk
            words = sec_text.split()
            if len(words) <= max_words:
                # single chunk
                chunk_id = f"chunk-{page_num}-{global_chunk_idx}-{uuid.uuid4().hex[:6]}"
                c = Chunk(
                    chunk_id=chunk_id,
                    page=page_num,
                    section_hint=(sec_hint or ""),
                    text=sec_text.strip(),
                    word_start=0,
                    word_end=len(words)
                )
                all_chunks.append(c)
                global_chunk_idx += 1
                continue

            # else break into multiple chunks
            chunks_info = sentence_level_chunks_for_section(sec_text, min_words=min_words, max_words=max_words, overlap_pct=overlap_pct)
            for chunk_text, wstart, wend in chunks_info:
                if not chunk_text.strip():
                    continue
                chunk_id = f"chunk-{page_num}-{global_chunk_idx}-{uuid.uuid4().hex[:6]}"
                c = Chunk(
                    chunk_id=chunk_id,
                    page=page_num,
                    section_hint=(sec_hint or ""),
                    text=chunk_text.strip(),
                    word_start=wstart,
                    word_end=wend
                )
                all_chunks.append(c)
                global_chunk_idx += 1

    return all_chunks

# -----------------------
# PDF reading
# -----------------------
def extract_pages_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception as e:
                # fallback: try whole page extraction
                text = page.extract_text() or ""
            pages.append((i, text))
    return pages

# -----------------------
# Embedding + FAISS
# -----------------------
def build_and_save_index(chunks: List[Chunk],
                         model_name: str,
                         out_dir: str,
                         faiss_index_file: str = "faiss.index",
                         embeddings_file: str = "chunks_embeddings.npy",
                         chunks_jsonl: str = "chunks.jsonl",
                         batch_size: int = 64):
    os.makedirs(out_dir, exist_ok=True)
    # 1) Save chunks metadata and text
    jsonl_path = os.path.join(out_dir, chunks_jsonl)
    with open(jsonl_path, 'w', encoding='utf-8') as fo:
        for c in chunks:
            record = {
                "chunk_id": c.chunk_id,
                "page": c.page,
                "section_hint": c.section_hint,
                "text": c.text,
                "word_start": c.word_start,
                "word_end": c.word_end
            }
            fo.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2) Embed chunks
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float32')  # shape (N, dim)
    # save embeddings (optional but helpful)
    np.save(os.path.join(out_dir, embeddings_file), embeddings)

    # 3) Normalize embeddings for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # 4) Build FAISS index: IndexFlatIP (fast, exact for small corpora)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(out_dir, faiss_index_file))

    print(f"[+] Saved FAISS index -> {os.path.join(out_dir, faiss_index_file)}")
    print(f"[+] Saved chunks metadata -> {jsonl_path}")
    print(f"[+] Saved embeddings -> {os.path.join(out_dir, embeddings_file)}")
    return index, embeddings

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ingest PDF -> chunk -> embed -> FAISS index")
    p.add_argument("--pdf", type=str, default="3. FYP-Handbook-2023.pdf", help="Path to PDF")
    p.add_argument("--out_dir", type=str, default="./handbook_index", help="Output directory to save index & metadata")
    p.add_argument("--min_words", type=int, default=250, help="Minimum words per chunk")
    p.add_argument("--max_words", type=int, default=400, help="Maximum words per chunk")
    p.add_argument("--overlap_pct", type=float, default=0.30, help="Overlap percent between chunks (0-0.5 recommended)")
    p.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformers model name")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for embeddings")
    return p.parse_args()


def main():
    args = parse_args()
    pdf_path = args.pdf
    out_dir = args.out_dir
    min_words = args.min_words
    max_words = args.max_words
    overlap_pct = args.overlap_pct
    model_name = args.model_name
    batch_size = args.batch_size

    assert os.path.exists(pdf_path), f"PDF not found: {pdf_path}"

    print("[*] Extracting pages from PDF...")
    pages = extract_pages_from_pdf(pdf_path)
    print(f"[*] Extracted {len(pages)} pages")

    print("[*] Chunking pages (heading-aware, sentence-boundary)...")
    chunks = chunk_pages_to_chunks(pages, min_words=min_words, max_words=max_words, overlap_pct=overlap_pct)
    print(f"[*] Produced {len(chunks)} chunks")

    # quick sanity display
    for c in chunks[:3]:
        print(f"  sample chunk: {c.chunk_id} (p.{c.page}) sec='{c.section_hint}' words={c.word_end - c.word_start}")

    print("[*] Building embeddings & FAISS index...")
    build_and_save_index(chunks, model_name=model_name, out_dir=out_dir, batch_size=batch_size)

    print("[+] Ingest finished. Index and metadata saved to:", out_dir)


if __name__ == "__main__":
    main()
