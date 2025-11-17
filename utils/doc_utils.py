# utils/doc_utils.py
import os
import re
from typing import List, Dict, Any, Tuple, Optional
import pdfplumber

# --- Helpers to read files -------------------------------------------------
def read_pdf(path: str) -> List[Tuple[int, str]]:
    """
    Read PDF and return list of (page_number, text) tuples.
    Page numbers are 1-indexed.
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append((i + 1, text))
    return pages

def read_text_file(path: str) -> List[Tuple[int, str]]:
    """
    Read a plain text file and return a single-element list with page_number=1.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [(1, text)]

def load_document(path: str) -> List[Tuple[int, str]]:
    """
    Generic loader: based on extension, returns list of (page_num, text) tuples.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext in [".txt", ".md"]:
        return read_text_file(path)
    else:
        # fallback: try reading as text
        return read_text_file(path)

# --- Text cleaning & sentence splitting -----------------------------------
def clean_text(text: str) -> str:
    """
    Simple cleaning: normalize whitespace and remove null characters.
    Keep it minimal to avoid losing information.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[\.\?\!])\s+(?=[A-Z0-9"\'“‘])'  # split at sentence end punctuation followed by a capital (or quote)
)

def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter. Not perfect but avoids heavy deps.
    """
    text = clean_text(text)
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    # further split very long sentences by commas if needed
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # if sentence is extremely long, break heuristically at commas/semicolons
        if len(s.split()) > 120:
            parts = re.split(r'[;,]\s+', s)
            parts = [p.strip() for p in parts if p.strip()]
            result.extend(parts)
        else:
            result.append(s)
    return result

# --- Chunking logic --------------------------------------------------------
def chunk_sentences(
    sentences: List[str],
    chunk_size_words: int = 500,
    overlap_words: int = 100
) -> List[str]:
    """
    Combine sentences into chunks of approx `chunk_size_words` words, with `overlap_words`.
    Returns list of chunk texts.
    """
    if not sentences:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_count = 0

    i = 0
    while i < len(sentences):
        s = sentences[i]
        words = s.split()
        wlen = len(words)

        # if single sentence longer than chunk, break it into word-based pieces
        if wlen > chunk_size_words:
            # flush current_chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []
                current_count = 0
            # split long sentence into smaller word blocks
            words_list = words
            j = 0
            while j < len(words_list):
                piece = words_list[j:j+chunk_size_words]
                chunks.append(" ".join(piece).strip())
                # move j with overlap
                j += chunk_size_words - overlap_words if chunk_size_words > overlap_words else chunk_size_words
            i += 1
            continue

        # if adding this sentence stays within chunk_size, add it
        if current_count + wlen <= chunk_size_words or not current_chunk:
            current_chunk.append(s)
            current_count += wlen
            i += 1
        else:
            # flush chunk, then start new chunk with overlap
            chunks.append(" ".join(current_chunk).strip())
            # prepare overlap: take last `overlap_words` from current_chunk as new start
            overlap_buf = " ".join(" ".join(current_chunk).split()[-overlap_words:]) if overlap_words > 0 else ""
            current_chunk = [overlap_buf] if overlap_buf else []
            current_count = len(overlap_buf.split()) if overlap_buf else 0

    # final flush
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # filter out empties
    chunks = [c for c in chunks if c]
    return chunks

def create_chunks_from_file(
    path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Read document, split into sentences, then chunk with overlap.
    Returns list of dicts:
      {
        "id": "<source-filename>-p<page>-chunk<index>",
        "text": "...",
        "page": <page_num>,
        "source": "<filename>",
        "source_id": <source_id or None>,
        "word_count": <int>
      }
    """
    filename = os.path.basename(path)
    pages = load_document(path)
    all_chunks: List[Dict[str, Any]] = []
    for page_num, page_text in pages:
        page_text = clean_text(page_text)
        if not page_text:
            continue
        sentences = split_into_sentences(page_text)
        page_chunks = chunk_sentences(sentences, chunk_size_words=chunk_size, overlap_words=chunk_overlap)
        for idx, chunk in enumerate(page_chunks):
            chunk_meta = {
                "id": f"{filename}-p{page_num}-c{idx}",
                "text": chunk,
                "page": page_num,
                "source": filename,
                "source_id": source_id,
                "word_count": len(chunk.split())
            }
            all_chunks.append(chunk_meta)
    return all_chunks

# --- Quick utility for multi-file ingestion --------------------------------
def create_chunks_from_paths(
    paths: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Process multiple files and return a combined list of chunk dicts.
    """
    all_chunks = []
    for p in paths:
        if not os.path.exists(p):
            continue
        file_chunks = create_chunks_from_file(p, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(file_chunks)
    return all_chunks

# --- Example small helper to estimate tokens (approx) -----------------------
def approx_word_count(text: str) -> int:
    return len(text.split())

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # quick local test
    sample = "This is a test. This sentence is short. " * 100
    sents = split_into_sentences(sample)
    chks = chunk_sentences(sents, chunk_size_words=50, overlap_words=10)
    print(f"Sentences: {len(sents)}; Chunks: {len(chks)}")
