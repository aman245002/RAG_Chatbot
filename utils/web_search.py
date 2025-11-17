# utils/web_search.py
import os
import time
import requests
from typing import List, Dict, Any, Optional
from config.config import config

# Optional LLM summarizer integration
try:
    from models.llm import OpenAIClient
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False

SERPAPI_URL = "https://serpapi.com/search.json"
GOOGLE_CX_ENV = "GOOGLE_CX"  # You must set this in .env or Streamlit secrets if you want to use Google Custom Search

def _serpapi_search(query: str, top_k: int = 5, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Use SerpAPI (https://serpapi.com/) to run a google search. Returns list of results.
    Each result: {"title","snippet","link","source"}
    """
    key = api_key or config.SERPAPI_API_KEY
    if not key:
        return []
    params = {
        "q": query,
        "api_key": key,
        "engine": "google",
        "num": top_k
    }
    resp = requests.get(SERPAPI_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = []
    # SerpAPI organizes results under 'organic_results'
    for r in data.get("organic_results", [])[:top_k]:
        title = r.get("title") or r.get("position") or ""
        snippet = r.get("snippet") or r.get("snippet_highlighted_words") or r.get("rich_snippet", {}).get("top", {}).get("snippet", "")
        link = r.get("link") or r.get("displayed_link") or r.get("serpapi_link")
        results.append({"title": title, "snippet": snippet or "", "link": link or "", "source": link or ""})
    return results

def _google_custom_search(query: str, top_k: int = 5, api_key: Optional[str] = None, cx: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Google Custom Search JSON API fallback. Needs both api_key and cx (search engine id).
    """
    key = api_key or config.GOOGLE_API_KEY
    cx = cx or os.getenv("GOOGLE_CX") or getattr(config, "GOOGLE_CX", None)
    if not key or not cx:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": key, "cx": cx, "num": min(10, top_k)}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("items", [])[:top_k]:
        title = item.get("title")
        snippet = item.get("snippet")
        link = item.get("link")
        results.append({"title": title, "snippet": snippet or "", "link": link or "", "source": link or ""})
    return results

def _fetch_page_text(url: str, max_chars: int = 2000) -> str:
    """
    Best-effort fetch of a page's text. Strips HTML tags minimally.
    Not a full-featured extractor (no readability), but better than nothing.
    """
    if not url:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; NeoStatsBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        text = r.text
        # strip scripts/styles
        import re
        text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", text)
        # remove HTML tags
        text = re.sub(r"(?s)<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""

def perform_search(query: str, top_k: int = 5, prefer: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Run a web search and return the top_k results. Preference order:
      - if prefer == "serpapi" -> try SerpAPI only
      - if prefer == "google" -> try Google Custom Search only
      - if prefer is None -> try SerpAPI, then Google Custom Search
    Each returned result dict: {"title","snippet","link","source","page_text"(optional)}
    """
    results = []
    # try serpapi first (recommended)
    if prefer == "serpapi" or prefer is None:
        try:
            results = _serpapi_search(query, top_k=top_k)
        except Exception:
            results = []
    # fallback to google custom search
    if (not results) and (prefer == "google" or prefer is None):
        try:
            results = _google_custom_search(query, top_k=top_k)
        except Exception:
            results = []

    # fetch page text if snippet missing
    enriched = []
    for r in results:
        snippet = r.get("snippet") or ""
        if not snippet and r.get("link"):
            page_txt = _fetch_page_text(r.get("link"), max_chars=1500)
        else:
            page_txt = ""  # we will not fetch if snippet present to save time
        item = {
            "title": r.get("title", ""),
            "snippet": snippet,
            "link": r.get("link", ""),
            "source": r.get("source", "") or r.get("link", ""),
            "page_text": page_txt
        }
        enriched.append(item)
    return enriched

# --- Summarization helper using LLM ---------------------------------------
def summarize_search_results(
    query: str,
    results: List[Dict[str, Any]],
    mode: str = "concise",
    llm_client: Optional[Any] = None
) -> str:
    """
    Create a short combined summary of the search results useful for RAG context.
    If llm_client is provided (OpenAIClient), uses LLM to create a coherent summary.
    Otherwise, it concatenates titles+snippets.
    Returns a single string summary.
    """
    if not results:
        return ""

    if llm_client is None:
        # simple concatenation fallback
        parts = []
        for r in results:
            t = r.get("title", "")
            s = r.get("snippet", "") or (r.get("page_text")[:400] if r.get("page_text") else "")
            parts.append(f"{t}\n{s}\nLINK: {r.get('link')}\n")
        if mode and mode.lower().startswith("d"):
            return "\n\n".join(parts)
        # concise: join top 3
        return "\n".join([p for p in parts[:3]])

    # Use LLM to summarize (prefer OpenAIClient-like interface)
    if not _LLM_AVAILABLE:
        # fall back to simple concatenation
        return summarize_search_results(query, results, mode=mode, llm_client=None)

    try:
        client = llm_client or OpenAIClient()
        # Build prompt
        snippets = []
        for i, r in enumerate(results[:6]):
            txt = r.get("snippet") or r.get("page_text") or ""
            snippets.append(f"Result {i+1} TITLE: {r.get('title','')}\nSNIPPET: {txt}\nURL: {r.get('link')}\n")
        prompt = f"""You are an assistant that summarizes web search results for the query: "{query}".
You will produce a {mode} summary suitable for supplying as additional context to an LLM answering the query.
Summarize the following search result excerpts, do not hallucinate facts, and include short citations (result numbers or urls).
========
{''.join(snippets)}
========
Provide a {mode} summary (2-4 sentences for concise; a few paragraphs for detailed).
"""
        # call LLM
        out = client.generate(prompt, max_tokens=400, temperature=0.0)
        # small post-processing
        return out.strip()
    except Exception:
        # on errors, fallback to concatenation
        return summarize_search_results(query, results, mode=mode, llm_client=None)

# --- Combined helper for RAG integration -----------------------------------
def search_and_prepare_context(query: str, top_k: int = 5, mode: str = "concise", prefer: Optional[str] = None, use_llm_summarizer: bool = True) -> Dict[str, Any]:
    """
    Convenience function:
      - performs web search
      - summarizes results (optionally using LLM)
      - returns a context dict ready to be appended to RAG context

    Returns:
      {
        "query": query,
        "results": [ {title,snippet,link,source,page_text}, ... ],
        "summary": <summary text>
      }
    """
    results = perform_search(query, top_k=top_k, prefer=prefer)
    summary = ""
    if results:
        if use_llm_summarizer and _LLM_AVAILABLE:
            try:
                summary = summarize_search_results(query, results, mode=mode, llm_client=OpenAIClient())
            except Exception:
                summary = summarize_search_results(query, results, mode=mode, llm_client=None)
        else:
            summary = summarize_search_results(query, results, mode=mode, llm_client=None)
    return {"query": query, "results": results, "summary": summary}

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # quick demo (requires SERPAPI_API_KEY or GOOGLE_API_KEY + GOOGLE_CX)
    q = "what is retrieval augmented generation (RAG) in NLP"
    res = search_and_prepare_context(q, top_k=4, mode="concise")
    print("SUMMARY:\n", res["summary"][:500])
    print("TOP RESULTS:")
    for r in res["results"]:
        print("-", r.get("title"), r.get("link"))
