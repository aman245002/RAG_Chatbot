# utils/rag.py
import traceback
from typing import List, Dict, Any, Optional, Tuple
from models.embeddings import get_embedding_engine
from models.llm import OpenAIClient
from utils.vectorstore import query_index, load_index_and_meta
from utils.web_search import search_and_prepare_context
from config.config import config

# Prompt templates
_PROMPT_TEMPLATE = """
You are an assistant that answers user questions using the provided CONTEXT excerpts from documents and optional web search summary.
When answering, you MUST:
- Use the provided CONTEXT and WEB_SUMMARY for factual claims and cite source filenames/pages or web result indicators when used.
- If the context and web summary do NOT contain the answer, say "I don't know from the provided documents and web results."
- Keep your tone helpful. Follow the requested mode (concise/detailed).

CONTEXT_FROM_DOCUMENTS:
{context}

WEB_SEARCH_SUMMARY:
{web_summary}

QUESTION:
{question}

INSTRUCTIONS:
- If mode is "concise", answer in 2-4 sentences.
- If mode is "detailed", provide a thorough answer, include steps, and mention which context excerpts (by source and page) and/or web results you used.
- At the end include a short "SOURCES" section listing used source ids/pages and web result URLs or titles.
Answer now.
"""

_FALLBACK_PROMPT = """
You are an assistant responding to the user's question without document context.

QUESTION:
{question}

INSTRUCTIONS:
- Answer in a helpful way.
- If you are uncertain, say "I don't have enough information to answer that from available documents and web results."
- Use mode = {mode} (concise or detailed).
"""

class RAGEngine:
    def __init__(self, llm: Optional[OpenAIClient] = None, embedding_engine = None, top_k: int = 4, web_top_k: int = 4, web_enabled: bool = True):
        self.llm = llm or OpenAIClient(model=getattr(config, "OPENAI_MODEL", None))
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.top_k = top_k
        self.web_top_k = web_top_k
        self.web_enabled = web_enabled

    def _format_context(self, hits: List[Tuple[str, float, Dict[str, Any]]]) -> Tuple[str, List[str]]:
        """
        Format retrieval hits into a single context string.
        Returns (context_text, used_sources).
        """
        blocks = []
        used_sources = []
        for cid, dist, meta in hits:
            # meta is expected to include 'text','source','page' if you saved it
            text = ""
            if meta:
                text = meta.get("text") or meta.get("excerpt") or ""
            if not text:
                text = f"[Content for {cid} not persisted in metadata.]"
            source = meta.get("source", "unknown") if meta else "unknown"
            page = meta.get("page", "?") if meta else "?"
            blocks.append(f"---\nID: {cid}\nSOURCE: {source} PAGE: {page}\n{text}\n")
            used_sources.append(f"{source}:p{page}")
        # deduplicate
        used_sources = list(dict.fromkeys(used_sources))
        return ("\n".join(blocks), used_sources)

    def _should_use_web(self, hits: List[Tuple[str, float, Dict[str, Any]]]) -> bool:
        """
        Heuristic to decide whether to call web search:
        - No hits -> True
        - If distances are large (non-expert heuristic) -> True
        """
        if not hits:
            return True
        # Examine distances; FAISS L2 distances vary by embedding model.
        # If top-1 distance is unusually large relative to next, or greater than threshold, fallback
        dists = [h[1] for h in hits if h[1] is not None]
        if not dists:
            return True
        # Simple threshold heuristic (tuneable): if best distance > 1.0 -> call web (works for many embedding dims)
        try:
            best = dists[0]
            if best > 1.0:
                return True
        except Exception:
            return True
        return False

    def answer_question(self, question: str, mode: str = "Concise", force_web: bool = False) -> Dict[str, Any]:
        """
        Main API: run RAG to answer question, optionally fallback to web search.
        Returns:
          {
            "answer": str,
            "used_document_sources": [...],
            "used_web_results": [...],
            "raw_retrievals": [...],
            "web_results": [...]
          }
        """
        try:
            index, id_list, id_to_meta = load_index_and_meta()
            # If no index -> attempt web only if enabled
            if index is None or id_to_meta is None:
                if self.web_enabled:
                    web_ctx = search_and_prepare_context(question, top_k=self.web_top_k, mode=mode, use_llm_summarizer=True)
                    prompt = _PROMPT_TEMPLATE.format(context="[NO LOCAL DOCUMENTS AVAILABLE]", web_summary=web_ctx.get("summary",""), question=question)
                    prompt = f"[MODE: {mode}]\n" + prompt
                    max_tokens = 700 if mode.lower().startswith("d") else 220
                    ans = self.llm.generate(prompt, max_tokens=max_tokens, temperature=0.2)
                    return {"answer": ans, "used_document_sources": [], "used_web_results": web_ctx.get("results", []), "raw_retrievals": [], "web_summary": web_ctx.get("summary","")}
                else:
                    prompt = _FALLBACK_PROMPT.format(question=question, mode=mode.lower())
                    return {"answer": self.llm.generate(prompt, max_tokens=300, temperature=0.2), "used_document_sources": [], "used_web_results": [], "raw_retrievals": [], "web_summary": ""}

            # Compute embedding and retrieve
            q_emb = self.embedding_engine.embed_texts([question])[0]
            hits = query_index(q_emb, top_k=self.top_k)
            # If hits are returned, each item is (chunk_id, distance, meta)
            use_web = force_web or (self.web_enabled and self._should_use_web(hits))
            web_results = []
            web_summary = ""
            if use_web:
                # perform web search and prepare summary (use LLM summarizer)
                web_ctx = search_and_prepare_context(question, top_k=self.web_top_k, mode=mode, use_llm_summarizer=True)
                web_results = web_ctx.get("results", [])
                web_summary = web_ctx.get("summary", "")

            # Build context from document hits (if any)
            context_text = ""
            used_doc_sources = []
            if hits:
                context_text, used_doc_sources = self._format_context(hits)
            else:
                context_text = "[NO RELEVANT LOCAL DOCUMENT FRAGMENTS FOUND]"

            # Combine contexts: prefer local docs first then web summary
            combined_web_summary = web_summary if web_summary else "[NO WEB SUMMARY]"
            prompt = _PROMPT_TEMPLATE.format(context=context_text, web_summary=combined_web_summary, question=question)
            prompt = f"[MODE: {mode}]\n" + prompt

            # LLM settings
            if mode.lower().startswith("d"):
                max_tokens = 700
                temp = 0.2
            else:
                max_tokens = 220
                temp = 0.0

            answer = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temp)

            return {
                "answer": answer,
                "used_document_sources": used_doc_sources,
                "used_web_results": web_results,
                "raw_retrievals": [{"chunk_id": cid, "distance": dist, "meta": meta} for cid, dist, meta in hits],
                "web_summary": web_summary
            }
        except Exception as e:
            traceback.print_exc()
            return {"answer": f"[RAG ERROR] {str(e)}", "used_document_sources": [], "used_web_results": [], "raw_retrievals": [], "web_summary": ""}


# Convenience factory
def get_rag_engine(llm: Optional[OpenAIClient] = None, embedding_engine = None, top_k: int = 4, web_top_k: int = 4, web_enabled: bool = True) -> RAGEngine:
    return RAGEngine(llm=llm, embedding_engine=embedding_engine, top_k=top_k, web_top_k=web_top_k, web_enabled=web_enabled)


# Quick test (requires index or SERPAPI/GOOGLE keys)
if __name__ == "__main__":
    rag = get_rag_engine()
    q = "What is retrieval augmented generation (RAG)?"
    res = rag.answer_question(q, mode="Concise")
    print("Answer:", res["answer"])
    print("Docs:", res["used_document_sources"])
    print("Web results count:", len(res["used_web_results"]) if res["used_web_results"] else 0)
