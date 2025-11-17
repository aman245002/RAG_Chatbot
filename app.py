# app.py
import os
import streamlit as st
from pathlib import Path
from config.config import config
from utils.doc_utils import create_chunks_from_paths
from models.embeddings import get_embedding_engine
from utils.vectorstore import build_index_from_embeddings, add_embeddings, load_index_and_meta, delete_vectorstore
from utils.rag import get_rag_engine
from models.llm import get_llm_client


# Ensure uploads dir exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="NeoStats — RAG Chatbot", layout="wide")
st.header("NeoStats — RAG Chatbot")

# --- Sidebar: settings & file upload ---------------------------------------
with st.sidebar:
    st.subheader("Settings")
    mode = st.radio("Response mode", ["Concise", "Detailed"])
    use_web = st.checkbox("Enable web-search fallback (SerpAPI/Google)", value=True)
    top_k = st.number_input("Retrieval top_k (local)", min_value=1, max_value=20, value=4, step=1)
    web_top_k = st.number_input("Web search top_k", min_value=1, max_value=10, value=4, step=1)

    st.markdown("---")
    st.subheader("Upload & Indexing")
    uploaded_files = st.file_uploader("Upload document(s) (pdf/txt/md). After upload click Build Index.", accept_multiple_files=True, type=["pdf", "txt", "md"])
    rebuild = st.checkbox("Force rebuild index (overwrite)", value=False)
    if st.button("Build / Rebuild Index"):
        if not uploaded_files:
            st.warning("Please upload one or more files first.")
        else:
            saved_paths = []
            for f in uploaded_files:
                out_path = UPLOAD_DIR / f.name
                with open(out_path, "wb") as wf:
                    wf.write(f.getbuffer())
                saved_paths.append(str(out_path))
            # chunk docs
            st.info("Creating chunks from uploaded documents...")
            chunks = create_chunks_from_paths(saved_paths, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            st.write(f"Created {len(chunks)} chunks across {len(saved_paths)} files.")
            if not chunks:
                st.error("No chunks created. Check the files.")
            else:
                with st.spinner("Computing embeddings (this may take a while)..."):
                    emb_engine = get_embedding_engine()
                    embeddings, ids = emb_engine.embed_chunks(chunks)
                # build metadata list (include text inside meta so RAG can present it)
                metadatas = []
                for c in chunks:
                    m = {k: v for k, v in c.items() if k != "text"}
                    m["text"] = c.get("text", "")  # ensure text preserved
                    metadatas.append(m)
                try:
                    if rebuild:
                        build_index_from_embeddings(embeddings, ids, metadatas, overwrite=True)
                        st.success(f"Index rebuilt with {len(ids)} vectors.")
                    else:
                        # try append if index exists
                        index, id_list, id_to_meta = load_index_and_meta()
                        if index is None:
                            build_index_from_embeddings(embeddings, ids, metadatas, overwrite=True)
                            st.success(f"Index built with {len(ids)} vectors.")
                        else:
                            add_embeddings(embeddings, ids, metadatas)
                            st.success(f"Added {len(ids)} vectors to existing index.")
                except Exception as e:
                    st.error(f"Failed to build index: {e}")

    st.markdown("---")
    if st.button("Delete local vectorstore (dev)"):
        delete_vectorstore()
        st.success("Local vectorstore deleted.")

    st.markdown("---")
    st.caption("Do not commit API keys. Set them in .env (local) or Streamlit secrets (deployment).")

# --- Main chat UI ----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"role":"user"/"assistant","text":...}

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Chat")
    query = st.text_input("Enter your question about uploaded docs or general query:")
    send = st.button("Send")

    if send and query.strip():
        # instantiate RAG engine with current settings
        # instantiate Groq LLM client
        llm_client = get_llm_client(provider="groq", model="llama-3.3-70b-versatile")

        rag = get_rag_engine(
            llm=llm_client,
            top_k=top_k,
            web_top_k=web_top_k,
            web_enabled=use_web
        )

        # record user message
        st.session_state.history.append({"role": "user", "text": query})
        with st.spinner("Retrieving and generating answer..."):
            res = rag.answer_question(query, mode=mode, force_web=False)
        ans = res.get("answer", "[No answer]")
        # store assistant message
        st.session_state.history.append({"role": "assistant", "text": ans, "meta": res})
        # display answer
        st.markdown("**Answer:**")
        st.write(ans)

        # show sources
        doc_sources = res.get("used_document_sources", [])
        web_res = res.get("used_web_results", [])
        if doc_sources:
            st.markdown("**Document sources used:**")
            for s in doc_sources:
                st.write("-", s)
        if web_res:
            st.markdown("**Web results used:**")
            for r in web_res[:5]:
                title = r.get("title") or r.get("link")
                snippet = r.get("snippet") or ""
                link = r.get("link") or ""
                st.write(f"- [{title}]({link})  — {snippet}")

with col2:
    st.subheader("Conversation")
    if st.button("Clear history"):
        st.session_state.history = []
    # display compact history
    for i, item in enumerate(reversed(st.session_state.history[-30:])):
        role = item.get("role", "")
        txt = item.get("text", "")
        if role == "user":
            st.info(f"User: {txt}")
        else:
            st.success(f"Assistant: {txt}")
            # small "show provenance" expander
            meta = item.get("meta", {})
            if meta:
                with st.expander("Show details (retrieval + web)"):
                    st.write("Raw retrievals:")
                    for r in meta.get("raw_retrievals", [])[:8]:
                        cid = r.get("chunk_id")
                        dist = r.get("distance")
                        m = r.get("meta", {})
                        src = m.get("source", "unknown")
                        page = m.get("page", "?")
                        excerpt = m.get("text", "")[:400]
                        st.write(f"- {cid} (dist={dist:.4f}) — {src}:p{page}")
                        st.write("  >", excerpt)
                    if meta.get("web_summary"):
                        st.write("Web summary (short):")
                        st.write(meta.get("web_summary")[:1000])

# --- Footer / notes --------------------------------------------------------
st.markdown("---")
st.write("Tips: 1) Upload PDFs and Build Index before asking doc-specific questions. 2) Toggle web-search fallback if you want live web info.")
