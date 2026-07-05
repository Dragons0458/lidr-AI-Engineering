"""Corpus index panel — stats and incremental expansion (Session 11)."""

from __future__ import annotations

import json
import runpy
import time
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import streamlit as st

from streamlit_ui.common import format_api_error, get_api_root_url
from streamlit_ui.rag import corpus_index_job, corpus_index_run, corpus_stats

st.set_page_config(page_title="Corpus Index", page_icon="📚", layout="wide")

api_root = get_api_root_url()
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

st.title("Corpus Index")
st.caption(f"`GET {api_root}/embeddings/index/stats` · `POST /embeddings/index/runs`")

if st.button("Refresh stats"):
    st.cache_data.clear()

try:
    stats = corpus_stats(api_root)
except Exception as exc:  # noqa: BLE001
    st.error(format_api_error(exc, api_base_url=api_root))
    st.stop()

st.subheader("Collections")
rows = [
    {
        "collection": c["collection"],
        "documents": c["documents"],
        "chunks": c["chunks"],
        "hnsw_indexed": c["hnsw_indexed"],
    }
    for c in stats.get("collections", [])
]
st.dataframe(rows, use_container_width=True, hide_index=True)
st.metric("Total chunks", stats.get("total_chunks", 0))

st.divider()
st.subheader("Expand corpus")
uploaded = st.file_uploader("Upload budgets JSON array", type=["json"])
default_path = DATA_DIR / "budgets_sample.json"
use_sample = st.checkbox("Use budgets_sample.json", value=uploaded is None)

documents: list[dict] = []
if uploaded is not None:
    documents = json.loads(uploaded.getvalue().decode("utf-8"))
elif use_sample and default_path.exists():
    documents = json.loads(default_path.read_text(encoding="utf-8"))

st.caption(f"{len(documents)} document(s) selected")

if st.button("Start index run", type="primary", disabled=not documents):
    try:
        run = corpus_index_run(api_root, documents=documents[:200])
        job_id = run["job_id"]
        st.session_state["corpus_job_id"] = job_id
        st.success(f"Job {job_id} started ({run['status']}).")
    except Exception as exc:  # noqa: BLE001
        st.error(format_api_error(exc, api_base_url=api_root))

job_id = st.session_state.get("corpus_job_id")
if job_id:
    st.subheader("Job status")
    if st.button("Poll job"):
        try:
            job = corpus_index_job(api_root, str(job_id))
            st.json(job)
            if job.get("status") in ("pending", "running"):
                time.sleep(1)
                st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(format_api_error(exc, api_base_url=api_root))
