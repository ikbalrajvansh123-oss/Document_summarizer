import streamlit as st

import rag_core as core

st.set_page_config(page_title="Local RAG", page_icon="🧠", layout="wide")

# Cached resources (loaded once per session)
@st.cache_resource
def init_resources():
    core.get_embedder()
    core.get_collection()
    return True


init_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"bot", "content": str, "sources": [...]}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Local RAG")

    lm_ok = core.check_lm_studio_health()
    if lm_ok:
        st.success("LM Studio connected", icon="✅")
    else:
        st.error("LM Studio not reachable (check http://127.0.0.1:1234)", icon="⚠️")

    st.divider()
    st.markdown("### 📁 Folder: `uploads/`")
    st.caption(
        "Drop `.txt` files into the **uploads/** folder, then click **Re-scan** to index them."
    )

    if st.button("⟳ Re-scan Folder", use_container_width=True):
        with st.spinner("Scanning uploads/ folder..."):
            result = core.scan_folder()
        n_new = len(result["ingested"])
        n_skip = len(result["skipped"])
        n_err = len(result["errors"])
        if n_new:
            st.success(f"✓ {n_new} new file(s) ingested.")
        if n_skip:
            st.info(f"{n_skip} file(s) already indexed (skipped).")
        if n_err:
            for e in result["errors"]:
                st.error(f"⚠ {e['filename']}: {e['error']}")
        if not n_new and not n_err:
            st.info("No new files found.")
        st.rerun()

    st.divider()
    st.markdown("### 📄 Indexed Files")

    files = core.list_files()
    if not files:
        st.caption("No files indexed yet.")
    else:
        for f in files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{f['filename']}**")
                st.caption(f"{f['num_chunks']} chunks")
            with col2:
                if st.button("🗑️", key=f"del_{f['file_id']}", help="Remove from index (file on disk is kept)"):
                    core.delete_file(f["file_id"])
                    st.rerun()

    st.divider()
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=core.TOP_K_DEFAULT)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main: chat interface ───────────────────────────────────────────────────────
st.title("Ask your documents")

if not files:
    st.info("Add `.txt` files to the `uploads/` folder and click **⟳ Re-scan Folder** in the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} source chunk(s)"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['filename']}** · chunk #{s['chunk_index']}")
                    st.caption(s["preview"] + "...")

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = core.query(question, top_k=top_k)
                confidence_pct = int(result.get("confidence", 1.0) * 100)
                if result.get("mode") == "full_document":
                    st.caption("🔎 Full-document mode (count/list/summary type question detected)")
                elif result.get("mode") == "not_found_in_document":
                    st.caption(f"⚠️ Confidence {confidence_pct}% (below 75% threshold) — logged to unanswered.json")
                else:
                    st.caption(f"✅ Confidence {confidence_pct}%")
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander(f"{len(result['sources'])} source chunk(s)"):
                        for s in result["sources"]:
                            st.markdown(f"**{s['filename']}** · chunk #{s['chunk_index']}")
                            st.caption(s["preview"] + "...")
                st.session_state.messages.append(
                    {"role": "bot", "content": result["answer"], "sources": result.get("sources", [])}
                )
            except ValueError as e:
                st.warning(str(e))
            except core.LMStudioError as e:
                st.error(str(e))
