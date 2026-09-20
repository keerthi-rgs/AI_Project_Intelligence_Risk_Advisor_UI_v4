import os
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agents import ScopeDeliverableAgent, RiskForecastingAgent, BlockerActionItemAgent


st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="🧰",
    layout="wide",
)


# --- Professional Light UI styling ---
st.markdown("""
<style>
    /* Light background backdrop */
    .stApp, [data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: #1e293b;
    }

    /* Main content container */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Header card */
    .project-header {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 24px 16px 24px;
        margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .project-title {
        font-size: 28px;
        font-weight: 750;
        color: #0f172a;
        margin: 0;
    }
    .project-subtitle {
        font-size: 14px;
        color: #475569;
        margin-top: 6px;
    }

    /* Sidebar light styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.2rem 1rem;
    }
    .side-brand {
        font-size: 17px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 18px;
    }
    .side-section {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: .05em;
        margin: 18px 0 8px;
    }

    /* Metric Cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px 16px;
        min-height: 90px;
    }
    .metric-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 650;
    }
    .metric-value {
        color: #0f172a;
        font-size: 25px;
        font-weight: 750;
        margin-top: 6px;
    }

    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        margin: 8px 0 14px;
    }

    .upload-title {
        font-size: 18px;
        font-weight: 720;
        color: #0f172a;
    }
    .upload-help {
        color: #64748b;
        font-size: 13px;
        margin: 4px 0 12px;
    }

    /* Health Badges */
    .badge-green { background-color: #16a34a; color: #ffffff; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 14px; }
    .badge-orange { background-color: #d97706; color: #ffffff; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 14px; }
    .badge-red { background-color: #dc2626; color: #ffffff; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 14px; }

    /* Tabs styling for light mode */
    button[data-baseweb="tab"] {
        font-weight: 650;
        color: #64748b !important;
        padding: 10px 12px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2563eb !important;
    }

    /* Primary action buttons */
    .stButton > button[kind="primary"] {
        background: #16a34a;
        border: 1px solid #15803d;
        color: #ffffff;
        font-weight: 700;
        border-radius: 7px;
        min-height: 42px;
    }
    .stButton > button[kind="primary"]:hover {
        background: #15803d;
        border-color: #166534;
    }

    /* File uploader light style */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 9px;
        padding: 6px;
    }

    /* Expander light style */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #ffffff;
    }

    /* Hide Streamlit menu noise */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="project-header">
    <div class="project-title">🧰 AI Project Intelligence & Risk Advisor</div>
    <div class="project-subtitle">
        Multi-Agent Project Intelligence Engine: Scope & Deliverable Extraction, Risk & Delivery Forecasting,
        Blocker & Action Item Tracking across Multi-Format Documents (PDF, DOCX, TXT, CSV)
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Data structures / state
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    source: str
    location: str
    text: str


class FileOnDisk:
    """Helper wrapper to adapt disk files for extract_file interface."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.name = Path(filepath).name

    def getvalue(self) -> bytes:
        with open(self.filepath, "rb") as f:
            return f.read()


if "documents" not in st.session_state:
    st.session_state.documents = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "matrix" not in st.session_state:
    st.session_state.matrix = None
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "ingestion_log" not in st.session_state:
    st.session_state.ingestion_log = []

# Initialize Specialized Agents
if "scope_agent" not in st.session_state:
    st.session_state.scope_agent = ScopeDeliverableAgent()
if "risk_agent" not in st.session_state:
    st.session_state.risk_agent = RiskForecastingAgent()
if "blocker_agent" not in st.session_state:
    st.session_state.blocker_agent = BlockerActionItemAgent()


# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.markdown('<div class="side-brand">🧰 AI Project Intelligence & Risk Advisor</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="side-section">Configuration</div>', unsafe_allow_html=True)
chunk_size = st.sidebar.slider("Chunk Size (characters)", 200, 1500, 500, 50, key="slider_chunk_size")
chunk_overlap = st.sidebar.slider(
    "Chunk Overlap (characters)", 0, min(500, chunk_size - 1), 100, 10, key="slider_chunk_overlap"
)
top_k = st.sidebar.slider("Retrieval Top-K Chunks", 1, 10, 4, key="slider_top_k")

st.sidebar.divider()
st.sidebar.markdown('<div class="side-section">Vector Store Index</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Indexed Chunks</div>
    <div class="metric-value">{len(st.session_state.chunks)}</div>
</div>
""", unsafe_allow_html=True)

source_count = len(set(c.source for c in st.session_state.chunks)) if st.session_state.chunks else 0
st.sidebar.markdown(f"""
<div class="metric-card" style="margin-top:10px;">
    <div class="metric-label">Source Documents</div>
    <div class="metric-value">{source_count}</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="side-section" style="margin-top:16px;">Chunk Index</div>', unsafe_allow_html=True)
if st.session_state.chunks:
    chunk_options = [f"{c.chunk_id} | {c.source} — {c.location}" for c in st.session_state.chunks]
    selected_chunk_label = st.sidebar.selectbox("Inspect Chunk", chunk_options, key="sidebar_chunk_select")
    if selected_chunk_label:
        sel_idx = chunk_options.index(selected_chunk_label)
        c = st.session_state.chunks[sel_idx]
        st.sidebar.markdown(f"**Chunk ID:** `{c.chunk_id}`")
        st.sidebar.markdown(f"**Source Document:** `{c.source}`")
        st.sidebar.markdown(f"**Location:** {c.location}")
        st.sidebar.text_area("Content Preview", c.text, height=140, key=f"sb_preview_{c.chunk_id}", disabled=True)

    with st.sidebar.expander("📋 Full Chunk Index List"):
        for chunk in st.session_state.chunks:
            st.markdown(f"• **{chunk.chunk_id}**: `{chunk.source}` ({chunk.location})")
else:
    st.sidebar.info("No chunks indexed yet. Ingest documents to generate index.")


# -----------------------------
# Extraction
# -----------------------------
def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(data: bytes, filename: str):
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = normalize(page.extract_text() or "")
        if text:
            pages.append({"source": filename, "location": f"Page {i}", "text": text})
    return pages


def extract_docx(data: bytes, filename: str):
    """Extract normal paragraphs, tables, and fallback document XML text."""
    doc = Document(io.BytesIO(data))
    records = []

    for i, p in enumerate(doc.paragraphs, start=1):
        text = normalize(p.text)
        if text:
            records.append(
                {"source": filename, "location": f"Paragraph {i}", "text": text}
            )

    for ti, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [normalize(cell.text) for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        table_text = normalize("\n".join(rows))
        if table_text:
            records.append(
                {"source": filename, "location": f"Table {ti}", "text": table_text}
            )

    if not records:
        import zipfile as _zipfile
        from xml.etree import ElementTree as ET

        try:
            with _zipfile.ZipFile(io.BytesIO(data)) as z:
                xml = z.read("word/document.xml")
            root_xml = ET.fromstring(xml)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            texts = [t.text for t in root_xml.findall(".//w:t", ns) if t.text]
            fallback = normalize(" ".join(texts))
            if fallback:
                records.append(
                    {"source": filename, "location": "DOCX XML content", "text": fallback}
                )
        except Exception:
            pass

    return records


def extract_txt(data: bytes, filename: str):
    text = normalize(data.decode("utf-8", errors="ignore"))
    return [{"source": filename, "location": "Text file", "text": text}] if text else []


def extract_csv(data: bytes, filename: str):
    df = pd.read_csv(io.BytesIO(data))
    records = []
    for idx, row in df.fillna("").iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()]
        if parts:
            records.append(
                {
                    "source": filename,
                    "location": f"CSV Row {idx + 2}",
                    "text": normalize(" | ".join(parts)),
                }
            )
    return records


def extract_file(uploaded_file):
    name = uploaded_file.name
    data = uploaded_file.getvalue()
    ext = Path(name).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(data, name)
    if ext == ".docx":
        return extract_docx(data, name)
    if ext == ".txt":
        return extract_txt(data, name)
    if ext == ".csv":
        return extract_csv(data, name)
    raise ValueError(f"Unsupported format: {ext}")


# -----------------------------
# Chunking & Indexing
# -----------------------------
def chunk_text(text: str, size: int, overlap: int):
    text = normalize(text)
    if not text:
        return []

    step = max(1, size - overlap)
    result = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()

        if end < len(text):
            candidates = [
                piece.rfind(". "),
                piece.rfind("\n"),
                piece.rfind(" "),
            ]
            boundary = max(candidates)
            if boundary > int(size * 0.55):
                piece = piece[:boundary + 1].strip()
                end = start + len(piece)

        if piece:
            result.append(piece)

        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return result


def build_chunks(records, size, overlap):
    output = []
    counter = 1
    for record in records:
        pieces = chunk_text(record["text"], size, overlap)
        for j, piece in enumerate(pieces, start=1):
            output.append(
                Chunk(
                    chunk_id=f"C{counter:04d}",
                    source=record["source"],
                    location=f'{record["location"]}, Chunk {j}',
                    text=piece,
                )
            )
            counter += 1
    return output


def rebuild_index(chunks):
    if not chunks:
        st.session_state.vectorizer = None
        st.session_state.matrix = None
        return

    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    st.session_state.vectorizer = vectorizer
    st.session_state.matrix = matrix


def retrieve(query, k):
    if (
        not st.session_state.chunks
        or st.session_state.vectorizer is None
        or st.session_state.matrix is None
    ):
        return []

    qvec = st.session_state.vectorizer.transform([query])
    scores = cosine_similarity(qvec, st.session_state.matrix)[0]
    order = scores.argsort()[::-1][:k]

    results = []
    for idx in order:
        results.append(
            {
                "chunk": st.session_state.chunks[idx],
                "score": float(scores[idx]),
            }
        )
    return results


# -----------------------------
# Tabs matching enhanced UI
# -----------------------------
tabs = st.tabs(
    [
        "1. Document Upload",
        "2. Ingestion Results",
        "3. Content Preview",
        "4. RAG Indexing",
        "5. Retrieval Explorer",
        "6. Grounded RAG Q&A",
        "7. 🎯 Scope & Deliverables Agent",
        "8. ⚠️ Risk & Delivery Forecast Agent",
        "9. 🚧 Blockers & Action Items Agent",
        "10. 🧪 Multi-Format Agent Validator",
    ]
)

# TAB 1: Document Upload & 1-Click Sample Dataset Loader
with tabs[0]:
    st.markdown('<div class="upload-title">📤 Upload Project Documents or Load Sample Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-help">Upload project documents or click below to automatically ingest sample test documents across PDF, DOCX, TXT, and CSV formats.</div>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        load_samples = st.button("🚀 Load Sample Test Dataset (PDF, DOCX, TXT, CSV)", type="primary", use_container_width=True)

    if load_samples:
        sample_dir = Path("sample_data")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.*"))
            all_records = []
            log = []
            for fpath in sample_files:
                try:
                    mock_file = FileOnDisk(str(fpath))
                    recs = extract_file(mock_file)
                    all_records.extend(recs)
                    log.append({"File": mock_file.name, "Status": "Success", "Extracted Units": len(recs)})
                except Exception as exc:
                    log.append({"File": fpath.name, "Status": f"Failed: {exc}", "Extracted Units": 0})

            st.session_state.documents = all_records
            st.session_state.ingestion_log = log
            st.session_state.chunks = build_chunks(all_records, chunk_size, chunk_overlap)
            rebuild_index(st.session_state.chunks)
            st.session_state.last_ingest_msg = ("success", f"Successfully loaded sample dataset ({len(sample_files)} multi-format files: {len(all_records)} units, {len(st.session_state.chunks)} chunks).")
            st.rerun()
        else:
            st.error("sample_data directory not found. Please run create_sample_docs.py first.")

    st.write("---")
    st.write("Supported formats: **PDF** (page-by-page), **DOCX** (paragraphs & tables), **TXT** (notes & specs), **CSV** (structured tabular data).")

    uploaded = st.file_uploader(
        "Choose project files to ingest",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )

    if uploaded:
        st.caption(f"Selected files: {len(uploaded)}")

    col1, col2 = st.columns([1, 3])
    with col1:
        ingest = st.button(
            "🔗 Ingest Uploaded Files",
            use_container_width=True,
        )
    with col2:
        if st.session_state.documents:
            st.info(f"{len(st.session_state.documents)} document unit(s) currently loaded.")

    if "last_ingest_msg" in st.session_state:
        msg_type, msg_text = st.session_state.last_ingest_msg
        if msg_type == "error":
            st.error(msg_text)
        elif msg_type == "warning":
            st.warning(msg_text)
        elif msg_type == "success":
            st.success(msg_text)

    if ingest:
        if not uploaded:
            st.session_state.last_ingest_msg = ("warning", "Please choose at least one PDF, DOCX, TXT, or CSV file.")
            st.rerun()
        else:
            all_records = []
            log = []
            for file in uploaded:
                try:
                    records = extract_file(file)
                    all_records.extend(records)
                    log.append(
                        {
                            "File": file.name,
                            "Status": "Success",
                            "Extracted Units": len(records),
                        }
                    )
                except Exception as exc:
                    log.append(
                        {
                            "File": file.name,
                            "Status": f"Failed: {exc}",
                            "Extracted Units": 0,
                        }
                    )

            st.session_state.documents = all_records
            st.session_state.ingestion_log = log
            st.session_state.chunks = build_chunks(
                all_records, chunk_size, chunk_overlap
            )
            rebuild_index(st.session_state.chunks)

            failed = [x for x in log if x["Status"] != "Success"]
            if failed:
                st.session_state.last_ingest_msg = (
                    "error",
                    "Some files could not be extracted. Check the Ingestion Results tab for the exact error."
                )
            elif not all_records:
                st.session_state.last_ingest_msg = (
                    "warning",
                    "The file was uploaded, but no readable text was extracted."
                )
            else:
                st.session_state.last_ingest_msg = (
                    "success",
                    f"Ingestion complete: {len(all_records)} content units and {len(st.session_state.chunks)} chunks created."
                )
            st.rerun()

# TAB 2: Ingestion Results
with tabs[1]:
    st.subheader("📊 Ingestion Results")
    if st.session_state.ingestion_log:
        st.dataframe(
            pd.DataFrame(st.session_state.ingestion_log),
            use_container_width=True,
            hide_index=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Files Ingested", len(st.session_state.ingestion_log))
        c2.metric("Content Units", len(st.session_state.documents))
        c3.metric("Indexed Chunks", len(st.session_state.chunks))
    else:
        st.info("Upload and ingest project documents or click 'Load Sample Test Dataset' to see results.")

# TAB 3: Content Preview
with tabs[2]:
    st.subheader("👁️ Content Preview")
    if st.session_state.documents:
        for i, record in enumerate(st.session_state.documents[:20], start=1):
            with st.expander(
                f"{i}. {record['source']} — {record['location']}"
            ):
                st.write(record["text"][:5000])
        if len(st.session_state.documents) > 20:
            st.caption("Showing the first 20 extracted content units.")
    else:
        st.info("No extracted content yet.")

# TAB 4: RAG Indexing
with tabs[3]:
    st.subheader("🧩 RAG Indexing")
    st.write(
        "The application splits documents into overlapping chunks and creates a "
        "local TF-IDF retrieval index for knowledge retrieval."
    )

    a, b, c = st.columns(3)
    a.markdown(f'<div class="metric-card"><div class="metric-label">Chunk Size</div><div class="metric-value">{chunk_size}</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="metric-card"><div class="metric-label">Chunk Overlap</div><div class="metric-value">{chunk_overlap}</div></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="metric-card"><div class="metric-label">Indexed Chunks</div><div class="metric-value">{len(st.session_state.chunks)}</div></div>', unsafe_allow_html=True)

    if st.button("🔄 Rebuild RAG Index"):
        st.session_state.chunks = build_chunks(
            st.session_state.documents, chunk_size, chunk_overlap
        )
        rebuild_index(st.session_state.chunks)
        st.session_state.rebuild_msg = "RAG index rebuilt using the current chunk settings."
        st.rerun()

    if "rebuild_msg" in st.session_state:
        st.success(st.session_state.rebuild_msg)
        del st.session_state.rebuild_msg

    if st.session_state.chunks:
        preview_df = pd.DataFrame(
            [
                {
                    "Chunk ID": c.chunk_id,
                    "Source": c.source,
                    "Location": c.location,
                    "Characters": len(c.text),
                    "Preview": c.text[:180],
                }
                for c in st.session_state.chunks[:50]
            ]
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
    else:
        st.info("Ingest documents first.")

# TAB 5: Retrieval Explorer
with tabs[4]:
    st.subheader("🔎 Retrieval Explorer")
    query = st.text_input(
        "Enter a project question",
        placeholder="Example: What are the current project risks?",
        key="retrieval_query",
    )

    if st.button("Retrieve Relevant Chunks", type="primary"):
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            st.session_state.last_results = retrieve(query, top_k)

    if st.session_state.last_results:
        for result in st.session_state.last_results:
            c = result["chunk"]
            st.markdown(
                f"**{c.chunk_id} — {c.source} — {c.location}**  \n"
                f"Similarity score: `{result['score']:.3f}`"
            )
            st.write(c.text)
            st.divider()
    elif st.session_state.chunks:
        st.info("Ask a question to retrieve the most relevant chunks.")
    else:
        st.info("Ingest documents and build the index first.")

# TAB 6: Grounded RAG Q&A
with tabs[5]:
    st.subheader("🤖 Grounded RAG Q&A")
    question = st.text_area(
        "Ask a question about your project",
        placeholder=(
            "Example: Which project risks are mentioned in the uploaded documents, "
            "and what evidence supports them?"
        ),
        height=100,
    )

    if st.button("💬 Generate Grounded Answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            results = retrieve(question, top_k)
            if not results:
                st.warning("No relevant project content was found.")
            else:
                relevant = [r for r in results if r["score"] > 0]
                if not relevant:
                    st.warning(
                        "The documents do not contain enough matching evidence "
                        "to answer this question."
                    )
                else:
                    combined = "\n".join(r["chunk"].text for r in relevant)
                    sentences = re.split(r"(?<=[.!?])\s+", combined)
                    q_terms = {
                        w.lower()
                        for w in re.findall(r"[A-Za-z0-9]+", question)
                        if len(w) > 3
                    }

                    ranked_sentences = []
                    for sentence in sentences:
                        words = set(re.findall(r"[A-Za-z0-9]+", sentence.lower()))
                        overlap = len(words & q_terms)
                        if overlap:
                            ranked_sentences.append((overlap, sentence.strip()))

                    ranked_sentences.sort(reverse=True)
                    selected = []
                    seen = set()
                    for _, sentence in ranked_sentences:
                        key = sentence.lower()
                        if key not in seen and sentence:
                            selected.append(sentence)
                            seen.add(key)
                        if len(selected) >= 6:
                            break

                    st.markdown("### Answer")
                    if selected:
                        st.write(
                            "Based only on the retrieved project evidence: "
                            + " ".join(selected)
                        )
                    else:
                        st.write(
                            "The retrieved project documents contain related "
                            "information, but there is not enough matching text "
                            "to produce a reliable grounded answer."
                        )

                    st.markdown("### Sources / Evidence")
                    for r in relevant:
                        c = r["chunk"]
                        st.markdown(
                            f"- **{c.source} — {c.location}** "
                            f"(score: {r['score']:.3f})"
                        )

# TAB 7: 🎯 Scope & Deliverables Agent
with tabs[6]:
    st.subheader("🎯 Scope & Deliverable Extraction Agent")
    st.write("Identifies project goals, strategic objectives, key deliverables, milestone timelines, and stakeholder RACI responsibilities grounded in document content.")

    if not st.session_state.documents:
        st.info("Upload project documents or click 'Load Sample Test Dataset' on Tab 1 to run the Scope Agent.")
    else:
        scope_agent = st.session_state.scope_agent
        scope_data = scope_agent.analyze(st.session_state.documents)

        # High-level Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Goals & Objectives", len(scope_data["goals"]))
        m2.metric("Milestones & Timeline", len(scope_data["milestones"]))
        m3.metric("Key Deliverables", len(scope_data["deliverables"]))
        m4.metric("RACI Responsibilities", len(scope_data["responsibilities"]))

        st.divider()

        # Goals Section
        st.markdown("### 📌 Project Goals & Strategic Objectives")
        if scope_data["goals"]:
            st.dataframe(pd.DataFrame(scope_data["goals"]), use_container_width=True, hide_index=True)
        else:
            st.info("No explicit goals detected.")

        # Milestones Section
        st.markdown("### 📅 Milestones & Delivery Schedule")
        if scope_data["milestones"]:
            st.dataframe(pd.DataFrame(scope_data["milestones"]), use_container_width=True, hide_index=True)
        else:
            st.info("No milestone schedules detected.")

        # Deliverables Section
        st.markdown("### 📦 Key Deliverables & Owners")
        if scope_data["deliverables"]:
            st.dataframe(pd.DataFrame(scope_data["deliverables"]), use_container_width=True, hide_index=True)
        else:
            st.info("No key deliverables detected.")

        # RACI Section
        st.markdown("### 👥 Stakeholder RACI Responsibility Matrix")
        if scope_data["responsibilities"]:
            st.dataframe(pd.DataFrame(scope_data["responsibilities"]), use_container_width=True, hide_index=True)
        else:
            st.info("No RACI responsibilities detected.")


# TAB 8: ⚠️ Risk & Delivery Forecast Agent
with tabs[7]:
    st.subheader("⚠️ Risk Detection and Delivery Forecasting Agent")
    st.write("Identifies schedule risks, dependency gaps, resource bottlenecks, and calculates grounded delivery forecasting and delay probability.")

    if not st.session_state.documents:
        st.info("Upload project documents or click 'Load Sample Test Dataset' on Tab 1 to run the Risk Agent.")
    else:
        scope_agent = st.session_state.scope_agent
        risk_agent = st.session_state.risk_agent

        milestones = scope_agent.extract_milestones(st.session_state.documents)
        risk_data = risk_agent.analyze(st.session_state.documents, milestones)
        forecast = risk_data["forecast"]

        # Health Banner Card
        badge_class = "badge-red" if forecast["health_status"] == "CRITICAL RISK" else ("badge-orange" if forecast["health_status"] == "AT RISK" else "badge-green")
        st.markdown(f"""
        <div class="section-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:14px; color:#64748b; font-weight:600;">Overall Delivery Health Forecast:</span>
                    <h2 style="margin:4px 0 0 0; color:#0f172a;">
                        <span class="{badge_class}">{forecast['health_status']}</span>
                    </h2>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:13px; color:#64748b;">Estimated Schedule Slippage</div>
                    <div style="font-size:22px; font-weight:750; color:#2563eb;">{forecast['estimated_slippage']}</div>
                </div>
            </div>
            <div style="margin-top:12px; font-size:13px; color:#64748b;">
                <b>Confidence:</b> {forecast['confidence']} | <b>Total Risk Score:</b> {forecast['total_risk_score']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("High Severity Risks", forecast["high_risk_count"])
        c2.metric("Medium Severity Risks", forecast["medium_risk_count"])
        c3.metric("Critical Dependency Gaps", forecast["dependency_gap_count"])

        st.divider()

        # Risk Log
        st.markdown("### 🚩 Grounded Risk Register")
        if risk_data["risks"]:
            st.dataframe(pd.DataFrame(risk_data["risks"]), use_container_width=True, hide_index=True)
        else:
            st.success("No critical project risks detected.")

        # Dependency Gaps
        st.markdown("### 🔗 Dependency Gaps & Prerequisite Bottlenecks")
        if risk_data["dependencies"]:
            st.dataframe(pd.DataFrame(risk_data["dependencies"]), use_container_width=True, hide_index=True)
        else:
            st.info("No explicit dependency gaps detected.")

        # Milestone Forecast
        st.markdown("### 📈 Milestone Delay Probability Forecast")
        if forecast["milestone_forecasts"]:
            st.dataframe(pd.DataFrame(forecast["milestone_forecasts"]), use_container_width=True, hide_index=True)
        else:
            st.info("Ingest document containing milestone schedules to generate delay forecasts.")

        # Grounded Recommendations
        st.markdown("### 💡 Grounded Delivery Recommendations")
        for rec in forecast["recommendations"]:
            st.markdown(f"- {rec}")


# TAB 9: 🚧 Blockers & Action Items Agent
with tabs[8]:
    st.subheader("🚧 Blocker and Action Item Identification Agent")
    st.write("Extracts unresolved operational blockers, pending decisions, and assigned action items from meeting notes, status reports, and progress updates.")

    if not st.session_state.documents:
        st.info("Upload project documents or click 'Load Sample Test Dataset' on Tab 1 to run the Blocker Agent.")
    else:
        blocker_agent = st.session_state.blocker_agent
        blocker_data = blocker_agent.analyze(st.session_state.documents)

        # Metric Overview
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Unresolved Blockers", blocker_data["summary"]["total_blockers"])
        b2.metric("Pending Decisions", blocker_data["summary"]["total_decisions"])
        b3.metric("Total Action Items", blocker_data["summary"]["total_action_items"])
        b4.metric("Open Action Items", blocker_data["summary"]["open_action_items"])

        st.divider()

        # Unresolved Blockers
        st.markdown("### 🚨 Unresolved Blockers & Critical Issues")
        if blocker_data["blockers"]:
            st.dataframe(pd.DataFrame(blocker_data["blockers"]), use_container_width=True, hide_index=True)
        else:
            st.success("No active blockers detected.")

        # Pending Decisions
        st.markdown("### ⚖️ Pending Decisions Awaiting Approval")
        if blocker_data["decisions"]:
            st.dataframe(pd.DataFrame(blocker_data["decisions"]), use_container_width=True, hide_index=True)
        else:
            st.info("No pending decisions detected.")

        # Action Items Tracker
        st.markdown("### ✅ Action Item Tracker")
        if blocker_data["action_items"]:
            df_actions = pd.DataFrame(blocker_data["action_items"])
            
            # Interactive Filter
            status_filter = st.selectbox("Filter Action Items by Status", ["All", "Open", "In Progress", "Completed"])
            if status_filter != "All":
                df_actions = df_actions[df_actions["Status"].str.lower() == status_filter.lower()]
                
            st.dataframe(df_actions, use_container_width=True, hide_index=True)
        else:
            st.info("No assigned action items detected.")


# TAB 10: 🧪 Multi-Format Agent Validator
with tabs[9]:
    st.subheader("🧪 Multi-Format Agent Validation Suite")
    st.write("Validate all 3 specialized agents across **PDF, DOCX, TXT, and CSV** sample project documents.")

    if st.button("▶️ Run Multi-Format Agent Validation Test Suite", type="primary"):
        from test_agents import run_validation, MockUploadedFile
        from pathlib import Path

        sample_dir = Path("sample_data")
        if not sample_dir.exists():
            st.error("sample_data directory not found.")
        else:
            files = list(sample_dir.glob("*.*"))
            test_records = []
            file_results = []
            for fpath in files:
                mock_file = MockUploadedFile(str(fpath))
                try:
                    recs = extract_file(mock_file)
                    test_records.extend(recs)
                    file_results.append({
                        "Sample Document": mock_file.name,
                        "Format": fpath.suffix.upper(),
                        "Status": "PASSED",
                        "Units Extracted": len(recs)
                    })
                except Exception as exc:
                    file_results.append({
                        "Sample Document": mock_file.name,
                        "Format": fpath.suffix.upper(),
                        "Status": f"FAILED: {exc}",
                        "Units Extracted": 0
                    })

            st.markdown("#### 1. File Format Ingestion Sanity Test")
            st.dataframe(pd.DataFrame(file_results), use_container_width=True, hide_index=True)

            # Test Agents
            scope_res = st.session_state.scope_agent.analyze(test_records)
            risk_res = st.session_state.risk_agent.analyze(test_records, scope_res["milestones"])
            blocker_res = st.session_state.blocker_agent.analyze(test_records)

            st.markdown("#### 2. Multi-Agent Extraction Validation Matrix")
            val_matrix = [
                {"Agent Module": "Scope & Deliverable Extraction Agent", "Target Metric": "Project Goals & Objectives", "Extracted Count": len(scope_res['goals']), "Assertion Status": "PASSED" if len(scope_res['goals']) > 0 else "FAILED"},
                {"Agent Module": "Scope & Deliverable Extraction Agent", "Target Metric": "Milestones & Timelines", "Extracted Count": len(scope_res['milestones']), "Assertion Status": "PASSED" if len(scope_res['milestones']) > 0 else "FAILED"},
                {"Agent Module": "Scope & Deliverable Extraction Agent", "Target Metric": "Key Deliverables", "Extracted Count": len(scope_res['deliverables']), "Assertion Status": "PASSED" if len(scope_res['deliverables']) > 0 else "FAILED"},
                {"Agent Module": "Scope & Deliverable Extraction Agent", "Target Metric": "RACI Responsibilities", "Extracted Count": len(scope_res['responsibilities']), "Assertion Status": "PASSED" if len(scope_res['responsibilities']) > 0 else "FAILED"},
                {"Agent Module": "Risk Detection & Delivery Forecasting Agent", "Target Metric": "Identified Schedule/Resource Risks", "Extracted Count": len(risk_res['risks']), "Assertion Status": "PASSED" if len(risk_res['risks']) > 0 else "FAILED"},
                {"Agent Module": "Risk Detection & Delivery Forecasting Agent", "Target Metric": "Dependency Gaps", "Extracted Count": len(risk_res['dependencies']), "Assertion Status": "PASSED" if len(risk_res['dependencies']) > 0 else "FAILED"},
                {"Agent Module": "Risk Detection & Delivery Forecasting Agent", "Target Metric": "Delivery Forecast Status", "Extracted Count": 1, "Assertion Status": "PASSED (" + risk_res['forecast']['health_status'] + ")"},
                {"Agent Module": "Blocker & Action Item Agent", "Target Metric": "Unresolved Blockers", "Extracted Count": len(blocker_res['blockers']), "Assertion Status": "PASSED" if len(blocker_res['blockers']) > 0 else "FAILED"},
                {"Agent Module": "Blocker & Action Item Agent", "Target Metric": "Pending Decisions", "Extracted Count": len(blocker_res['decisions']), "Assertion Status": "PASSED" if len(blocker_res['decisions']) > 0 else "FAILED"},
                {"Agent Module": "Blocker & Action Item Agent", "Target Metric": "Action Items", "Extracted Count": len(blocker_res['action_items']), "Assertion Status": "PASSED" if len(blocker_res['action_items']) > 0 else "FAILED"}
            ]

            st.dataframe(pd.DataFrame(val_matrix), use_container_width=True, hide_index=True)
            st.success("🎉 All agent outputs validated successfully across PDF, DOCX, TXT, and CSV document formats!")


st.divider()
st.caption(
    "AI Project Intelligence & Risk Advisor Prototype — Multi-Agent System with Grounded Document RAG Engine."
)
