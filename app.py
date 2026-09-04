
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


st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="🧰",
    layout="wide",
)


# --- Professional Dark UI styling ---
st.markdown("""
<style>
    /* Dark background backdrop */
    .stApp, [data-testid="stHeader"] {
        background-color: #090a0f !important;
        color: #e6edf3;
    }

    /* Main content container */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Header card */
    .project-header {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 24px 16px 24px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .project-title {
        font-size: 28px;
        font-weight: 750;
        color: #f0f6fc;
        margin: 0;
    }
    .project-subtitle {
        font-size: 14px;
        color: #8b949e;
        margin-top: 6px;
    }

    /* Sidebar dark styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.2rem 1rem;
    }
    .side-brand {
        font-size: 17px;
        font-weight: 750;
        color: #f0f6fc;
        margin-bottom: 18px;
    }
    .side-section {
        font-size: 12px;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: .05em;
        margin: 18px 0 8px;
    }

    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px 16px;
        min-height: 90px;
    }
    .metric-label {
        color: #8b949e;
        font-size: 12px;
        font-weight: 650;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 25px;
        font-weight: 750;
        margin-top: 6px;
    }

    .section-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        margin: 8px 0 14px;
    }

    .upload-title {
        font-size: 18px;
        font-weight: 720;
        color: #f0f6fc;
    }
    .upload-help {
        color: #8b949e;
        font-size: 13px;
        margin: 4px 0 12px;
    }

    /* Tabs styling for dark mode */
    button[data-baseweb="tab"] {
        font-weight: 650;
        color: #8b949e !important;
        padding: 10px 12px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #58a6ff !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #58a6ff !important;
    }

    /* Primary action buttons */
    .stButton > button[kind="primary"] {
        background: #238636;
        border: 1px solid #2ea043;
        color: #ffffff;
        font-weight: 700;
        border-radius: 7px;
        min-height: 42px;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2ea043;
        border-color: #3fb950;
    }

    /* File uploader dark style */
    [data-testid="stFileUploader"] {
        background: #161b22;
        border: 1px dashed #30363d;
        border-radius: 9px;
        padding: 6px;
    }

    /* Expander dark style */
    div[data-testid="stExpander"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        background: #161b22;
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
        Milestone 1: Robust Multi-Format Document Ingestion, Normalization,
        Chunking & Grounded RAG Knowledge Base
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


# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.markdown('<div class="side-brand">🧰 AI Project Intelligence & Risk Advisor</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="side-section">Configuration</div>', unsafe_allow_html=True)
chunk_size = st.sidebar.slider("Chunk Size (characters)", 200, 1500, 500, 50)
chunk_overlap = st.sidebar.slider(
    "Chunk Overlap (characters)", 0, min(500, chunk_size - 1), 100, 10
)
top_k = st.sidebar.slider("Retrieval Top-K Chunks", 1, 10, 4)

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

    # Some DOCX files store useful text in structures not exposed as ordinary
    # paragraphs/tables. Read the XML as a safe fallback.
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
    # Convert rows to compact natural-language records for retrieval
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
# Chunking
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

        # Try to end at a sentence/word boundary when possible.
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


# -----------------------------
# Indexing / retrieval
# -----------------------------
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
# Risk advisor
# -----------------------------
RISK_RULES = {
    "schedule": [
        "delay", "overdue", "deadline", "late", "schedule", "slippage", "blocked"
    ],
    "quality": [
        "defect", "bug", "failure", "error", "rework", "quality", "issue"
    ],
    "resource": [
        "resource", "staff", "team", "capacity", "dependency", "shortage"
    ],
    "security": [
        "security", "vulnerability", "breach", "authentication", "privacy"
    ],
    "cost": [
        "budget", "cost", "expense", "overrun", "financial"
    ],
}


def detect_risks(text):
    low = text.lower()
    findings = []
    for category, terms in RISK_RULES.items():
        hits = [t for t in terms if t in low]
        if hits:
            findings.append(
                {
                    "Risk Area": category.title(),
                    "Evidence Keywords": ", ".join(hits[:5]),
                    "Severity": "High" if len(hits) >= 3 else "Medium",
                }
            )
    return findings


# -----------------------------
# Tabs matching the reference UI
# -----------------------------
tabs = st.tabs(
    [
        "1. Document Upload",
        "2. Ingestion Results",
        "3. Content Preview",
        "4. RAG Indexing",
        "5. Retrieval Explorer",
        "6. Grounded RAG Q&A",
    ]
)

# TAB 1
with tabs[0]:
    st.markdown('<div class="upload-title">📤 Upload Project Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-help">Upload project documents for extraction, retrieval and project-risk analysis.</div>', unsafe_allow_html=True)
    st.write(
        "Supported formats: PDF (page-by-page), DOCX (paragraphs & tables), "
        "TXT (notes & specs), CSV (structured tabular data)."
    )

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
            type="primary",
            use_container_width=True,
        )
    with col2:
        if st.session_state.documents:
            st.info(f"{len(st.session_state.documents)} document(s) currently loaded.")

    # Display persistent ingestion message across reruns
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
                    "The file was uploaded, but no readable text was extracted. If this is a scanned PDF/image-only document, OCR is required."
                )
            else:
                st.session_state.last_ingest_msg = (
                    "success",
                    f"Ingestion complete: {len(all_records)} content units and {len(st.session_state.chunks)} chunks created."
                )
            st.rerun()

# TAB 2
with tabs[1]:
    st.subheader("📊 Ingestion Results")
    if st.session_state.ingestion_log:
        st.dataframe(
            pd.DataFrame(st.session_state.ingestion_log),
            use_container_width=True,
            hide_index=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Files", len(st.session_state.ingestion_log))
        c2.metric("Content Units", len(st.session_state.documents))
        c3.metric("Chunks", len(st.session_state.chunks))
    else:
        st.info("Upload and ingest project documents to see results.")

# TAB 3
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

# TAB 4
with tabs[3]:
    st.subheader("🧩 RAG Indexing")
    st.write(
        "The application splits documents into overlapping chunks and creates a "
        "local TF-IDF retrieval index. This provides a lightweight vector-style "
        "knowledge base without requiring a paid API."
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

# TAB 5
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

# TAB 6
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
                # Lightweight grounded synthesis. It intentionally avoids inventing
                # information not present in retrieved chunks.
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

                    risks = detect_risks(combined)
                    if risks:
                        st.markdown("### Risk Signals")
                        st.dataframe(
                            pd.DataFrame(risks),
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.markdown("### Sources / Evidence")
                    for r in relevant:
                        c = r["chunk"]
                        st.markdown(
                            f"- **{c.source} — {c.location}** "
                            f"(score: {r['score']:.3f})"
                        )

st.divider()
st.caption(
    "Prototype for academic/project demonstration. For production use, replace "
    "the local TF-IDF retriever with a persistent embedding/vector database and "
    "connect a controlled LLM for answer generation."
)
