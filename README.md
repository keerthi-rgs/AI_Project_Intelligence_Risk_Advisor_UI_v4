# AI Project Intelligence and Risk Advisor

Streamlit prototype for grounded project intelligence: ingest PDF/DOCX/TXT/CSV files, extract scope/risks/blockers, score health, and answer questions with source citations.


## Run

```bash
pip install -r requirements.txt
python create_sample_docs.py
streamlit run app.py
```

Open the local Streamlit URL. Recommended demo path:

1. **Documents** → Load Sample Test Dataset (PDF, DOCX, TXT, CSV).
2. **Dashboard** — review health score, scope summary, risks, and action items.
3. **Documents** → Add Sample Progress Update (incremental) and confirm the dashboard/index refresh.
4. **AI Assistant** — ask “Are we on track?” and inspect grounded evidence.
5. **Reports & Doc Generation** — download the markdown governance report.
6. **Settings** → Run End-to-End Validation.

## Tests

```bash
python test_agents.py
python test_conversational_rag.py
python test_milestone4.py
python test_extraction_accuracy.py   # precision/recall vs hand-labelled sample ground truth
python test_generic_documents.py     # documents in other layouts (Word headings, tables, course plans, trackers)
```

## Supported files

- PDF, DOCX, TXT, CSV

## Demo settings

- Chunk size: 500
- Chunk overlap: 100
- Retrieval top-k: 4

Local retrieval uses TF-IDF + cosine similarity (no API key). Answers show retrieved evidence and source locations.

## Architecture (high level)

```
Upload (incremental or overwrite)
        ↓
Extract text units by format
        ↓
Chunk + rebuild TF-IDF index automatically
        ↓
Scope / Risk / Blocker / Doc-gen / Health agents
        ↓
Dashboard + Conversational RAG (grounded citations)
```
