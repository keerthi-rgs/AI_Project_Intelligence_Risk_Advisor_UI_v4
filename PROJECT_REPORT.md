# Project Report — AI Project Intelligence and Risk Advisor (Milestone 4)

## 1. Purpose

This project turns uploaded project documents into an operator dashboard: health scoring, risk detection, action tracking, scope extraction, and a grounded conversational assistant. Milestone 4 completes the insights dashboard, incremental knowledge-base updates, testing, and demonstration materials.

## 2. What was delivered

| Requirement | Where it lives |
| --- | --- |
| Insights & risk summary dashboard (health, risks, actions, scope) | `app.py` → **Dashboard** |
| Incremental document upload with automatic KB + RAG refresh | `app.py` → `apply_ingestion()`, **Documents** page |
| End-to-end tests across formats, agents, conversation | `test_agents.py`, `test_conversational_rag.py`, `test_milestone4.py`, `test_extraction_accuracy.py`, `test_generic_documents.py`, Settings validation |
| Technical documentation and demo script | `README.md`, this report, **Architecture & Milestones** page |

## 3. Pipeline

1. **Ingest** PDF, DOCX, TXT, CSV into content units with source + location.
2. **Incremental merge** replaces units from the same filename and appends new files, then rebuilds chunks and the TF-IDF index.
3. **Agents**
   - Scope: goals, milestones, deliverables, RACI
   - Risk: category, severity (including Critical), probability/impact, mitigation, dependency gaps, delivery forecast
   - Blockers: unresolved issues, pending decisions, action items
   - Documentation: user stories, risk register, action list, markdown report
   - Health: 0–100 overall score from scope clarity, timeline risk, and blocker impact
4. **Assistant** routes known intents, otherwise retrieves TF-IDF chunks and cites sources.

## 4. Evaluation mapping

1. **Accuracy/completeness of scope, risk, and blocker extraction** — agents scan sentence/line units with keyword + pattern rules; sample charter, status report, meeting notes, and backlog are designed to fire those rules. Tests assert non-empty goals, milestones, risks, blockers, and actions.
2. **Quality of generated user stories, risk registers, and action lists** — stories use “As a … I want …”; register rows include severity, likelihood, mitigation, and source; action items include assignee, priority, due date, status. Download from **Reports & Doc Generation**.
3. **Relevance and groundedness of conversational answers** — intent answers cite agent-extracted sources; open questions cite retrieved chunks. Tests require `grounded=True` and at least one source.
4. **Clarity of health scoring and insights dashboard** — KPI cards, dimension bars, scope tables, ranked risks, action tracker, and Plotly risk charts.
5. **Completeness of implementation, testing, documentation, and demo** — UI pages, three test scripts, README, and this report.

## 5. Final demonstration script (~10 minutes)

1. Load sample dataset (PDF status report, DOCX charter, TXT meeting notes, CSV backlog).
2. Show Dashboard: health score, scope summary, risks, action items, charts.
3. Add `progress_update.txt` incrementally; point out new GPU blocker / extra actions and that chunk count increased without wiping prior files.
4. Ask the assistant: “Are we on track?”, “What delayed model evaluation?” — open Grounded Evidence.
5. Generate and download the governance markdown report.
6. Run Settings → End-to-End Validation (or `python test_milestone4.py`).

## 6. Known limits and next upgrades

- Retrieval is TF-IDF, not embeddings/FAISS/Chroma.
- Extraction is rule-based (no LLM), so phrasing outside keyword patterns may be missed.
- Session state is in-memory; refresh without rerun loses the index unless files are re-ingested.

A later milestone can swap the retriever for embeddings and connect Gemini/OpenAI/Ollama for generation while keeping the same dashboard and incremental ingest contract.

## 7. Verified results (sample dataset: PDF, DOCX, TXT, CSV)

| Extraction | Found | Expected | Precision | Recall |
|---|---|---|---|---|
| Goals | 5 | 5 | 100% | 100% |
| Milestones | 4 | 4 | 100% | 100% |
| Deliverables | 3 | 3 | 100% | 100% |
| RACI roles | 4 | 4 | 100% | 100% |
| Risks | 14 | 14 | 100% | 100% |
| Blockers | 5 | 5 | 100% | 100% |
| Pending decisions | 3 | 3 | 100% | 100% |
| Action items | 8 | 8 | 100% | 100% |

Scope extraction is not tied to the sample layout. It reads explicit `Key: value` lines, section headings (Objectives, Timeline, Deliverables, Roles & Responsibilities), Word tables, CSV columns (Type/Task/Owner/Deadline/Status), and phrasing such as "aims to", "X is responsible for Y" and "Milestone: ... by <date>". `test_generic_documents.py` covers documents in these other layouts.

The ground truth was labelled by hand from the sample documents, which the rule-based agents were tuned against. Expect lower scores on unseen documents.
All four test scripts pass, and every app page renders without errors.
