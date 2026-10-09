"""Milestone 4 end-to-end tests: document types, incremental KB, agents, conversational RAG."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import extract_file, build_chunks, merge_documents_incremental, FileOnDisk
from agents import (
    ScopeDeliverableAgent,
    RiskForecastingAgent,
    BlockerActionItemAgent,
    DocumentationGenerationAgent,
    ProjectHealthScorer,
    ConversationalProjectIntelligenceAssistant,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ingest_path(path: Path):
    return extract_file(FileOnDisk(str(path)))


def run_milestone4_validation():
    print("=" * 70)
    print("=== MILESTONE 4 END-TO-END VALIDATION ===")
    print("=" * 70)

    sample_dir = Path("sample_data")
    assert sample_dir.exists(), "sample_data directory not found."

    by_ext = {}
    for fpath in sample_dir.glob("*.*"):
        recs = ingest_path(fpath)
        by_ext[fpath.suffix.lower()] = by_ext.get(fpath.suffix.lower(), 0) + len(recs)
        print(f"  [PASS] {fpath.name}: {len(recs)} units")
        assert len(recs) > 0, f"No text extracted from {fpath.name}"

    for ext in [".pdf", ".txt", ".csv"]:
        assert ext in by_ext, f"Missing sample document type: {ext}"
    if ".docx" not in by_ext:
        print("  [WARN] project_charter.docx missing — generating samples.")
        from create_sample_docs import create_samples
        create_samples()
        recs = ingest_path(sample_dir / "project_charter.docx")
        assert len(recs) > 0, "DOCX generation/extraction failed"
        by_ext[".docx"] = len(recs)

    baseline_files = [
        sample_dir / "status_report.pdf",
        sample_dir / "meeting_notes.txt",
        sample_dir / "sprint_backlog.csv",
        sample_dir / "project_charter.docx",
    ]
    baseline = []
    for fpath in baseline_files:
        if fpath.exists():
            baseline.extend(ingest_path(fpath))

    print(f"\n1. Baseline knowledge base: {len(baseline)} units")
    assert len(baseline) > 0

    update_path = sample_dir / "progress_update.txt"
    assert update_path.exists(), "progress_update.txt is required for incremental upload tests"
    update_recs = ingest_path(update_path)
    merged = merge_documents_incremental(baseline, update_recs)
    print(f"2. Incremental merge added {len(update_recs)} units → {len(merged)} total")
    assert len(merged) == len(baseline) + len(update_recs), "Incremental merge should append a new source"
    assert any("progress_update.txt" in r.get("source", "") for r in merged)

    replaced = merge_documents_incremental(merged, update_recs)
    assert len(replaced) == len(merged), "Re-ingesting the same filename must replace, not duplicate"

    chunks = build_chunks(merged, size=500, overlap=100)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([c.text for c in chunks])
    print(f"3. Auto RAG index: {len(chunks)} chunks")
    assert len(chunks) > 0

    qvec = vectorizer.transform(["GPU instance shortage delayed model evaluation"])
    scores = cosine_similarity(qvec, matrix)[0]
    assert float(scores.max()) > 0, "Incremental content was not searchable in the rebuilt index"

    scope = ScopeDeliverableAgent().analyze(merged)
    risk = RiskForecastingAgent().analyze(merged, scope["milestones"])
    blocker = BlockerActionItemAgent().analyze(merged)
    docs = DocumentationGenerationAgent().analyze(merged)
    health = ProjectHealthScorer().calculate_health_score(merged, scope, risk, blocker)

    print("\n4. Agent outputs after incremental update:")
    print(f"   Scope: {len(scope['goals'])} goals, {len(scope['milestones'])} milestones, "
          f"{len(scope['deliverables'])} deliverables, {len(scope['responsibilities'])} RACI")
    print(f"   Risks: {len(risk['risks'])} | Blockers: {len(blocker['blockers'])} | Actions: {len(blocker['action_items'])}")
    print(f"   Doc gen: {len(docs['user_stories'])} stories, {len(docs['risk_register'])} register rows")
    print(f"   Health: {health['overall_score']}/100 ({health['health_status']})")

    assert scope["goals"], "Scope extraction incomplete (goals)"
    assert scope["milestones"], "Scope extraction incomplete (milestones)"
    assert risk["risks"], "Risk detection incomplete"
    assert blocker["blockers"], "Blocker identification incomplete"
    assert docs["user_stories"], "User story generation incomplete"
    assert docs["risk_register"], "Risk register generation incomplete"
    assert docs["action_items"], "Action item list incomplete"
    assert any(s.get("User Story", "").lower().startswith("as a") for s in docs["user_stories"])
    assert 0 <= health["overall_score"] <= 100

    assistant = ConversationalProjectIntelligenceAssistant()
    queries = [
        "Are we on track?",
        "What are our biggest risks?",
        "What active blockers are open?",
        "What action items are assigned?",
        "Who is responsible for deliverables?",
        "What delayed model evaluation?",
    ]
    print("\n5. Conversational groundedness:")
    for query in queries:
        res = assistant.answer_query(
            query=query,
            records=merged,
            chunks=chunks,
            vectorizer=vectorizer,
            matrix=matrix,
            top_k=4,
            scope_data=scope,
            risk_data=risk,
            blocker_data=blocker,
            health_data=health,
        )
        print(f"   Q: {query} | grounded={res['grounded']} | sources={len(res.get('sources', []))}")
        assert res["grounded"] is True, f"Ungrounded answer for: {query}"
        assert len(res["answer"]) > 20, f"Empty answer for: {query}"
        assert res.get("sources"), f"Missing citations for: {query}"

    print("\n" + "=" * 70)
    print("=== MILESTONE 4 VALIDATION PASSED ===")
    print("=" * 70)


if __name__ == "__main__":
    run_milestone4_validation()
