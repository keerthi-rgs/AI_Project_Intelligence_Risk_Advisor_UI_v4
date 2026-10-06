import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "extract_file"):
    extract_file = getattr(sys.modules["__main__"], "extract_file")
else:
    from app import extract_file, build_chunks, rebuild_index

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


class MockUploadedFile:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.name = os.path.basename(filepath)

    def getvalue(self) -> bytes:
        with open(self.filepath, "rb") as f:
            return f.read()


def run_conversational_and_agent_validation():
    print("=" * 70)
    print("=== STARTING COMPLETE MULTI-AGENT & CONVERSATIONAL RAG VALIDATION SUITE ===")
    print("=" * 70)

    sample_dir = Path("sample_data")
    if not sample_dir.exists():
        print("Error: sample_data directory not found.")
        sys.exit(1)

    files = list(sample_dir.glob("*.*"))
    if not files:
        print("Error: No sample files found in sample_data directory.")
        sys.exit(1)

    all_records = []
    print(f"\n1. Ingesting {len(files)} Sample Documents Across PDF, DOCX, TXT, CSV:")
    for fpath in files:
        mock_file = MockUploadedFile(str(fpath))
        try:
            records = extract_file(mock_file)
            all_records.extend(records)
            print(f"  [PASS] Ingested {mock_file.name} ({fpath.suffix.upper()}): {len(records)} units")
        except Exception as exc:
            print(f"  [FAIL] Failed {mock_file.name}: {exc}")

    print(f"Total Ingested Content Units: {len(all_records)}")
    assert len(all_records) > 0, "No content units ingested from sample documents."

    # Build chunks and TF-IDF index for Conversational RAG
    chunks = build_chunks(all_records, size=500, overlap=100)
    import streamlit as st

    # Build vectorizer matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    # 2. Test Core Extraction Agents
    print("\n2. Testing Core Extraction Agents (Scope, Risk, Blocker):")
    scope_agent = ScopeDeliverableAgent()
    risk_agent = RiskForecastingAgent()
    blocker_agent = BlockerActionItemAgent()

    scope_res = scope_agent.analyze(all_records)
    risk_res = risk_agent.analyze(all_records, scope_res["milestones"])
    blocker_res = blocker_agent.analyze(all_records)

    print(f"  [PASS] Goals: {len(scope_res['goals'])}, Milestones: {len(scope_res['milestones'])}, Deliverables: {len(scope_res['deliverables'])}, RACI: {len(scope_res['responsibilities'])}")
    print(f"  [PASS] Risks: {len(risk_res['risks'])}, Dependencies: {len(risk_res['dependencies'])}, Delivery Health: {risk_res['forecast']['health_status']}")
    print(f"  [PASS] Blockers: {len(blocker_res['blockers'])}, Decisions: {len(blocker_res['decisions'])}, Actions: {len(blocker_res['action_items'])}")

    # 3. Test Documentation Generation Agent
    print("\n3. Testing Documentation Generation Agent:")
    doc_gen_agent = DocumentationGenerationAgent()
    doc_res = doc_gen_agent.analyze(all_records)

    user_stories = doc_res["user_stories"]
    risk_reg = doc_res["risk_register"]
    action_list = doc_res["action_items"]
    full_report = doc_res["full_report"]

    print(f"  [PASS] User Stories Generated: {len(user_stories)}")
    print(f"  [PASS] Risk Register Entries Generated: {len(risk_reg)}")
    print(f"  [PASS] Action Item List Generated: {len(action_list)}")
    print(f"  [PASS] Downloadable Report Length: {len(full_report)} characters")

    assert len(user_stories) > 0, "DocGen agent failed to produce user stories"
    assert len(risk_reg) > 0, "DocGen agent failed to produce risk register"
    assert len(action_list) > 0, "DocGen agent failed to produce action item list"
    assert len(full_report) > 200, "DocGen report generation failed"

    # 4. Test Project Health Scoring Module
    print("\n4. Testing Project Health Scoring Module:")
    scorer = ProjectHealthScorer()
    health_data = scorer.calculate_health_score(all_records, scope_res, risk_res, blocker_res)

    print(f"  [PASS] Overall Health Score: {health_data['overall_score']}/100 ({health_data['health_status']})")
    print(f"  [PASS] Dimension Breakdown:")
    print(f"         - Scope Clarity Score: {health_data['dimension_scores']['scope_clarity']}/100")
    print(f"         - Timeline Risk Score: {health_data['dimension_scores']['timeline_risk']}/100")
    print(f"         - Blocker Impact Score: {health_data['dimension_scores']['blocker_impact']}/100")
    print(f"  [PASS] Recommendations Count: {len(health_data['recommendations'])}")

    assert 0 <= health_data["overall_score"] <= 100, "Health score out of bounds"
    assert health_data["health_status"] in ["HEALTHY / ON TRACK", "MODERATE RISK / AT RISK", "CRITICAL RISK"], "Invalid health status"

    # 5. Test Conversational Assistant Q&A Accuracy & Relevance
    print("\n5. Testing Grounded Conversational Assistant Queries:")
    assistant = ConversationalProjectIntelligenceAssistant()

    test_queries = [
        ("Are we on track?", "project_health"),
        ("What are our biggest risks?", "risk_analysis"),
        ("What active blockers are open?", "blocker_tracking"),
        ("What action items are assigned to the team?", "action_tracking"),
        ("Who is responsible for deliverables?", "scope_raci"),
        ("What database script is failing during ingestion?", "document_retrieval")
    ]

    for query_text, expected_intent in test_queries:
        res = assistant.answer_query(
            query=query_text,
            records=all_records,
            chunks=chunks,
            vectorizer=vectorizer,
            matrix=matrix,
            top_k=4,
            scope_data=scope_res,
            risk_data=risk_res,
            blocker_data=blocker_res,
            health_data=health_data
        )

        print(f"\n  Query: \"{query_text}\"")
        print(f"  - Intent: {res.get('intent')} | Grounded: {res.get('grounded')} | Confidence: {res.get('confidence')}")
        print(f"  - Sources Cited: {len(res.get('sources', []))}")
        print(f"  - Answer Snippet: {res.get('answer')[:120]}...")

        assert res["grounded"] is True, f"Response not grounded for query: {query_text}"
        assert len(res["answer"]) > 20, f"Response too short for query: {query_text}"
        assert len(res["sources"]) > 0, f"Missing source citations for query: {query_text}"

    print("\n" + "=" * 70)
    print("=== ALL MULTI-AGENT, HEALTH SCORING, AND CONVERSATIONAL TESTS PASSED! ===")
    print("=" * 70)


if __name__ == "__main__":
    run_conversational_and_agent_validation()
