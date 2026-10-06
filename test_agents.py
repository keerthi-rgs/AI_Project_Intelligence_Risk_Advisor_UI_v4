import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "extract_file"):
    extract_file = getattr(sys.modules["__main__"], "extract_file")
else:
    from app import extract_file

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


def run_validation():
    print("=" * 60)
    print("=== STARTING MULTI-FORMAT AGENT VALIDATION TEST SUITE ===")
    print("=" * 60)

    sample_dir = Path("sample_data")
    if not sample_dir.exists():
        print("Error: sample_data directory not found.")
        sys.exit(1)

    files = list(sample_dir.glob("*.*"))
    if not files:
        print("Error: No sample files found in sample_data directory.")
        sys.exit(1)

    all_records = []
    file_results = []

    print(f"\n1. Ingesting {len(files)} Sample Documents Across PDF, DOCX, TXT, CSV:")
    for fpath in files:
        mock_file = MockUploadedFile(str(fpath))
        try:
            records = extract_file(mock_file)
            all_records.extend(records)
            file_results.append({
                "File": mock_file.name,
                "Format": fpath.suffix.upper(),
                "Status": "SUCCESS",
                "Units Extracted": len(records)
            })
            print(f"  [PASS] Loaded {mock_file.name} ({fpath.suffix.upper()}): {len(records)} content units")
        except Exception as exc:
            file_results.append({
                "File": mock_file.name,
                "Format": fpath.suffix.upper(),
                "Status": f"FAILED: {exc}",
                "Units Extracted": 0
            })
            print(f"  [FAIL] Failed {mock_file.name}: {exc}")

    print(f"\nTotal Ingested Content Units: {len(all_records)}")

    # Initialize Agents
    scope_agent = ScopeDeliverableAgent()
    risk_agent = RiskForecastingAgent()
    blocker_agent = BlockerActionItemAgent()
    doc_gen_agent = DocumentationGenerationAgent()
    health_scorer = ProjectHealthScorer()
    conversational_assistant = ConversationalProjectIntelligenceAssistant()

    # 2. Test Scope and Deliverable Extraction Agent
    print("\n2. Testing Scope & Deliverable Extraction Agent:")
    scope_res = scope_agent.analyze(all_records)
    print(f"  [PASS] Goals Extracted: {len(scope_res['goals'])}")
    print(f"  [PASS] Milestones Extracted: {len(scope_res['milestones'])}")
    print(f"  [PASS] Deliverables Extracted: {len(scope_res['deliverables'])}")
    print(f"  [PASS] Responsibilities / RACI Extracted: {len(scope_res['responsibilities'])}")

    assert len(scope_res["goals"]) > 0, "Scope agent failed to extract goals"
    assert len(scope_res["milestones"]) > 0, "Scope agent failed to extract milestones"
    assert len(scope_res["deliverables"]) > 0, "Scope agent failed to extract deliverables"
    assert len(scope_res["responsibilities"]) > 0, "Scope agent failed to extract RACI responsibilities"

    # 3. Test Risk Detection and Delivery Forecasting Agent
    print("\n3. Testing Risk Detection & Delivery Forecasting Agent:")
    risk_res = risk_agent.analyze(all_records, scope_res["milestones"])
    forecast = risk_res["forecast"]
    print(f"  [PASS] Risks Identified: {len(risk_res['risks'])}")
    print(f"  [PASS] Dependency Gaps Identified: {len(risk_res['dependencies'])}")
    print(f"  [PASS] Delivery Health Forecast: {forecast['health_status']} (Slippage: {forecast['estimated_slippage']})")

    assert len(risk_res["risks"]) > 0, "Risk agent failed to detect risks"
    assert len(risk_res["dependencies"]) > 0, "Risk agent failed to identify dependency gaps"
    assert forecast["health_status"] in ["ON TRACK", "AT RISK", "CRITICAL RISK"], "Invalid forecast health status"

    # 4. Test Blocker and Action Item Agent
    print("\n4. Testing Blocker & Action Item Agent:")
    blocker_res = blocker_agent.analyze(all_records)
    print(f"  [PASS] Unresolved Blockers Identified: {len(blocker_res['blockers'])}")
    print(f"  [PASS] Pending Decisions Identified: {len(blocker_res['decisions'])}")
    print(f"  [PASS] Action Items Extracted: {len(blocker_res['action_items'])}")

    assert len(blocker_res["blockers"]) > 0, "Blocker agent failed to extract blockers"
    assert len(blocker_res["decisions"]) > 0, "Blocker agent failed to extract pending decisions"
    assert len(blocker_res["action_items"]) > 0, "Blocker agent failed to extract action items"

    # 5. Test Documentation Generation Agent
    print("\n5. Testing Documentation Generation Agent:")
    doc_res = doc_gen_agent.analyze(all_records)
    print(f"  [PASS] User Stories Extracted: {len(doc_res['user_stories'])}")
    print(f"  [PASS] Risk Register Entries Extracted: {len(doc_res['risk_register'])}")
    print(f"  [PASS] Action Item List Formatted: {len(doc_res['action_items'])}")
    assert len(doc_res["user_stories"]) > 0, "DocGen agent failed to generate user stories"
    assert len(doc_res["risk_register"]) > 0, "DocGen agent failed to generate risk register"
    assert len(doc_res["action_items"]) > 0, "DocGen agent failed to generate action item list"

    # 6. Test Project Health Scoring Module
    print("\n6. Testing Project Health Scoring Module:")
    health_data = health_scorer.calculate_health_score(all_records, scope_res, risk_res, blocker_res)
    print(f"  [PASS] Overall Health Score: {health_data['overall_score']}/100 ({health_data['health_status']})")
    assert 0 <= health_data["overall_score"] <= 100, "Invalid overall health score"

    print("\n" + "=" * 60)
    print("=== ALL AGENT VALIDATION TESTS PASSED SUCCESSFULLY! ===")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
