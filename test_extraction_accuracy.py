"""Accuracy test: compare agent output with a hand-labelled ground truth for the sample dataset.

Run `python create_sample_docs.py` first. Reports precision / recall per extraction type.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

from app import extract_file, FileOnDisk
from agents import ScopeDeliverableAgent, RiskForecastingAgent, BlockerActionItemAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Each expected item is identified by a short phrase that must appear in exactly one extracted item.
GROUND_TRUTH = {
    "goals": ["ai-driven project intelligence", "reduce manual project reporting", "detect delivery schedule risks",
              "automate extraction of unresolved blockers", "close remaining milestone 4"],
    "milestones": ["milestone phase 1", "milestone phase 2", "milestone phase 3", "milestone sprint 4"],
    "deliverables": ["scope and deliverable extraction agent", "risk detection and delivery forecasting agent",
                     "blocker and action item agent"],
    "responsibilities": ["project manager", "lead architect", "security officer", "product sponsor"],
    "risks": ["database migration script", "oauth2 token refresh", "gpu instance shortage", "qa team is understaffed",
              "backend team is understaffed", "slippage on milestone 4 documentation", "2-week delay",
              "phase 2 is experiencing schedule slippage", "latency spike", "budget overrun",
              "third-party api integration delays", "provisioning sign-off", "documentation remains incomplete",
              "rate-limit upgrade"],
    "blockers": ["database migration script", "oauth2 token refresh", "gpu instance shortage",
                 "rate limit blocking ingestion", "documentation remains incomplete"],
    "decisions": ["chromadb and faiss", "budget overrun", "cloud deployment region"],
    "action_items": ["docx table parsing", "risk scoring algorithm", "security audit", "executive milestone demo deck",
                     "implement scope and deliverable", "build risk detection", "weekly risk register snapshot",
                     "re-run pdf, docx, txt"],
}
TEXT_FIELD = {
    "goals": "Goal / Objective", "milestones": "Milestone / Phase", "deliverables": "Deliverable",
    "responsibilities": "Role", "risks": "Risk Description", "blockers": "Blocker", "decisions": "Decision",
    "action_items": "Action Item",
}
MIN_PRECISION, MIN_RECALL = 0.9, 0.9


def score(kind, items):
    texts = [str(i[TEXT_FIELD[kind]]).lower() for i in items]
    expected = GROUND_TRUTH[kind]
    found = [p for p in expected if any(p in t for t in texts)]
    correct = [t for t in texts if any(p in t for p in expected)]
    precision = len(correct) / len(texts) if texts else 0.0
    recall = len(found) / len(expected)
    return precision, recall, sorted(set(expected) - set(found)), [t for t in texts if t not in correct]


def run_accuracy_test():
    records = []
    for f in sorted(Path("sample_data").glob("*.*")):
        records.extend(extract_file(FileOnDisk(str(f))))
    assert records, "sample_data is empty — run python create_sample_docs.py"

    scope = ScopeDeliverableAgent().analyze(records)
    risk = RiskForecastingAgent().analyze(records, scope["milestones"])
    blk = BlockerActionItemAgent().analyze(records)
    outputs = {**scope, "risks": risk["risks"], "blockers": blk["blockers"],
               "decisions": blk["decisions"], "action_items": blk["action_items"]}

    print("=" * 70)
    print(f"{'Extraction':<18}{'Found':>7}{'Expected':>10}{'Precision':>11}{'Recall':>9}")
    print("-" * 70)
    failures = []
    for kind in GROUND_TRUTH:
        p, r, missed, extra = score(kind, outputs[kind])
        print(f"{kind:<18}{len(outputs[kind]):>7}{len(GROUND_TRUTH[kind]):>10}{p:>11.0%}{r:>9.0%}")
        for m in missed:
            print(f"    missed: {m}")
        for e in extra:
            print(f"    unexpected: {e[:70]}")
        if p < MIN_PRECISION or r < MIN_RECALL:
            failures.append(kind)
    print("=" * 70)
    assert not failures, f"Accuracy below threshold for: {failures}"
    print("=== EXTRACTION ACCURACY TEST PASSED ===")


if __name__ == "__main__":
    run_accuracy_test()
