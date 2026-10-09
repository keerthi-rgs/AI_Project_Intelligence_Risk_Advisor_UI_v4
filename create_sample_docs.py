import os
import io
import pandas as pd
from docx import Document
from pypdf import PdfWriter, PdfReader
from pypdf.annotations import FreeText

def create_samples():
    os.makedirs("sample_data", exist_ok=True)
    print("Creating sample files in sample_data directory...")

    # 1. Create DOCX sample: project_charter.docx
    doc = Document()
    doc.add_heading("Project Charter: AI Project Intelligence Platform", 0)
    
    doc.add_heading("1. Project Goals & Strategic Objectives", level=1)
    doc.add_paragraph("Goal: Build an AI-driven project intelligence and risk advisory platform to automate document analysis.")
    doc.add_paragraph("Objective 1: Reduce manual project reporting effort by 60% across engineering teams.")
    doc.add_paragraph("Objective 2: Detect delivery schedule risks and dependency gaps early grounded in document content.")
    doc.add_paragraph("Objective 3: Automate extraction of unresolved blockers, pending decisions, and action items.")

    doc.add_heading("2. Milestones & Delivery Timelines", level=1)
    doc.add_paragraph("Milestone Phase 1 | Target Date: 2026-10-15 | Status: Completed | Scope: Multi-Format Document Ingestion & RAG Indexing")
    doc.add_paragraph("Milestone Phase 2 | Target Date: 2026-11-01 | Status: In Progress | Scope: Multi-Agent Extraction & Risk Forecasting Engine")
    doc.add_paragraph("Milestone Phase 3 | Target Date: 2026-12-05 | Status: Planned | Scope: Executive Dashboard & Production Release")

    doc.add_heading("3. Key Deliverables & Owners", level=1)
    doc.add_paragraph("Deliverable: Scope and Deliverable Extraction Agent | Owner: Alex Rivera | Target Completion: 2026-10-20")
    doc.add_paragraph("Deliverable: Risk Detection and Delivery Forecasting Agent | Owner: Sarah Jenkins | Target Completion: 2026-11-01")
    doc.add_paragraph("Deliverable: Blocker and Action Item Agent | Owner: Michael Chen | Target Completion: 2026-11-15")

    doc.add_heading("4. Stakeholder RACI Matrix", level=1)
    doc.add_paragraph("Project Manager: Accountable for overall project schedule, milestones, and governance.")
    doc.add_paragraph("Lead Architect: Responsible for system architecture, RAG vector indexing, and pipeline stability.")
    doc.add_paragraph("Security Officer: Consulted on enterprise compliance, access control, and data encryption.")
    doc.add_paragraph("Product Sponsor: Informed on weekly progress, risk status, and milestone completion.")

    docx_path = os.path.join("sample_data", "project_charter.docx")
    doc.save(docx_path)
    print(f"Created DOCX sample: {docx_path}")

    # 2. Create TXT sample: meeting_notes.txt
    txt_content = """=== Weekly Executive & Engineering Sync Meeting Notes ===
Date: 2026-09-10
Location: Conference Room B & Virtual

1. Unresolved Blockers & Operational Issues:
- Blocker: Database migration script fails on legacy CSV schema format during high-volume ingestion. Status: Unresolved. Severity: Critical.
- Issue: OAuth2 token refresh failing silently in staging environment under heavy load.

2. Pending Decisions Required:
- Pending Decision: Choose between ChromaDB and FAISS for vector store persistence layer. Decision Owner: Architecture Committee.
- Pending Decision: Approval of $15,000 budget overrun for dedicated GPU server instances. Decision Owner: Executive Sponsor.

3. Assigned Action Items:
- Action Item: Fix DOCX table parsing edge cases in document extractor. Assigned to: Sarah Jenkins. Priority: High. Due Date: 2026-09-20. Status: Open.
- Action Item: Update risk scoring algorithm with dependency gap weights. Assigned to: Alex Rivera. Priority: High. Due Date: 2026-09-18. Status: In Progress.
- Action Item: Conduct security audit on file upload endpoint. Assigned to: Michael Chen. Priority: Medium. Due Date: 2026-09-25. Status: Open.
- Action Item: Prepare Q4 executive milestone demo deck. Assigned to: Emily Davis. Priority: Low. Due Date: 2026-10-01. Status: Completed.
"""
    txt_path = os.path.join("sample_data", "meeting_notes.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"Created TXT sample: {txt_path}")

    # 3. Create CSV sample: sprint_backlog.csv
    csv_data = [
        {"Item ID": "ACT-101", "Category": "Action Item", "Description": "Implement Scope and Deliverable Extraction Agent", "Assignee": "Alex Rivera", "Priority": "High", "Status": "Completed", "Due Date": "2026-10-01", "Notes": "Delivers goals and milestone extraction."},
        {"Item ID": "ACT-102", "Category": "Action Item", "Description": "Build Risk Detection and Delivery Forecasting Agent", "Assignee": "Sarah Jenkins", "Priority": "High", "Status": "In Progress", "Due Date": "2026-10-15", "Notes": "Evaluates risk severity and delay probability."},
        {"Item ID": "BLK-201", "Category": "Blocker", "Description": "Vendor authentication API rate limit blocking ingestion tests", "Assignee": "Michael Chen", "Priority": "Critical", "Status": "Unresolved", "Due Date": "ASAP", "Notes": "Requires vendor key upgrade."},
        {"Item ID": "DEC-301", "Category": "Pending Decision", "Description": "Finalize cloud deployment region for compliance", "Assignee": "Steering Committee", "Priority": "High", "Status": "Pending Decision", "Due Date": "2026-09-22", "Notes": "Awaiting security review."},
        {"Item ID": "MS-401", "Category": "Milestone", "Description": "Milestone Sprint 4 Delivery & Multi-Format Validation", "Assignee": "Project Team", "Priority": "High", "Status": "In Progress", "Due Date": "2026-10-30", "Notes": "Validates PDF, DOCX, TXT, CSV parsing."}
    ]
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join("sample_data", "sprint_backlog.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created CSV sample: {csv_path}")

    # 4. Create PDF sample: status_report.pdf
    # We can create a simple PDF text document using pypdf Page / Stream or python library.
    # Let's create PDF content via reportlab or simple text canvas via pypdf / fallback.
    pdf_path = os.path.join("sample_data", "status_report.pdf")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, 750, "Q4 Project Delivery & Risk Report")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, 720, "1. Executive Summary & Delivery Health")
        c.setFont("Helvetica", 10)
        c.drawString(40, 700, "Project delivery status is currently AT RISK due to third-party API integration delays.")
        c.drawString(40, 685, "Target Milestone Phase 2 is experiencing schedule slippage of 2 weeks.")
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, 650, "2. Schedule Risks & Bottlenecks")
        c.setFont("Helvetica", 10)
        c.drawString(40, 630, "• Schedule Slippage Risk: Third-party API rate limits caused 2-week delay in dataset processing.")
        c.drawString(40, 615, "• Resource Bottleneck: Backend team is understaffed by 2 developers during Sprint 4.")
        c.drawString(40, 600, "• Technical Complexity: Vector index search latency spike on documents over 100 pages.")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, 565, "3. Dependency Gaps & External Bottlenecks")
        c.setFont("Helvetica", 10)
        c.drawString(40, 545, "• Dependency: Waiting for cloud infrastructure provisioning sign-off from IT Security.")
        c.drawString(40, 530, "• Blocked by: Vendor authentication API documentation remains incomplete.")
        
        c.save()
        print(f"Created PDF sample with reportlab: {pdf_path}")
    except ImportError:
        # Fallback PDF generator using raw PDF syntax if reportlab is not installed
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
            b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>> >> endobj\n"
            b"4 0 obj <</Length 600>> stream\n"
            b"BT /F1 16 Tf 40 750 Td (Q4 Project Delivery & Risk Report) Tj ET\n"
            b"BT /F1 12 Tf 40 720 Td (1. Executive Summary & Delivery Health) Tj ET\n"
            b"BT /F1 10 Tf 40 700 Td (Project delivery status is currently AT RISK due to third-party API integration delays.) Tj ET\n"
            b"BT /F1 10 Tf 40 685 Td (Target Milestone Phase 2 is experiencing schedule slippage of 2 weeks.) Tj ET\n"
            b"BT /F1 12 Tf 40 650 Td (2. Schedule Risks & Bottlenecks) Tj ET\n"
            b"BT /F1 10 Tf 40 630 Td (- Schedule Slippage Risk: Third-party API rate limits caused 2-week delay in dataset processing.) Tj ET\n"
            b"BT /F1 10 Tf 40 615 Td (- Resource Bottleneck: Backend team is understaffed by 2 developers during Sprint 4.) Tj ET\n"
            b"BT /F1 10 Tf 40 600 Td (- Technical Complexity: Vector index search latency spike on documents over 100 pages.) Tj ET\n"
            b"BT /F1 12 Tf 40 565 Td (3. Dependency Gaps & External Bottlenecks) Tj ET\n"
            b"BT /F1 10 Tf 40 545 Td (- Dependency: Waiting for cloud infrastructure provisioning sign-off from IT Security.) Tj ET\n"
            b"BT /F1 10 Tf 40 530 Td (- Blocked by: Vendor authentication API documentation remains incomplete.) Tj ET\n"
            b"endstream endobj\n"
            b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000900 00000 n \ntrailer <</Size 6 /Root 1 0 R>>\nstartxref\n977\n%%EOF"
        )
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)
        print(f"Created PDF sample with fallback raw stream: {pdf_path}")

    progress = """=== Sprint Progress Update — 2026-10-05 ===
Date: 2026-10-05
Type: Incremental meeting notes / progress update

1. Progress:
- Milestone Phase 2 ingestion pipeline is in progress.
- Goal: Close remaining Milestone 4 dashboard and documentation items before the final demonstration.

2. New Risks:
- Schedule slippage on Milestone 4 documentation due to late vendor API keys. Behind schedule.
- Resource shortage: QA team is understaffed for multi-format regression testing.

3. New Blocker:
- Blocker: Staging GPU instance shortage delayed model evaluation. Status: Unresolved. Severity: Critical.
- Waiting for vendor authentication API rate-limit upgrade. This is a prerequisite for ingestion load tests.

4. Assigned Action Items:
- Action Item: Publish weekly risk register snapshot. Assigned to: Emily Davis. Priority: High. Due Date: 2026-10-08. Status: Open.
- Action Item: Re-run PDF, DOCX, TXT, and CSV validation suite. Assigned to: Alex Rivera. Priority: High. Due Date: 2026-10-07. Status: In Progress.
"""
    progress_path = os.path.join("sample_data", "progress_update.txt")
    with open(progress_path, "w", encoding="utf-8") as f:
        f.write(progress)
    print(f"Created incremental update sample: {progress_path}")

if __name__ == "__main__":
    create_samples()
