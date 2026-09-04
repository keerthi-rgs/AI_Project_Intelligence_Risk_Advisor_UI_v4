# AI Project Intelligence & Risk Advisor

A Streamlit prototype matching the requested Milestone-1 workflow:

1. Document Upload
2. Ingestion Results
3. Content Preview
4. RAG Indexing
5. Retrieval Explorer
6. Grounded RAG Q&A

## Supported files
- PDF
- DOCX
- TXT
- CSV

## Run in VS Code / terminal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Recommended demo settings
- Chunk Size: 500
- Chunk Overlap: 100
- Retrieval Top-K: 4

The prototype uses TF-IDF + cosine similarity for local retrieval, so it can run without an API key. It also displays retrieved evidence and source locations to demonstrate grounded answers.

For a later milestone, the retriever can be upgraded to embeddings + FAISS/Chroma and the answer generator can be connected to Gemini/OpenAI/Ollama.

## UI design
The interface is styled as a professional project dashboard inspired by the provided reference:
- left configuration/sidebar area
- horizontal six-stage workflow tabs
- clean cards and metrics
- document upload area
- RAG indexing metrics
- retrieval and grounded-answer evidence sections
