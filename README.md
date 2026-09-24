# Liberate Systems - RFP Response Assistant

A retrieval-augmented generation pipeline that extracts requirements from engineering RFPs and drafts responses using a firm's own past bid documents as reference material.

Built as the core product of an independent venture (May–Sept 2026) serving BC engineering and construction consultancies.

What it does
Parses an RFP PDF and extracts structured data — project name, client, deadline, scope summary, key requirements, evaluation criteria — via PyMuPDF text extraction + an LLM call.
Retrieves the most relevant passages from a firm's past bid documents for each requirement, using Voyage AI embeddings indexed in ChromaDB.
Drafts a response for each requirement via the Anthropic API, written in the voice of the firm's past bids, with a confidence score (HIGH / MEDIUM / LOW) and a flag on any section needing custom human input.
Exports the result as a formatted Word document — cover page, sections with confidence footers, and an action-items checklist of everything flagged for review.

Runs behind a Streamlit interface. Processes a 60-page RFP in under a minute.

Architecture
rfp_parser.py      → PDF text extraction + structured JSON (PyMuPDF + Anthropic API)
bid_memory.py       → Chunking, embedding, and retrieval over past bids (Voyage AI + ChromaDB)
pipeline.py          → Wires parsing to retrieval — RFP in, ranked chunks per requirement out
response_drafter.py  → Drafts each section with a confidence score (Anthropic API)
exporter.py          → Formats the final Word document (python-docx)
main.py             → CLI entry point: `python3 main.py rfp.pdf past_bids/`
rfp_app.py             → Streamlit UI wrapping the pipeline


Setup
bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
export VOYAGE_API_KEY=your_key
Usage
bash
# CLI
python3 main.py path/to/rfp.pdf path/to/past_bids/

# Or launch the UI
streamlit run app.py

Evaluation

The functionality of the app worked well with synthesized documents, the extraction, comparison, and drafting. However, the issue was with the. later compliance test where the search was passed with single-word comparisons. Although the program was built to behave like this, it created unreliable output that could not be shipped. 

An AI-assisted multi-agent version of the drafting tool (not in repo) was created to build an improved version with additional features. In order to test the functions, I built a hand-labelled ground-truth rubric and tested the extraction/evaluation output against it on identical inputs. Results:

~26% output variance across repeated runs on the same input
6 missed mandatory requirements out of the test set

Based on that, I descoped the multi-agent approach to a single-pass, human-verified assist rather than ship output the evaluation showed was unreliable. 

Status

This was built and validated through 16 customer-discovery interviews with BC engineering and construction firms. After the evidence gained through those interviews, finding that 2 of 16 firms had already substituted a general-purpose AI chatbot for the same task, competitors, mitigated pain, and a $750 paid-pilot offer was declined, the standalone product was discontinued as a SaaS offering. Liberate is now shifted towards automating repetitive tasks in workflows. 

Stack

Python · Anthropic API · Voyage AI · ChromaDB · PyMuPDF · python-docx · Streamlit
