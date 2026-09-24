"""Main orchestrator file that acts as a system with every file working together. 
It has three modes:
--mode draft: parse --> ingest --> retrieve --> draft --> save session --> export
--mode load: list sessions --> pick one --> load session --> export

Client_id flows through every call. """

import sys
import os
import argparse
from datetime import date

from rfp_parser import extract_text, parse_rfp
from bid_memory import ingest_bids, find_relevant_chunks
from response_drafter import draft_section
from exporter import export_to_word
from session import save_session, load_session, list_sessions

#reusable function of extracting bid files from directory
def get_bid_files(bids_dir: str) -> list:
    if not os.path.isdir(bids_dir):
        return []
    
    files = [ os.path.join(bids_dir, f) for f in os.listdir(bids_dir) if f.lower().endswith((".txt", ".pdf"))]

    return sorted(files)

#Function to run the draft mode, which executes the full pipeline from parsing the RFP to exporting the drafted response. 
def run_draft_mode(rfp_pdf_path, bids_dir, client_id, firm_name, output_path):
    print("\n" + "=" * 60)
    print(f" MODE: DRAFT | CLIENT: {client_id}")
    print("\n" + "=" * 60)

    #Validating if RFP file and bid files exists properly
    if not os.path.exists(rfp_pdf_path):
        return None, f"RFP file not found: {rfp_pdf_path}"
    
    bid_files = get_bid_files(bids_dir)
    if not bid_files:
        return None, f"No .txt or .pdf files found in: {bids_dir}"
    
    print(f"RFP: {rfp_pdf_path}")
    print(f"Bids_dir: {bids_dir} ({len(bid_files)} files)")
    print(f"Firm: {firm_name}")


    #Step 1: Parsing RFP using parse_rfp function, which uses LLM to extract key requirements from the pdf and project name.
    print("\n[1/4] Parsing RFP...")
    raw_rfp = extract_text(rfp_pdf_path)

    if not raw_rfp.strip():
        return None, "PDF produced no text. Is it a scanned/image-only PDF?"

    parsed_rfp = parse_rfp(raw_rfp)
    requirements = parsed_rfp.get('key_requirements',[])
    rfp_name = parsed_rfp.get("project_name") or os.path.basename(rfp_pdf_path)

    if not requirements:
        return None, "No requirements found. Check parse_rfp output."
    

# Step 2: Ingesting Bids using ingest_bids function, which uses LLM to chunk the bid documents and store them in a vector database for retrieval.
    print("\n[2/4] Ingesting Bids...")
    ingest_result = ingest_bids(bid_files, client_id=client_id)
    if ingest_result["succeeded"] == 0:
        return None, "Bid documents failed to ingest. Check VOYAGE_API_KEY"
    
    print(f"{ingest_result['succeeded']} documents, {ingest_result['total_chunks']} chunks")

# Step 3: Drafting Sections using draft_section function, which retrieves relevant chunks from the vector database for each requirement and uses LLM to draft a response section. 
# It also assigns a confidence level to each drafted section based on the relevance and quality of the retrieved information.
    print(f"\n[3/4] Drafting {len(requirements)} sections...")
    draft_sections = []

#this loop iterates through each requirement extracted from the RFP, retrieves relevant chunks from the ingested bid documents, and attempts to draft a response section for each requirement.
    for i,req in enumerate(requirements):
        req_text = req if isinstance(req, str) else req.get("text", "")
        if not req_text.strip():
            continue

        print(f"\n  [{i+1}/{len(requirements)}] {req_text[:65]}...")

        chunks = find_relevant_chunks(req_text, client_id=client_id, n=3)
        result, error = draft_section(req_text, chunks)

        if error:
            print(f"Draft failed: {error}")
            draft_sections.append({
                "requirement": req_text,
                "draft": f"\n[DRAFT FAILED]\n\n [NEEDS CUSTOM INPUT] Write this section manually.\n\nError: {error}",
                "confidence": "LOW",
                "confidence_reason": f"Automated drafting failed: {error}"
            })
        else:
            draft_sections.append(result)
            print(f"Confidence: {result['confidence']}")


# Step 4: Saving Session and Exporting using save_session and export_to_word functions. The session is saved with all relevant information for future retrieval, and the drafted sections are exported to a Word document. 
    print("\n[4/4] Saving session + exporting...")
    session_path = save_session(rfp_name, parsed_rfp, draft_sections, client_id)
    print(f"Session saved: {session_path}")

# The output file is named based on the RFP name and current date for easy identification. 
    if output_path is None:
        safe_name = rfp_name.replace(" ", "_")[:40]
        output_path = f"rfp_draft_{safe_name}_{date.today().isoformat()}.docx"

    saved_path, export_error = export_to_word(rfp_name, draft_sections, output_path, firm_name)
    if export_error:
        return None, f"Word export failed: {export_error}"
    

    high = sum(1 for s in draft_sections if s['confidence'] == "HIGH")
    medium = sum(1 for s in draft_sections if s['confidence'] == "MEDIUM")
    low = sum(1 for s in draft_sections if s['confidence'] == "LOW")

    print("\n" + "=" * 60)
    print(" DONE")
    print(f"Document: {saved_path}")
    print(f"Confidence: {high} HIGH | {medium} MEDIUM | {low} LOW")
    print("=" * 60)

    return saved_path, None


#Function to load a previously saved session. 
# It lists all sessions for the client, allows the user to select one, and then loads the session data to export it again to Word. 
def run_load_mode(client_id, firm_name):
    print("\n" + "=" * 60)
    print(f"\n MODE: LOAD | CLIENT: {client_id}")
    print("=" * 60)

    sessions = list_sessions(client_id)
    if not sessions:
        return None, f"No saved sessions found for client: {client_id}"
    
    print(f"Saved sessions for '{client_id}':")
    for i, name in enumerate(sessions):
        print(f"[{i+1}] {name}")

    session_number = input("\nEnter a session number to load (or 'q' to quit): ").strip()

    if session_number.lower() == "q":
        print("Cancelled.")
        return None, None
    
    try:
        index = int(session_number) - 1
        if index < 0 or index >= len(sessions):
            return None, f"Invalid session number: {session_number}. Please enter a number between 1 and {len(sessions)}."
    except ValueError:
        return None, f"'{session_number}' is not a number. Enter the number shown next to the session."
    
    filename = sessions[index]
    data = load_session(filename, client_id)
    if data is None:
        return None, f"Could not load session '{filename}'. It may be corrupted or belong to another client."
    
    rfp_name = data["rfp_name"]
    draft_sections = data["draft_sections"]
    print(f"Loaded: {rfp_name} ({len(draft_sections)} sections)")

    safe_name = rfp_name.replace(" ", "_")[:40]
    output_path = f"rfp_draft_{safe_name}_{date.today().isoformat()}.docx"

    saved_path, export_error = export_to_word(rfp_name, draft_sections, output_path, firm_name)
    if export_error:
        return None, f"Word export failed: {export_error}"
    
    print(f"  Document: {saved_path}")
    return saved_path, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "RFP Response Pipeline - draft mode or load mode",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog="""
  Examples:
    python3 main.py rfp.pdf past_bids/ --mode draft --client acme
    python3 main.py rfp.pdf past_bids/ --mode draft --client acme --firm "Acme Builders"
    python3 main.py --mode load --client acme
          """
    )

    parser.add_argument("rfp_pdf",  nargs="?", help="Path to RFP PDF (required for --mode draft)")
    parser.add_argument("bids_dir", nargs="?", default="past_bids", help="Folder of past bid files")
    parser.add_argument("--mode",   choices=["draft", "load"], default="draft")
    parser.add_argument("--client", default="default", help="Client ID — scopes sessions and ChromaDB")
    parser.add_argument("--firm",   default="[FIRM NAME — UPDATE BEFORE SENDING]")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.mode == "draft":
        if not args.rfp_pdf:  
            print("ERROR: --mode draft requires a PDF path.")
            print("Usage: python3 main.py rfp.pdf past_bids/ --mode draft")
            sys.exit(1)
        output, error = run_draft_mode(args.rfp_pdf, args.bids_dir, args.client, args.firm, args.output)

    else:
        output, error = run_load_mode(args.client, args.firm)

    if error:
        print(f"\n {error}")
        sys.exit(1)

    sys.exit(0)








    


