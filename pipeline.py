"""
pipeline.py
-----------
Connects rfp_parser.py and bid_memory.py into one workflow.

Step 1: Parse the RFP PDF → extract structured requirements
Step 2: For each requirement, retrieve relevant past bid chunks
Step 3: Print the paired output — requirement + matching experience

This is the raw material a bid drafter would use to write a response.

Usage:
    python3 pipeline.py your_rfp.pdf

Dependencies:
    All dependencies from rfp_parser.py and bid_memory.py must be installed.
    past_bid_1.txt, past_bid_2.txt, past_bid_3.txt must be in the same folder.
"""
  # 1. imports
  # 2. argparse with --step choices
  # 3. if args.step in ("parse", "all"):  → call extract_text + parse_rfp, print JSON
  # 4. if args.step in ("ingest", "all"): → ingest past_bids/, print result dict
  # 5. if args.step in ("draft", "all"):  → draft one hard-coded requirement, print result

#imports from other files and system libraries
from rfp_parser import parse_rfp, extract_text
from bid_memory import ingest_bids, find_relevant_chunks
from response_drafter import draft_section
import sys
import json
import os
import argparse

#creates an argument parser for command-line arguments. It reads whatever the user types after python pipeline_redo.py... and uses in code
parser = argparse.ArgumentParser(description="Diagnostic pipeline runner")

#Defines the arguments that the script accepts. The first two are optional positional arguments (rfp_pdf and bids_dir), and the third is an optional argument (--step) that specifies which part of the pipeline to run.
parser.add_argument("rfp_pdf", nargs="?", help="Path to the RFP PDF file")
parser.add_argument("bids_dir", nargs="?", default="past_bids", help="Folder of past bids")
parser.add_argument("--step", choices=["parse", "ingest", "draft", "all"], default="all") #if no --step is provided, it defaults to "all"

#Parsing and storing arguments in the args variable to be able to use.
args = parser.parse_args()

#if statement runs if user specifies --step parse or all. It extracts text from the PDF and parses it, then prints the resulting JSON.
# If the user did not provide a PDF path, it prints an error message and exits.
if args.step in ("parse", "all"):
    print("\n===STEP: PARSE===\n")

    if not args.rfp_pdf:
        print("Error: --step parse requires a PDF path. Example: python3 pipeline.py your_rfp.pdf --step parse")
        sys.exit(1)

    raw_text = extract_text(args.rfp_pdf)
    print(f"Extracted {len(raw_text)} characters from PDF.")

    parsed_rfp = parse_rfp(raw_text)
    print("Parsed RFP JSON: ")
    print(json.dumps(parsed_rfp, indent=2))


# This if statement runs if user specifies --step ingest or all.
# It checks if the specified bids directory exists and calls the ingest_bids function with the list of file paths and prints the result. 
if args.step in ("ingest", "all"):
    print("\n===STEP: INGEST===\n")
    
    bids_dir = args.bids_dir
    if not os.path.isdir(bids_dir):
        print(f"ERROR: bids directory not found: {bids_dir}")
        sys.exit(1)

    filepaths = [
        os.path.join(bids_dir, f) for f in os.listdir(bids_dir) if f.lower().endswith((".txt", ".pdf"))
    ]

    print(f"Found {len(filepaths)} bid files in '{bids_dir}'")

    ingest_result = ingest_bids(filepaths, client_id="diagnostic")
    print(f"Ingestion result: {ingest_result}")


#This if statement runs if user specifies --step draft or all. It defines a test requirement and uses the find_relevant_chunks function to retrieve relevant chunks from the ingested bids.
# It then calls the draft_section function to generate a draft response based on the test requirement and retrieved chunks, printing the confidence score, reason, and the draft itself. 
# If there is an error during drafting, it prints an error message.
if args.step in ("draft", "all"):
    print("\n===STEP: DRAFT===\n")

    test_requirement = "Demonstrate your firm's experience managing safety programs on construction sites with concurrent trades."

    print(f"Query: '{test_requirement[:60]}...'")
    chunks = find_relevant_chunks(test_requirement, client_id="diagnostic", n=3)
    print(f"Retrieved {len(chunks)} chunks")

    for i, c in enumerate(chunks):
        print(f" [{i+1}] source={c['source']} distance={c['distance']}  text={c['text'][:80]}...")

    result, error = draft_section(test_requirement, chunks)
    if error:
        print(f"Draft failed: {error}")

    else: 
        print(f"Confidence: {result['confidence']}")
        print(f"Reason: {result['confidence_reason']}")
        print(f"\nDraft:\n {result['draft']}")







