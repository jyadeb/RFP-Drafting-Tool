"""
rfp_parser.py
-------------
Takes a PDF path as a command-line argument, extracts text using PyMuPDF,
then calls Claude to parse it into structured JSON.

Usage:
    python3 rfp_parser.py your_rfp.pdf

Dependencies:
    pip3 install anthropic pymupdf python-dotenv --break-system-packages
"""
#imports for sys.argv, json.loads(), os.path, PyMuPDF, Claude SDK, and env file reading
import sys       
import json     
import os          
import fitz         
import anthropic
from dotenv import load_dotenv

# Loads api-key from .env file 
load_dotenv()

# ─────────────────────────────────────────────
# FUNCTION 1: Extract raw text from a PDF file
# ─────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """
      Opens a PDF and returns all its text as a single string. 
    
    """

    # Checks if the file actually exists before PyMuPDF tries to open it
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # list to collect each page's text
    all_text = []

    # fitz.open() opens the PDF and returns a Document object. It is already parsed by PyMuPDF and gives us a Document we can iterate over
    with fitz.open(pdf_path) as doc:

        # doc is iterable, each item is a Page object
        print(f"  PDF opened: {len(doc)} pages found")

        #loop to store a string with each page's text content using .get_text() and append to all_text if content exists
        for page_num, page in enumerate(doc):
            page_text = page.get_text()

            if page_text.strip():
                all_text.append(page_text)

    # Joining the content of all pages with a visual break for Claude
    full_text = "\n\n".join(all_text)

    print(f"  Text extracted: {len(full_text)} characters across {len(all_text)} pages")
    return full_text


# ─────────────────────────────────────────────
# HELPER: Clean up Claude's response into valid JSON
# ─────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """
    Defensive JSON parser.

    WHY this exists:
        Even with a well-crafted prompt, Claude occasionally wraps its
        JSON in markdown code fences (```json ... ```) or adds a one-line
        preamble. json.loads() fails on anything that isn't pure JSON.

        This function strips those wrappers before parsing.

    The two failure modes it handles:
        1. Claude returns:  ```json\n{...}\n```
        2. Claude returns:  Here is the JSON:\n{...}
    """

    text = text.strip()

    # Handle markdown code fences: ```json ... ``` or ``` ... ``` by splitting the text by lines
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()# Joins the lines with newlines in between while exclduing the first line (```json) and the last line (```)

    # attempt to parse the cleaned JSON text.
    return json.loads(text)


# ─────────────────────────────────────────────
# FUNCTION 2: Parse RFP text into structured JSON using Claude
# ─────────────────────────────────────────────

def parse_rfp(text: str) -> dict:
    """
    Sends the extracted RFP text to Claude and returns a structured dict.

    WHY a detailed system prompt:
        Claude is a language model — it defaults to conversational responses.
        Without explicit instructions it might say "Sure! Here's what I found..."
        which breaks json.loads() immediately.

        Four techniques used here to force clean JSON:
        1. Explicit instruction: "Return ONLY valid JSON"
        2. Schema definition: exact keys, types, and fallback values
        3. Example of expected output format
        4. Defensive strip in parse_json_response() as a safety net
    """

    #creates API client object with api-key in .env
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # The system prompt is the most important part as it gives Claude its role, output format, instructions, and rules.
    system_prompt = """You are a construction procurement analyst. Your only job is to extract
structured data from RFP documents and return it as valid JSON.

CRITICAL RULES — follow these exactly:
- Return ONLY valid JSON. No explanation, no markdown, no backticks, no preamble.
- Your entire response must be parseable by Python's json.loads() with no preprocessing.
- If a field cannot be found in the document, use null for strings and false for booleans.
- For lists (key_requirements, evaluation_criteria), return an empty list [] if not found.
- Deduplicate requirements — if two requirements cover the same topic, 
  combine them into one. Maximum 6 requirements, each covering a distinct topic.

Extract the following fields and return them in this exact JSON structure:

{
  "project_name": "the full official name of the project being procured",
  "client_org": "the organization issuing the RFP (the owner/buyer)",
  "submission_deadline": "the proposal closing date and time as written in the document",
  "scope_summary": "2-5 sentence summary of what the contractor must deliver",
  "key_requirements": ["requirement 1", "requirement 2", "...up to 8 most important requirements"],
  "evaluation_criteria": [
    {"criterion": "criterion name", "weight": "percentage or null if not specified"}
  ],
  "bonding_required": true or false based on whether the RFP requires bid bonds or performance bonds
}

Return nothing except this JSON object."""

    # Truncate the text if it's very long - Claude has a context window limit
    MAX_CHARS = 200000
    if len(text) > MAX_CHARS:
        print(f"  Warning: text truncated from {len(text)} to {MAX_CHARS} characters")
        text = text[:MAX_CHARS]

    print("\n  Calling Claude API...")

    #The API call to create the message
    message = client.messages.create(
        model="claude-sonnet-4-5",   # sonnet: fast enough, smart enough for extraction
        max_tokens=1024,
        temperature=0,               # deterministic output so no creativity needed here
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Extract structured data from this RFP document:\n\n{text}"
            }
        ]
    )

    # message.content is a list of content blocks. For a standard text response there's only one block.
    raw_response = message.content[0].text

    # Parse and return — parse_json_response handles cleanup if needed
    return parse_json_response(raw_response)


# ─────────────────────────────────────────────
# MAIN: Wire everything together
# ─────────────────────────────────────────────

def main():
    """
    Entry point. Reads the PDF path from the command line, runs the pipeline,
    prints the JSON result.

    WHY sys.argv:
        sys.argv is a list of command-line arguments.
        sys.argv[0] is always the script name itself.
        sys.argv[1] is the first argument the user passes — in our case, the PDF path.
        Running: python3 rfp_parser.py test_rfp.pdf
        Gives:   sys.argv = ["rfp_parser.py", "test_rfp.pdf"]
    """

    # Guard: make sure the user actually provided a PDF path
    if len(sys.argv) < 2:
        print("Usage: python3 rfp_parser.py <path_to_pdf>")
        print("Example: python3 rfp_parser.py vienna_house_rfp.pdf")
        sys.exit(1)  # exit with error code 1 — signals something went wrong

    pdf_path = sys.argv[1]

    print(f"\nParsing RFP: {pdf_path}")
    print("-" * 40)

    # Step 1: Extract text from PDF
    print("Step 1: Extracting text from PDF...")
    raw_text = extract_text(pdf_path)

    # Step 2: Parse with Claude
    print("\nStep 2: Parsing with Claude...")
    result = parse_rfp(raw_text)

    # Step 3: Print clean, readable JSON
    # json.dumps() converts a Python dict back to a formatted JSON string
    # indent=2 makes it human-readable with 2-space indentation
    print("\nStep 3: Result:")
    print("-" * 40)
    print(json.dumps(result, indent=2))
    print("-" * 40)
    print(f"\nDone. Extracted {len(result.get('key_requirements', []))} key requirements "
          f"and {len(result.get('evaluation_criteria', []))} evaluation criteria.")


# Standard Python pattern: only run main() if this script is executed directly
# (not imported as a module by another file)
if __name__ == "__main__":
    main()