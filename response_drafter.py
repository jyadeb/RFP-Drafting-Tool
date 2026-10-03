"""
Purpose:
  Implements the 3-step chaining workflow for RFP bid response drafting.
    Step 1 (PARSE)    – Understand what the requirement is actually asking for
    Step 2 (RETRIEVE) – Find relevant past bid chunks
    Step 3 (DRAFT)    – Write the bid section using requirement + evidence

  Separate calls for each different cognitive job for Claude to have a single, focused prupose on each call rather than simultaenously doing all of them. 

Usage:
  result = draft_section(requirement_text, retrieved_chunks)
  
  result is a dict: {
      "requirement": str,
      "draft": str,
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "confidence_reason": str
  }
"""

import os
import json
import anthropic
from dotenv import load_dotenv

# environment setup, securely storing and accessing api keys
load_dotenv()

# Create one Anthropic client. We reuse this across all three API calls
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# The model to use. Sonnet balances quality and cost well for generation tasks.
MODEL = "claude-sonnet-5-5"

# These system prompts are defined as module-level constants as they are configuration and you can load from files or prompt registry in prod. 
# Parse system prompt for extracting structured meaning from raw RFP requirement text.
PARSE_SYSTEM_PROMPT = """You are an expert at analyzing RFP (Request for Proposal) requirements 
for BC construction and engineering firms.

Your ONLY job in this step is to parse and structure what a requirement is asking for.
Do NOT attempt to answer or draft anything.

For each requirement, extract:
1. The core capability or experience being requested
2. Any specific qualifications, certifications, or standards mentioned
3. The type of evidence that would best satisfy this requirement (project examples, certifications, team credentials, methodologies)
4. Keywords that would appear in relevant past bid experience

<output_format>
Respond ONLY with a JSON object. No preamble, no explanation outside the JSON.
{
  "core_ask": "1-2 sentences describing what this requirement fundamentally wants",
  "specific_qualifiers": ["list", "of", "specific", "things", "required"],
  "evidence_types_needed": ["what kinds of evidence would satisfy this"],
  "search_keywords": ["keywords", "to", "find", "relevant", "past", "bids"]
}
</output_format>"""


# Draft system prompt which determines the quality of the final output applying XML tags, reference example outputs, and
# explicit rules to prevent specific failure modes.
DRAFT_SYSTEM_PROMPT =DRAFT_SYSTEM_PROMPT = """You are drafting a bid response section for a BC construction 
firm responding to a BC Housing RFP.

<role_and_context>
You write in the voice of an experienced BC construction company responding to a 
competitive public procurement. Your reader is a BC Housing evaluator who reads 
dozens of bids and scores them against specific criteria. They reward specificity, 
concrete proof, and direct relevance to the project at hand. They penalize vague 
assertions, generic claims, and padding.

Write in plain declarative sentences. Professional but direct — not corporate, 
not salesy. The evidence does the convincing, not adjectives.
</role_and_context>

<strict_rules>
DO NOT invent, imply, or extrapolate:
  - Certifications or professional designations not in the excerpts
  - Project experience for locations, types, or scales not in the excerpts
  - Team member credentials not shown in the excerpts
  - Awards, recognitions, or affiliations not mentioned in the excerpts

If a requirement cannot be addressed with the provided evidence, output 
[NEEDS CUSTOM INPUT] at the exact point where information is missing and 
specify precisely what is needed.

Keep each section under 250 words unless the requirement demands technical detail.

DO NOT use em-dashes. Use commas or separate sentences instead.

Use correct natural article usage throughout: write "achieved a Passive House 
Classic certification", not "achieved Passive House Classic certification". 
Write "holds a COR certification", not "holds COR certification".

DO NOT open with the firm's name or "Our firm". Vary sentence openings.

DO NOT use adjectives to claim quality — exceptional, outstanding, proven, 
robust. Let specific numbers, certifications, and outcomes make the case instead.

Use natural article usage throughout: "achieved a Passive House Classic 
certification", not "achieved Passive House Classic certification".

If no past bid excerpts are provided, do NOT open with generic capability 
statements. Begin the response immediately with [NEEDS CUSTOM INPUT] and 
specify what evidence is required.
</strict_rules>

<connection_rule>
For every piece of evidence cited from past bids, add one sentence connecting 
that outcome to why it is directly relevant to this specific requirement. 
Do not leave the connection implicit — state it explicitly.

Example of weak evidence citation:
"All 14 trade packages were tendered competitively."

Example of correct evidence citation with connection:
"All 14 trade packages on Fir Street Commons were tendered competitively with 
a minimum of three qualified bidders per package. For Vienna House, this same 
pre-qualification approach would be applied from pre-construction to control 
both bidder pool quality and cost risk at award."
</connection_rule>

<confidence_scoring>
Assign confidence based strictly on how well the provided excerpts support 
the requirement:
  HIGH   — Excerpts contain direct, specific experience that clearly satisfies 
            the requirement
  MEDIUM — Excerpts contain partially relevant experience; some inference required
  LOW    — Excerpts are tangentially related or insufficient

Be honest. A LOW with [NEEDS CUSTOM INPUT] is more valuable to the firm than 
a falsely confident HIGH that embarrasses them during evaluation.
</confidence_scoring>

<example>
<requirement_input>
Demonstrate experience managing construction projects to BC Housing Design and 
Construction Standards, including full electrification and zero fossil fuel usage.
</requirement_input>
<past_bid_excerpts_input>
[Excerpt: Fir Street Commons 2020] Building achieved full electrification via 
air-source heat pump systems with no fossil fuel use. Greenhouse gas reduction 
of 74% vs. baseline. Designed and built to BC Housing Design and Construction 
Standards throughout.

[Excerpt: Strathcona Living 2023] Full electrification, no fossil fuels. 
GHG reduction of 81% vs. ASHRAE 90.1 baseline. Subject to City of Vancouver 
Green Building Programs Branch oversight.
</past_bid_excerpts_input>
<ideal_output>
{
  "draft": "Two completed projects demonstrate direct compliance with BC Housing 
Design and Construction Standards and the zero fossil fuel requirement. Fir Street 
Commons, a 88-unit affordable rental building in Vancouver, achieved full 
electrification through an air-source heat pump system with a 74% greenhouse gas 
reduction against baseline. Strathcona Living, an 112-unit mass timber affordable 
housing project, achieved full electrification with an 81% GHG reduction against 
the ASHRAE 90.1 baseline and was subject to City of Vancouver Green Building 
Programs Branch oversight throughout construction. Both projects were delivered 
under BC Housing Supplementary General Conditions — the same contract framework 
applicable to Vienna House — confirming familiarity with BC Housing's specific 
standards, reporting requirements, and approval processes.",
  "confidence": "HIGH",
  "confidence_reason": "Two excerpts directly confirm full electrification, 
  zero fossil fuel use, and delivery under BC Housing Design and Construction 
  Standards with specific GHG reduction figures."
}
</ideal_output>
</example>

<instructions>
You will receive a <requirement> block and a <past_bid_excerpts> block.
Use ONLY the evidence in the excerpts.
If no excerpts are provided or excerpts are insufficient, open directly with 
[NEEDS CUSTOM INPUT] — do not open with a generic capability statement.

Respond ONLY with a JSON object. No text outside the JSON.
{
  "requirement": "exact text of the requirement",
  "draft": "the written bid response section",
  "confidence": "HIGH or MEDIUM or LOW",
  "confidence_reason": "one sentence explaining the confidence level"
}
</instructions>"""


#HELPER FUNCTIONS

# wraps around every Anthropic API call to handle errors like network issues, rate limits, improper inputs. 
def safe_api_call(messages: list, system_prompt: str, step_name: str):
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        # For a standard text response, we want content[0].text.
        # The strip() removes any leading/trailing whitespace.
        return response.content[0].text.strip(), None
    
    except anthropic.AuthenticationError as e:
        # Bad API key, no point retrying
        return None, f"[{step_name}] Authentication failed. Check your ANTHROPIC_API_KEY: {e}"
    
    except anthropic.RateLimitError as e:
        return None, f"[{step_name}] Rate limit hit. Consider adding retry logic: {e}"
    
    except anthropic.APIError as e:
        return None, f"[{step_name}] API error: {e}"


def parse_json_response(raw_text: str, step_name: str):
    # Strip markdown code fences if present
    cleaned = raw_text
    if cleaned.startswith("```"):
        # Find the first newline (end of opening fence) and last ``` (closing fence)
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            cleaned = cleaned[first_newline:last_fence].strip()
    
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        return None, f"[{step_name}] Failed to parse JSON response: {e}\nRaw text: {raw_text[:200]}"


# Step 1 of the chain, parsing a raw RFP requirement into structured form 
def parse_requirement(requirement_text: str):
    messages = [
        {
            "role": "user",
            "content": f"Parse this RFP requirement:\n\n{requirement_text}"
        }
    ]
    
    raw_response, error = safe_api_call(messages, PARSE_SYSTEM_PROMPT, "PARSE")
    if error:
        return None, error
    
    parsed, error = parse_json_response(raw_response, "PARSE")
    if error:
        return None, error
    
    return parsed, None


# Format retrieved RAG chunks into the XML prompt structure
def format_chunks_for_prompt(retrieved_chunks: list) -> str:
    if not retrieved_chunks:
        return "<past_bid_excerpts>\nNo relevant past bid excerpts found.\n</past_bid_excerpts>"
    
    excerpts_xml = "<past_bid_excerpts>\n"
    for chunk in retrieved_chunks:
        # Pull source from the chunk dict, default to "unknown" if not present
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        excerpts_xml += f'<excerpt source="{source}">\n{text}\n</excerpt>\n'
    
    excerpts_xml += "</past_bid_excerpts>"
    return excerpts_xml


# main function that runs parse, formatting chunks, and draft. Also, validates that the draft's JSON has the
# required keys (requirement, draft, confidence, confidence_reason), falling back to LOW confidence if the model returns something unexpected.

def draft_section(requirement_text: str, retrieved_chunks: list) -> tuple:
  # Step 1: Parse requirements
    print(f"\n[CHAIN] Step 1: Parsing requirement...")
    parsed_req, error = parse_requirement(requirement_text)
    if error:
        print(f"  ✗ Parse failed: {error}")
        # If parsing fails, continue without it.
        # The draft step can still run on raw requirement text.
        parsed_req = None
    else:
        print(f"  ✓ Core ask: {parsed_req.get('core_ask', 'unknown')}")
    
    # Step 2: Format retrieved chunks
    print(f"[CHAIN] Step 2: Formatting {len(retrieved_chunks)} retrieved chunks...")
    formatted_excerpts = format_chunks_for_prompt(retrieved_chunks)
    
    # Step 3: Draft the bid section, building the user message with XML tags
    print(f"[CHAIN] Step 3: Drafting bid section...")
    
    user_message = f"""<requirement> 
    {requirement_text} 
    </requirement>
    
    {formatted_excerpts}"""
    
    # If parsing succeeded, we can optionally include the structured understanding
    # to give the model even more context.
    if parsed_req:
        core_ask = parsed_req.get("core_ask", "")
        evidence_types = ", ".join(parsed_req.get("evidence_types_needed", []))
        user_message += f"""

<parsed_requirement_context>
Core ask: {core_ask}
Evidence types that would satisfy this: {evidence_types}
</parsed_requirement_context>"""
    
    messages = [{"role": "user", "content": user_message}]
    
    raw_response, error = safe_api_call(messages, DRAFT_SYSTEM_PROMPT, "DRAFT")
    if error:
        return None, error
    
    result, error = parse_json_response(raw_response, "DRAFT")
    if error:
        return None, error
    
    # Validate the result has the expected keys 
    required_keys = {"requirement", "draft", "confidence", "confidence_reason"}
    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        return None, f"[DRAFT] Response missing required keys: {missing_keys}\nGot: {list(result.keys())}"
    
    # Validate confidence is a valid value
    valid_confidence = {"HIGH", "MEDIUM", "LOW"}
    if result.get("confidence") not in valid_confidence:
        # Don't fail hard, just normalise to LOW if unexpected value
        result["confidence"] = "LOW"
        result["confidence_reason"] = f"(original confidence value was invalid: {result.get('confidence')})"
    
    print(f"  ✓ Draft complete. Confidence: {result['confidence']}")
    return result, None


# Test block only runs when you execute this file directly
if __name__ == "__main__":
    
    print("=" * 60)
    print("RESPONSE DRAFTER — 3-STEP CHAIN TEST")
    print("=" * 60)
    
    # ── Test Case 1: Well-supported requirement ────────────────────────────────
    # HIGH confidence expected because the excerpts directly address the requirement
    
    requirement_1 = """Demonstrate your firm's experience managing safety programs on 
    construction sites with concurrent trades. Provide evidence of your safety record 
    including any COR (Certificate of Recognition) certification and total recordable 
    incident rate (TRIR) over the past three years."""
    
    chunks_1 = [
        {
            "source": "Langley_Commercial_Bid_2023.txt",
            "text": """Our firm holds an active COR certification through the BC 
            Construction Safety Alliance, renewed in 2023. On our Langley commercial 
            project, we managed 6 concurrent trade contractors across 18 months, 
            achieving zero lost-time incidents. Our TRIR over the past three years 
            is 0.8, well below the BC construction industry average of 2.4."""
        },
        {
            "source": "Abbotsford_Industrial_Bid_2022.txt",
            "text": """Site safety on this project was managed through our internal 
            safety management system, including weekly toolbox talks, daily hazard 
            assessments, and monthly third-party safety audits. Our project safety 
            coordinator holds a NCSO designation through BCIT."""
        }
    ]
    
    print(f"\nTEST 1: Well-supported requirement")
    print(f"Requirement excerpt: '{requirement_1[:80]}...'")
    
    result_1, error = draft_section(requirement_1, chunks_1)
    
    if error:
        print(f"\n✗ Error: {error}")
    else:
        print(f"\n── RESULT ──")
        print(f"Confidence:  {result_1['confidence']}")
        print(f"Why:         {result_1['confidence_reason']}")
        print(f"\nDraft:\n{result_1['draft']}")
    
    print("\n" + "=" * 60)
    
    # ── Test Case 2: Poorly-supported requirement ──────────────────────────────
    # LOW confidence expected — chunks don't match the requirement well
    
    requirement_2 = """Demonstrate experience with Indigenous community consultation 
    and engagement, including a demonstrated track record of building long-term 
    relationships with First Nations communitiess in BC."""
    
    chunks_2 = [
        {
            "source": "Kelowna_Residential_Bid_2022.txt",
            "text": """Our project team coordinated with the City of Kelowna's 
            development permit office throughout the approval process, maintaining 
            clear communication with all municipal stakeholders."""
        }
    ]
    
    print(f"\nTEST 2: Poorly-supported requirement (expects LOW confidence + [NEEDS CUSTOM INPUT])")
    print(f"Requirement excerpt: '{requirement_2[:80]}...'")
    
    result_2, error = draft_section(requirement_2, chunks_2)
    
    if error:
        print(f"\n✗ Error: {error}")
    else:
        print(f"\n── RESULT ──")
        print(f"Confidence:  {result_2['confidence']}")
        print(f"Why:         {result_2['confidence_reason']}")
        print(f"\nDraft:\n{result_2['draft']}")
    
    print("\n" + "=" * 60)
    
    # ── Test Case 3: No excerpts at all ───────────────────────────────────────
    # Absolute LOW — should be almost entirely [NEEDS CUSTOM INPUT]
    
    requirement_3 = """Provide a detailed quality management plan including your 
    ISO 9001:2015 certification documentation and describe your non-conformance 
    tracking and resolution process."""
    
    print(f"\nTEST 3: No supporting excerpts (expects LOW confidence)")
    print(f"Requirement excerpt: '{requirement_3[:80]}...'")
    
    result_3, error = draft_section(requirement_3, [])
    
    if error:
        print(f"\n✗ Error: {error}")
    else:
        print(f"\n── RESULT ──")
        print(f"Confidence:  {result_3['confidence']}")
        print(f"Why:         {result_3['confidence_reason']}")
        print(f"\nDraft:\n{result_3['draft']}")
