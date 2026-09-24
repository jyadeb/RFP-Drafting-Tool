"""
exporter.py
-----------
Generates a formatted Word (.docx) document from bid draft sections.

PURE-PYTHON VERSION (python-docx)
  This used to be a thin bridge that wrote JSON to a temp file and called
  `exporter.js` via a Node.js subprocess. That worked fine on a laptop where
  Node is installed, but it cannot run on Streamlit Cloud — that environment is
  a Python-only container with no `node` binary and no `npm install` step. So
  the document generation has been reimplemented here in pure Python using the
  `python-docx` package. No subprocess, no Node, no temp files.

  The visual layout (cover page, per-section blocks, confidence badges,
  action-items page, footer with page numbers) is reproduced as closely as
  python-docx allows so the output matches the previous JS exporter.

Public API (unchanged — what rfp_app.py / main.py call):
    result_path, error = export_to_word(rfp_name, draft_sections, output_path)
"""

import os
import re
from datetime import date

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ─── COLOUR PALETTE ─────────────────────────────────────────────────────────
# RGBColor objects for text, and bare hex strings (no #) for cell shading,
# which is applied through raw XML.
class C:
    BRAND_BLUE = RGBColor(0x1F, 0x4E, 0x79)
    MID_BLUE   = RGBColor(0x2E, 0x75, 0xB6)
    HIGH_GREEN = RGBColor(0x37, 0x56, 0x23)
    MED_ORANGE = RGBColor(0x7F, 0x4B, 0x1A)
    LOW_RED    = RGBColor(0x7B, 0x1F, 0x1F)
    TEXT_DARK  = RGBColor(0x1A, 0x1A, 0x1A)
    TEXT_MUTED = RGBColor(0x59, 0x59, 0x59)

# Hex fills (no leading #) for table-cell / shading XML
FILL_MID_BLUE = "2E75B6"
FILL_HIGH_BG  = "E2EFDA"
FILL_MED_BG   = "FCE4D6"
FILL_LOW_BG   = "FDECEA"
FILL_FLAG_BG  = "FFF2CC"   # soft amber for [NEEDS CUSTOM INPUT] callouts
FLAG_ACCENT   = "BA7517"   # amber accent bar / tag colour

FONT = "Arial"


def _confidence_style(level: str):
    """Return (text_colour, cell_fill_hex) for a confidence level."""
    if level == "HIGH":
        return C.HIGH_GREEN, FILL_HIGH_BG
    if level == "MEDIUM":
        return C.MED_ORANGE, FILL_MED_BG
    return C.LOW_RED, FILL_LOW_BG


# ─── LOW-LEVEL XML HELPERS ───────────────────────────────────────────────────
# python-docx exposes most styling through its API, but a few things
# (paragraph borders, cell shading, the PAGE field) require touching the
# underlying OOXML directly. Each helper below does one such thing.

def _set_paragraph_border(paragraph, edges: dict):
    """
    Add borders to a paragraph. `edges` maps an edge name to a spec dict, e.g.
        {"bottom": {"sz": 6, "color": "2E75B6", "space": 1}}
    Used for the horizontal-rule dividers and the quoted-requirement left bar.
    """
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        if edge in edges:
            spec = edges[edge]
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), spec.get("val", "single"))
            el.set(qn("w:sz"), str(spec.get("sz", 6)))
            el.set(qn("w:space"), str(spec.get("space", 1)))
            el.set(qn("w:color"), spec.get("color", "000000"))
            pBdr.append(el)
    pPr.append(pBdr)


def _shade_cell(cell, fill_hex: str):
    """Set a solid background fill on a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def _set_cell_border(cell, edge: str, color_hex: str, sz: int = 4):
    """Put a coloured border on one edge of a cell ('left', 'right', ...)."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), color_hex)
    borders.append(el)
    tcPr.append(borders)


def _add_page_number_field(paragraph):
    """Append a live { PAGE } field to a paragraph (used in the footer)."""
    run = paragraph.add_run()
    run.font.name = FONT
    run.font.size = Pt(8)
    run.font.color.rgb = C.TEXT_MUTED

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


# ─── HIGH-LEVEL CONTENT HELPERS ──────────────────────────────────────────────

def _styled_run(paragraph, text, *, size=11, color=C.TEXT_DARK, bold=False,
                italic=False, all_caps=False, highlight=None):
    """Add a text run to a paragraph with our standard font applied."""
    run = paragraph.add_run(text)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.bold = bold
    run.italic = italic
    if all_caps:
        run.font.all_caps = True
    if highlight is not None:
        run.font.highlight_color = highlight
    return run


def _spacer(doc):
    """An empty paragraph for vertical spacing."""
    return doc.add_paragraph()


def _horizontal_rule(doc):
    """A thin blue divider line (empty paragraph with a bottom border)."""
    p = doc.add_paragraph()
    _set_paragraph_border(p, {"bottom": {"sz": 6, "color": FILL_MID_BLUE, "space": 1}})
    return p


def _build_cover_page(doc, rfp_name, firm_name, date_str):
    for _ in range(4):
        _spacer(doc)

    # Firm name
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    _styled_run(p, firm_name, size=26, color=C.BRAND_BLUE, bold=True)

    _horizontal_rule(doc)
    _spacer(doc)

    # Document type
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    _styled_run(p, "BID RESPONSE DRAFT", size=18, color=C.TEXT_MUTED,
                bold=True, all_caps=True)

    # RFP name
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(24)
    _styled_run(p, rfp_name, size=22, color=C.BRAND_BLUE, bold=True)

    _spacer(doc)
    _spacer(doc)

    # Generated date
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    _styled_run(p, f"Generated: {date_str}", size=10, color=C.TEXT_MUTED, italic=True)

    # Disclaimer
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _styled_run(p, "AI-assisted first draft — review all sections before submission",
                size=9, color=C.LOW_RED, italic=True)

    doc.add_page_break()


# Matches a flag tag whether the explanation is inside the brackets
# ("[NEEDS CUSTOM INPUT: provide X]") or written as normal text after it
# ("[NEEDS CUSTOM INPUT] provide X").
_FLAG_RE = re.compile(r"\[NEEDS CUSTOM INPUT[^\]]*\]")


def _flag_explanation(match, draft_text, next_start):
    """Pull the explanation that belongs to a flag, from inside or after it."""
    inside = re.match(r"\[NEEDS CUSTOM INPUT(.*)\]", match.group(0), re.DOTALL).group(1)
    inside = inside.strip(" :—–-\t\n")
    after = draft_text[match.end():next_start].strip()
    return " ".join(part for part in (inside, after) if part).strip()


def _add_plain_paragraphs(doc, text):
    """Render ordinary draft text as justified paragraphs."""
    for line in text.split("\n"):
        if not line.strip():
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        _styled_run(p, line.strip(), size=11, color=C.TEXT_DARK)


def _add_custom_input_callout(doc, explanation):
    """
    Render a [NEEDS CUSTOM INPUT] flag as a soft-amber callout card: a clear
    bracketed tag on its own line, then the explanation as readable body text.
    """
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.75)
    _shade_cell(cell, FILL_FLAG_BG)
    _set_cell_border(cell, "left", FLAG_ACCENT, sz=18)

    tag_p = cell.paragraphs[0]
    tag_p.paragraph_format.space_after = Pt(2)
    _styled_run(tag_p, "⚠  [NEEDS CUSTOM INPUT]", size=11,
                color=RGBColor(0x85, 0x4F, 0x0B), bold=True)

    if explanation:
        body_p = cell.add_paragraph()
        body_p.paragraph_format.space_before = Pt(2)
        _styled_run(body_p, explanation, size=10.5, color=C.TEXT_DARK)


def _add_draft_body(doc, draft_text):
    """
    Render the draft text. Ordinary prose becomes justified paragraphs;
    each [NEEDS CUSTOM INPUT ...] flag becomes a readable amber callout card.
    """
    matches = list(_FLAG_RE.finditer(draft_text))

    if not matches:
        if draft_text.strip():
            _add_plain_paragraphs(doc, draft_text)
        else:
            _styled_run(doc.add_paragraph(), draft_text, size=11, color=C.TEXT_DARK)
        return

    # Prose before the first flag
    _add_plain_paragraphs(doc, draft_text[:matches[0].start()])

    # Each flag + the explanation that runs up to the next flag (or the end)
    for i, match in enumerate(matches):
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(draft_text)
        explanation = _flag_explanation(match, draft_text, next_start)
        _add_custom_input_callout(doc, explanation)


def _build_section(doc, section, index):
    confidence = section.get("confidence", "LOW")
    reason     = section.get("confidence_reason", "")
    requirement = section.get("requirement", "")
    text_color, fill_hex = _confidence_style(confidence)

    # Section heading (Heading 1 → shows in Word's navigation/TOC)
    heading = doc.add_heading(level=1)
    run = heading.add_run(f"Section {index + 1}")
    run.font.name = FONT
    run.font.color.rgb = C.BRAND_BLUE

    # Requirement — quoted style with a left accent bar
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    _set_paragraph_border(p, {"left": {"sz": 6, "color": FILL_MID_BLUE, "space": 8}})
    _styled_run(p, "Requirement: ", size=10, color=C.TEXT_MUTED, bold=True)
    _styled_run(p, requirement, size=10, color=C.TEXT_MUTED, italic=True)

    # Draft body
    _add_draft_body(doc, section.get("draft", ""))

    _spacer(doc)

    # Confidence footer — borderless 2-cell shaded table
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    badge_cell, reason_cell = table.rows[0].cells
    badge_cell.width = Inches(1.25)
    reason_cell.width = Inches(5.5)

    _shade_cell(badge_cell, fill_hex)
    _shade_cell(reason_cell, fill_hex)
    _set_cell_border(badge_cell, "right", "%02X%02X%02X" % (text_color[0], text_color[1], text_color[2]))

    bp = badge_cell.paragraphs[0]
    bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _styled_run(bp, confidence, size=10, color=text_color, bold=True)

    rp = reason_cell.paragraphs[0]
    _styled_run(rp, reason, size=9, color=text_color, italic=True)

    # Divider before the next section
    _spacer(doc)
    _horizontal_rule(doc)
    _spacer(doc)


def _build_action_items_page(doc, action_items):
    doc.add_page_break()

    heading = doc.add_heading(level=1)
    run = heading.add_run("Action Items — Required Before Submission")
    run.font.name = FONT
    run.font.color.rgb = C.BRAND_BLUE

    count = len(action_items)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    plural = "s" if count != 1 else ""
    _styled_run(
        p,
        f"{count} item{plural} require firm-specific information before this draft can be submitted.",
        size=11, color=C.TEXT_MUTED, italic=True,
    )

    _horizontal_rule(doc)
    _spacer(doc)

    if count == 0:
        p = doc.add_paragraph()
        _styled_run(
            p,
            "No action items — all sections were fully supported by past bid evidence.",
            size=11, color=C.HIGH_GREEN, bold=True,
        )
        return

    for item in action_items:
        p = doc.add_paragraph(style="List Number")
        _styled_run(p, item, size=11, color=C.TEXT_DARK)


def _configure_document(doc, firm_name, rfp_name):
    """Page size, margins, default font, heading style, and footer."""
    # Default (Normal) font
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(11)
    normal.font.color.rgb = C.TEXT_DARK

    # Heading 1 style
    h1 = doc.styles["Heading 1"]
    h1.font.name = FONT
    h1.font.size = Pt(16)
    h1.font.bold = True
    h1.font.color.rgb = C.BRAND_BLUE

    section = doc.sections[0]
    # US Letter
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(0.875)
    section.right_margin = Inches(0.875)

    # Footer: "Firm | RFP — DRAFT" left, page number right
    footer_p = section.footer.paragraphs[0]
    footer_p.paragraph_format.tab_stops.add_tab_stop(Inches(6.75), WD_TAB_ALIGNMENT.RIGHT)
    _set_paragraph_border(footer_p, {"top": {"sz": 4, "color": FILL_MID_BLUE, "space": 4}})
    _styled_run(footer_p, f"{firm_name}  |  {rfp_name}  —  DRAFT", size=8, color=C.TEXT_MUTED)
    _styled_run(footer_p, "\t", size=8, color=C.TEXT_MUTED)
    _add_page_number_field(footer_p)


def extract_action_items(draft_sections: list) -> list:
    """
    Scans all draft sections for [NEEDS CUSTOM INPUT] flags and returns
    them as a flat list of action item strings.

    Args:
        draft_sections: List of dicts from response_drafter.draft_section()
                        Each has "requirement", "draft", "confidence", "confidence_reason"

    Returns:
        List of strings — one per [NEEDS CUSTOM INPUT] block found
    """
    action_items = []

    for i, section in enumerate(draft_sections):
        draft_text = section.get("draft", "")
        section_label = f"Section {i + 1}"

        pattern = r"\[NEEDS CUSTOM INPUT[^\]]*\]"
        matches = re.findall(pattern, draft_text, re.DOTALL)

        for match in matches:
            display = match[:150] + "..." if len(match) > 150 else match
            action_items.append(f"{section_label}: {display}")

        if "[NEEDS CUSTOM INPUT" in draft_text and not matches:
            action_items.append(f"{section_label}: Custom input required — see draft body")

    return action_items


def export_to_word(
    rfp_name: str,
    draft_sections: list,
    output_path: str = None,
    firm_name: str = "[FIRM NAME — UPDATE BEFORE SENDING]",
) -> tuple:
    """
    Generates a formatted Word document from bid draft sections.

    Args:
        rfp_name:       Display name of the RFP (appears on cover page)
        draft_sections: List of dicts from response_drafter.draft_section()
        output_path:    Where to save the .docx — defaults to rfp_draft_YYYY-MM-DD.docx
        firm_name:      Construction firm name for the cover page and footer

    Returns:
        (output_path, None) on success
        (None, error_message) on failure
    """
    if output_path is None:
        today = date.today().isoformat()
        safe_name = re.sub(r"[^\w\-]", "_", rfp_name)[:40]
        output_path = f"rfp_draft_{safe_name}_{today}.docx"

    try:
        action_items = extract_action_items(draft_sections)

        today = date.today()
        date_str = f"{today:%B} {today.day}, {today.year}"

        doc = Document()
        _configure_document(doc, firm_name, rfp_name)

        _build_cover_page(doc, rfp_name, firm_name, date_str)
        for i, section in enumerate(draft_sections):
            _build_section(doc, section, i)
        _build_action_items_page(doc, action_items)

        doc.save(output_path)

        if not os.path.exists(output_path):
            return None, f"[EXPORT] Document was generated but not found at: {output_path}"

        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"  [EXPORT] ✓ Document saved: {output_path} ({file_size_kb:.1f} KB)")
        print(f"  [EXPORT] ✓ Action items extracted: {len(action_items)}")

        return output_path, None

    except Exception as e:
        return None, f"[EXPORT] Failed to generate Word document: {e}"


# ─── TEST HARNESS ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("EXPORTER TEST — generating sample Word document")
    print("=" * 60)

    test_sections = [
        {
            "requirement": "Demonstrate your firm's experience managing safety programs on construction sites with concurrent trades. Provide evidence of COR certification and TRIR over the past three years.",
            "draft": "Our firm holds an active COR certification through the BC Construction Safety Alliance, renewed in 2023.\n\nWe have direct experience managing site safety across complex multi-trade environments. On our Langley commercial project, we coordinated six concurrent trade contractors over an 18-month period, achieving zero lost-time incidents.\n\n[NEEDS CUSTOM INPUT] The RFP requests a year-by-year TRIR breakdown for the past three years. Please provide individual annual TRIR figures (2022, 2023, 2024) to satisfy this requirement.",
            "confidence": "MEDIUM",
            "confidence_reason": "Excerpts support COR certification and multi-trade experience, but year-by-year TRIR breakdown is absent.",
        },
        {
            "requirement": "Demonstrate experience with Indigenous community consultation and engagement, including a track record of building long-term relationships with First Nations communities in BC.",
            "draft": "Our provided project excerpts do not contain direct evidence of Indigenous community consultation.\n\n[NEEDS CUSTOM INPUT] To respond credibly, provide: named project examples with First Nations band councils, description of consultation processes, any Impact Benefit Agreements, or cultural competency training completed by your team.",
            "confidence": "LOW",
            "confidence_reason": "No relevant excerpts — municipal coordination does not satisfy Indigenous engagement requirements.",
        },
        {
            "requirement": "Provide evidence of your firm's experience delivering projects on accelerated timelines, including examples where you maintained quality while compressing schedule.",
            "draft": "Our firm has delivered multiple projects under compressed schedules while maintaining BC Building Code compliance and client quality expectations.\n\nOn the Surrey mixed-use project, a late-stage design change required us to compress the structural framing phase by three weeks. We achieved this through concurrent trade sequencing — beginning mechanical rough-in on floors 1–2 while framing continued on floors 3–4 — without incurring deficiencies at the final inspection.\n\nOur Burnaby residential project similarly required a four-week compression after permit delays. We recovered schedule through extended work hours in the first two weeks and pre-fabrication of bathroom pods off-site, delivering on the original handover date.",
            "confidence": "HIGH",
            "confidence_reason": "Two excerpts contain specific schedule compression examples with named projects, strategies used, and outcomes achieved.",
        },
    ]

    output, error = export_to_word(
        rfp_name="Vienna House RFP — Test Export",
        draft_sections=test_sections,
        output_path="test_export.docx",
        firm_name="Meridian Construction Ltd.",
    )

    if error:
        print(f"\n✗ Export failed: {error}")
    else:
        print(f"\n✓ Export successful: {output}")
        print("\nAction items that should appear on final page:")
        items = extract_action_items(test_sections)
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item[:100]}...")
