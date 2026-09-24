"""
rfp_app.py — Streamlit UI for the RFP Response Assistant
"""

import os
import re
import tempfile
import streamlit as st


# ── Page config ────────────────────────────────────────────────────────────────
# Must be the first Streamlit command on the page.
st.set_page_config(
    page_title="RFP Response Assistant",
    page_icon="🏗️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Secrets → environment bridge ────────────────────────────────────────────────
# The backend modules (rfp_parser, bid_memory, response_drafter) read their API
# keys with os.getenv(...) and, in some cases, construct their API clients at
# IMPORT time. Locally those keys come from a .env file via python-dotenv. On
# Streamlit Cloud there is no .env — keys live in the Streamlit secrets manager.
# Copy them into os.environ here, BEFORE importing the backend, so the same
# os.getenv(...) code works identically in both places.
for _secret_key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
    try:
        if _secret_key in st.secrets and not os.environ.get(_secret_key):
            os.environ[_secret_key] = str(st.secrets[_secret_key])
    except Exception:
        # st.secrets raises if no secrets file/manager is configured at all.
        # That's fine — load_dotenv() in the backend will handle the local case.
        pass


# ── Backend imports ────────────────────────────────────────────────────────────
# Imported after the secrets bridge so modules that build API clients on import
# (e.g. bid_memory's Voyage client, response_drafter's Anthropic client) find
# their keys already present in the environment.
try:
    from rfp_parser import extract_text, parse_rfp
    from bid_memory import ingest_bids, find_relevant_chunks
    from response_drafter import draft_section
    from exporter import export_to_word
    from session import save_session, load_session, list_sessions
    BACKEND_LOADED = True
    BACKEND_ERROR = ""
except ImportError as e:
    BACKEND_LOADED = False
    BACKEND_ERROR = str(e)


# ── Access gate ────────────────────────────────────────────────────────────────
# Must run before any other content so unauthorised users see nothing else.
if "client_id" not in st.session_state:
    st.session_state.client_id = None

if st.session_state.client_id is None:
    st.markdown(
        "<div style='max-width:360px;margin:80px auto 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown("### 🏗️ RFP Response Assistant")
    code_input = st.text_input("Enter your access code", type="password")
    if st.button("Continue", use_container_width=True):
        clients = st.secrets.get("clients", {})
        matched = next((name for name, code in clients.items() if code == code_input), None)
        if matched:
            st.session_state.client_id = matched
            st.rerun()
        else:
            st.error("Access code not recognised. Please check your code and try again.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

client_id = st.session_state.client_id


# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in [
    ("bids_ingested_count", 0),
    ("bids_ingested", False),
    ("parsed_rfp", None),
    ("draft_sections", []),
    ("rfp_filename", ""),
    ("word_bytes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #ffffff; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    font-family: monospace;
}
.badge-HIGH   { background: #d1fae5; color: #065f46; }
.badge-MEDIUM { background: #fef3c7; color: #92400e; }
.badge-LOW    { background: #fee2e2; color: #991b1b; }

/* Summary card */
.summary-card {
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 18px 22px;
    background: #f0f4ff;
    margin-bottom: 16px;
}
.summary-card h3 { margin: 0 0 8px 0; font-size: 18px; color: #111827; }
.summary-card p  { margin: 3px 0; color: #374151; font-size: 14px; }

/* Compliance table status cells */
.status-ok  { color: #065f46; font-weight: 600; }
.status-gap { color: #991b1b; font-weight: 600; }

details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Helper: plain-English error messages ──────────────────────────────────────
def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "AuthenticationError" in type(exc).__name__ or "api_key" in msg.lower():
        return "Your Anthropic API key was rejected. Check that ANTHROPIC_API_KEY is set correctly in your .env file."
    if "RateLimitError" in type(exc).__name__ or "rate limit" in msg.lower():
        return "The API rate limit was hit. Wait a minute and try again, or reduce the number of requirements."
    if "FileNotFoundError" in type(exc).__name__:
        return f"A required file could not be found: {msg}"
    if "node" in msg.lower() or "exporter.js" in msg.lower():
        return "Word document generation failed. Make sure Node.js is installed and exporter.js is in the same folder."
    if "voyage" in msg.lower() or "embed" in msg.lower():
        return "The embedding service (Voyage AI) returned an error. Check that VOYAGE_API_KEY is set in your .env file."
    if "json" in msg.lower():
        return "The AI returned an unexpected response format. This sometimes happens on complex RFPs — try processing again."
    return f"Something went wrong: {msg}"


# ── Helper: word-overlap criterion matching ────────────────────────────────────
_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "for", "with",
         "that", "is", "are", "be", "on", "at", "by", "from", "its", "their"}

def _words(text: str) -> set:
    return {w for w in re.sub(r"[^\w\s]", "", text.lower()).split() if w not in _STOP and len(w) > 2}

def criterion_is_addressed(criterion_text: str, draft_sections: list) -> bool:
    criterion_words = _words(criterion_text)
    if not criterion_words:
        return False
    for section in draft_sections:
        req_words = _words(section.get("requirement", ""))
        if len(criterion_words & req_words) >= 1:
            return True
    return False


# ── Helper: render a draft with readable [NEEDS CUSTOM INPUT] callouts ──────────
# A flag's explanation may sit inside the brackets ("[NEEDS CUSTOM INPUT: do X]")
# or as normal text after them ("[NEEDS CUSTOM INPUT] do X"). Either way we show
# a clean amber callout: the bracketed tag as a label, the explanation as body.
_FLAG_RE = re.compile(r"\[NEEDS CUSTOM INPUT[^\]]*\]")

def render_draft(draft_text: str) -> None:
    matches = list(_FLAG_RE.finditer(draft_text))
    if not matches:
        st.markdown(draft_text)
        return

    pre = draft_text[:matches[0].start()].strip()
    if pre:
        st.markdown(pre)

    for i, m in enumerate(matches):
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(draft_text)
        inside = re.match(r"\[NEEDS CUSTOM INPUT(.*)\]", m.group(0), re.DOTALL).group(1).strip(" :—–-\t\n")
        after  = draft_text[m.end():next_start].strip()
        explanation = " ".join(part for part in (inside, after) if part).strip()

        body = "**⚠ \\[NEEDS CUSTOM INPUT\\]**"
        if explanation:
            body += f"\n\n{explanation}"
        st.warning(body)


# ── Sidebar — previous sessions ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💾 Saved Sessions")
    st.markdown(
        "Sessions are saved automatically each time you process an RFP. "
        "Select one below to reload your previous results without reprocessing."
    )
    st.divider()
    sessions = list_sessions(client_id) if BACKEND_LOADED else []
    if not sessions:
        st.info("No saved sessions yet. Process an RFP to create your first session.")
    else:
        selected = st.selectbox("Select a session to resume", sessions)
        if st.button("📂 Load this session", type="primary", use_container_width=True):
            data = load_session(selected, client_id)
            if data:
                drafts    = data.get("draft_sections", [])
                rfp_name  = data.get("rfp_name", "")
                parsed    = data.get("parsed_rfp")

                st.session_state.parsed_rfp     = parsed
                st.session_state.draft_sections = drafts
                st.session_state.rfp_filename   = rfp_name
                st.session_state.word_bytes     = None

                # Regenerate the Word doc from the loaded data
                try:
                    import tempfile
                    tmp_dir = tempfile.mkdtemp()
                    output_path = os.path.join(tmp_dir, "rfp_draft.docx")
                    saved_path, export_error = export_to_word(rfp_name, drafts, output_path=output_path)
                    if not export_error:
                        with open(saved_path, "rb") as wf:
                            st.session_state.word_bytes = wf.read()
                except Exception:
                    pass  # download button simply won't appear if export fails

                st.rerun()
            else:
                st.error("Could not read that session file — it may be corrupted or missing.")


# ── Header ─────────────────────────────────────────────────────────────────────
if not BACKEND_LOADED:
    st.error(
        "The app could not start because one or more backend modules failed to load.\n\n"
        f"Missing or broken module: `{BACKEND_ERROR}`\n\n"
        "Make sure `rfp_parser.py`, `bid_memory.py`, `response_drafter.py`, and `exporter.py` "
        "are in the same folder as `rfp_app.py`, then refresh the page."
    )
    st.stop()

st.title("🏗️ RFP Response Assistant")
st.markdown(
    "Upload past bid documents to build your firm's knowledge base, "
    "then process a new RFP to generate a tailored draft response."
)
st.markdown(
    "💾 **Previous sessions are saved in the sidebar** — click the **›** arrow on the top left to open it.",
    help="Sessions are saved automatically after each RFP is processed."
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UPLOAD PAST BIDS
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 · Upload Past Bids")
st.caption("PDF or TXT files — 5 to 20 documents recommended for best results.")

uploaded_bids = st.file_uploader(
    label="Drag and drop bid documents here",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="bid_uploader",
    label_visibility="collapsed",
)

if uploaded_bids:
    names = ", ".join(f.name for f in uploaded_bids)
    st.caption(f"{len(uploaded_bids)} file(s) selected — {names}")

ingest_clicked = st.button(
    "⬆ Ingest Bid Documents",
    type="primary",
    disabled=not uploaded_bids,
    use_container_width=True,
)

if ingest_clicked and uploaded_bids:
    tmp_dir = tempfile.mkdtemp()
    tmp_paths = []

    # Save all uploaded files to disk before opening the status panel
    for f in uploaded_bids:
        tmp_path = os.path.join(tmp_dir, f.name)
        with open(tmp_path, "wb") as disk_file:
            disk_file.write(f.read())
        tmp_paths.append(tmp_path)

    total_files = len(tmp_paths)

    with st.status(f"Ingesting {total_files} document(s)…", expanded=True) as status:
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        def on_doc_done(completed, total, filename):
            progress_bar.progress(completed / total)
            status_text.text(f"Processing bid {completed} of {total}: {filename}")

        try:
            result = ingest_bids(tmp_paths, client_id=client_id, progress_callback=on_doc_done)

            # Replace live text with final summary
            progress_bar.progress(1.0)
            status_text.text(
                f"Done — {result['succeeded']} document(s) indexed, "
                f"{result['total_chunks']} chunks stored"
            )

            if result["failed"]:
                for path, err in result["failed"]:
                    st.warning(f"⚠ {os.path.basename(path)} could not be ingested: {err}")

            if result["succeeded"] > 0:
                st.session_state.bids_ingested = True
                st.session_state.bids_ingested_count = result["succeeded"]
                status.update(
                    label=f"✅ {result['succeeded']} of {total_files} document(s) ingested"
                          f" — {result['total_chunks']} chunks indexed",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(label="Ingestion failed — see errors above", state="error")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            status.update(label="Ingestion failed — see error below", state="error")
            st.error(friendly_error(e))

if st.session_state.bids_ingested:
    st.success(
        f"✅ Bid library ready — "
        f"{st.session_state.bids_ingested_count} document(s) indexed"
    )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROCESS RFP
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 · Process RFP")
st.caption("Upload the RFP PDF you are responding to.")

uploaded_rfp = st.file_uploader(
    label="Upload RFP (PDF)",
    type=["pdf"],
    accept_multiple_files=False,
    key="rfp_uploader",
    label_visibility="collapsed",
)

if uploaded_rfp:
    st.caption(f"Selected: **{uploaded_rfp.name}**")

if uploaded_rfp and not st.session_state.bids_ingested:
    st.warning("Complete Step 1 first — ingest your bid documents before processing an RFP.")

process_clicked = st.button(
    "⚙ Process RFP",
    type="primary",
    disabled=(not uploaded_rfp or not st.session_state.bids_ingested),
    use_container_width=True,
)

if process_clicked and uploaded_rfp:
    st.session_state.draft_sections = []
    st.session_state.word_bytes = None
    st.session_state.rfp_filename = uploaded_rfp.name
    st.session_state.parsed_rfp = None

    tmp_dir = tempfile.mkdtemp()
    rfp_path = os.path.join(tmp_dir, uploaded_rfp.name)
    with open(rfp_path, "wb") as f:
        f.write(uploaded_rfp.read())

    st.info(
        "⏱ Processing typically takes **2–5 minutes** depending on the number of requirements. "
        "Claude reads and parses the RFP, searches your bid library for relevant experience, "
        "then drafts each section individually — please keep this tab open."
    )

    with st.status("Starting…", expanded=True) as status:
        try:
            # Step 1 — extract text
            status.update(label="Extracting text from PDF…")
            st.write("📄 Reading PDF pages…")
            raw_text = extract_text(rfp_path)

            # Step 2 — parse RFP
            status.update(label="Parsing RFP…")
            st.write("🔍 Identifying project details and requirements with Claude…")
            parsed = parse_rfp(raw_text)
            st.session_state.parsed_rfp = parsed
            rfp_name = parsed.get("project_name") or uploaded_rfp.name

            requirements = parsed.get("key_requirements", [])
            n = len(requirements)
            st.write(f"Found **{n}** requirement(s) — drafting sections…")

            # Step 3 — draft each section
            drafts = []
            for i, req in enumerate(requirements):
                status.update(label=f"Drafting section {i + 1} of {n}…")
                st.write(f"🔎 Finding relevant past bids for section {i + 1}…")
                chunks = find_relevant_chunks(req, client_id=client_id, n=3)

                st.write(f"✍ Drafting section {i + 1} of {n}…")
                result, err = draft_section(req, chunks)
                if err:
                    raise RuntimeError(err)
                drafts.append(result)

            st.session_state.draft_sections = drafts
            save_session(rfp_name, st.session_state.parsed_rfp, drafts, client_id=client_id)

            # Step 4 — export Word doc
            status.update(label="Generating Word document…")
            st.write("📝 Building Word document…")
            rfp_name = parsed.get("project_name") or uploaded_rfp.name
            output_path = os.path.join(tmp_dir, "rfp_draft.docx")
            saved_path, export_error = export_to_word(rfp_name, drafts, output_path=output_path)

            if export_error:
                raise RuntimeError(export_error)

            with open(saved_path, "rb") as word_file:
                st.session_state.word_bytes = word_file.read()

            status.update(
                label=f"✅ Done — {n} section(s) drafted",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            status.update(label="Processing stopped — see error below", state="error")
            st.error(friendly_error(e))


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.draft_sections:
    st.divider()
    st.header("Results")

    # ── Summary card ──────────────────────────────────────────────────────────
    p = st.session_state.parsed_rfp or {}
    project  = p.get("project_name") or "—"
    client   = p.get("client_org") or "—"
    deadline = p.get("submission_deadline") or "—"
    n_req    = len(p.get("key_requirements", []))
    bonding  = "Yes" if p.get("bonding_required") else "No"

    st.markdown(f"""
<div class="summary-card">
  <h3>📋 {project}</h3>
  <p><strong>Client:</strong> {client}</p>
  <p><strong>Submission deadline:</strong> {deadline}</p>
  <p><strong>Requirements identified:</strong> {n_req} &nbsp;|&nbsp; <strong>Bonding required:</strong> {bonding}</p>
</div>
""", unsafe_allow_html=True)

    # ── Confidence summary row ─────────────────────────────────────────────────
    sections = st.session_state.draft_sections
    high   = sum(1 for d in sections if d.get("confidence") == "HIGH")
    medium = sum(1 for d in sections if d.get("confidence") == "MEDIUM")
    low    = sum(1 for d in sections if d.get("confidence") == "LOW")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sections",    len(sections))
    c2.metric("🟢 High",     high)
    c3.metric("🟡 Medium",   medium)
    c4.metric("🔴 Low",      low)
    st.markdown("")

    # ── Compliance checklist ───────────────────────────────────────────────────
    criteria = p.get("evaluation_criteria", [])

    st.subheader("Compliance Checklist")
    if not criteria:
        st.caption("No scoring criteria were found in this RFP — checklist unavailable.")
    else:
        rows = []
        for item in criteria:
            criterion = item.get("criterion", "")
            weight    = item.get("weight") or "—"
            addressed = criterion_is_addressed(criterion, sections)
            rows.append({
                "Criterion": criterion,
                "Weight": weight,
                "_addressed": addressed,
            })

        # Render as an HTML table so we can colour the Status column
        header = "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
        header += "<tr style='border-bottom:2px solid #e5e7eb;text-align:left;'>"
        header += "<th style='padding:8px 12px;width:55%'>Criterion</th>"
        header += "<th style='padding:8px 12px;width:15%'>Weight</th>"
        header += "<th style='padding:8px 12px;width:30%'>Status</th></tr>"

        body = ""
        for row in rows:
            if row["_addressed"]:
                status_html = '<span class="status-ok">✔ Addressed</span>'
            else:
                status_html = '<span class="status-gap">✘ Not addressed</span>'
            body += (
                f"<tr style='border-bottom:1px solid #f3f4f6;'>"
                f"<td style='padding:8px 12px'>{row['Criterion']}</td>"
                f"<td style='padding:8px 12px'>{row['Weight']}</td>"
                f"<td style='padding:8px 12px'>{status_html}</td>"
                f"</tr>"
            )

        st.markdown(header + body + "</table>", unsafe_allow_html=True)
        st.markdown("")

        gaps = sum(1 for r in rows if not r["_addressed"])
        if gaps:
            st.warning(
                f"{gaps} criterion/criteria have no matching draft section. "
                "Review the sections below and consider whether any requirement covers these topics."
            )

    st.markdown("")

    # ── Draft sections ─────────────────────────────────────────────────────────
    st.subheader("Draft Sections")
    for i, section in enumerate(sections):
        conf   = section.get("confidence", "LOW")
        reason = section.get("confidence_reason", "")
        req    = section.get("requirement", "")

        label = f"Section {i + 1} — {req[:70]}{'…' if len(req) > 70 else ''}"
        with st.expander(label, expanded=(conf == "LOW")):
            badge_html = (
                f'<span class="badge badge-{conf}">{conf}</span>'
                f'<span style="color:#6b7280;font-size:12px;margin-left:8px;">{reason}</span>'
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown("")
            render_draft(section.get("draft", ""))


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.word_bytes:
    st.divider()
    safe_name = st.session_state.rfp_filename.replace(".pdf", "")
    st.download_button(
        label="📥 Download Word Document",
        data=st.session_state.word_bytes,
        file_name=f"rfp_draft_{safe_name}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        use_container_width=True,
    )
    st.caption(
        "Includes all drafted sections, confidence ratings, "
        "and an action items checklist for anything flagged [NEEDS CUSTOM INPUT]."
    )
