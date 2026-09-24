# session.py

import json
import os
import re
from datetime import date

# Always save sessions relative to this file's location, not the working directory.
# This means sessions are found regardless of where `streamlit run` is called from.
_SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")


def _ensure_dir():
    os.makedirs(_SESSIONS_DIR, exist_ok=True)


def save_session(rfp_name, parsed_rfp, draft_sections, client_id="default"):
    _ensure_dir()
    safe_name   = re.sub(r'[^\w\-]', '_', rfp_name)[:40]
    safe_client = re.sub(r'[^\w\-]', '_', client_id)
    today       = date.today().isoformat()
    filename    = f"{safe_client}_{safe_name}_{today}.json"
    filepath    = os.path.join(_SESSIONS_DIR, filename)

    payload = {
        "rfp_name":       rfp_name,
        "parsed_rfp":     parsed_rfp,
        "draft_sections": draft_sections,
        "saved_date":     today,
        "client_id":      client_id,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return filepath


def list_sessions(client_id="default"):
    _ensure_dir()
    safe_client = re.sub(r'[^\w\-]', '_', client_id)
    prefix = f"{safe_client}_"
    files = [f for f in os.listdir(_SESSIONS_DIR)
             if f.startswith(prefix) and f.endswith(".json")]
    return sorted(files, reverse=True)


def load_session(filename, client_id="default"):
    filepath = os.path.join(_SESSIONS_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Reject files that don't belong to this client
    if data.get("client_id", "default") != client_id:
        return None
    return data