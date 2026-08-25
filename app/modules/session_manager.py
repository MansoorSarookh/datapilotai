"""
DataPilot AI — Session Manager
Manages persistent session state with save/load support.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional
from io import BytesIO


class SessionManager:
    """Manages persistent session state."""

    def __init__(self):
        if "_session_id" not in st.session_state:
            st.session_state._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def session_id(self) -> str:
        return st.session_state._session_id

    def save_state(self, key: str, value: Any):
        st.session_state[key] = value

    def load_state(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    def export_session(self) -> Dict:
        """Export serializable session state as a dict."""
        export = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "version": "3.0",
        }

        # Export chat history
        if "chat_history" in st.session_state:
            export["chat_history"] = st.session_state.chat_history

        # Export cleaning operations log
        if "audit_log" in st.session_state:
            export["audit_log"] = st.session_state.audit_log

        # Export ML results
        if "ml_results" in st.session_state:
            export["ml_results"] = str(st.session_state.ml_results)

        # File info
        if "file_name" in st.session_state:
            export["file_name"] = st.session_state.file_name

        # Trust score
        if "trust_score" in st.session_state and st.session_state.trust_score:
            export["trust_score"] = st.session_state.trust_score

        return export

    def export_session_json(self) -> bytes:
        """Export session as JSON bytes for download."""
        data = self.export_session()
        return json.dumps(data, indent=2, default=str).encode("utf-8")

    def import_session(self, data: Dict):
        """Import session state from a dict."""
        if "chat_history" in data:
            st.session_state.chat_history = data["chat_history"]
        if "audit_log" in data:
            st.session_state.audit_log = data["audit_log"]
        if "trust_score" in data:
            st.session_state.trust_score = data["trust_score"]

    def import_session_json(self, json_bytes: bytes):
        """Import session from JSON bytes."""
        data = json.loads(json_bytes.decode("utf-8"))
        self.import_session(data)

    def get_summary(self) -> Dict:
        """Get a summary of the current session."""
        return {
            "session_id": self.session_id,
            "has_data": st.session_state.get("df") is not None,
            "file_name": st.session_state.get("file_name", "None"),
            "chat_messages": len(st.session_state.get("chat_history", [])),
            "cleaning_ops": len(st.session_state.get("audit_log", [])),
            "has_cleaned_df": st.session_state.get("cleaned_df") is not None,
        }


def get_session_manager() -> SessionManager:
    """Get or create the singleton SessionManager."""
    return SessionManager()
