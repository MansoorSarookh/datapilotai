"""
DataPilot AI — Cleaning Audit Trail
Logs every cleaning operation with timestamp, type, columns affected, rows changed.
Supports undo via DataFrame snapshot history.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class AuditTrail:
    """Manages cleaning operation history with undo support."""

    MAX_SNAPSHOTS = 10

    def __init__(self):
        if "audit_log" not in st.session_state:
            st.session_state.audit_log = []
        if "df_snapshots" not in st.session_state:
            st.session_state.df_snapshots = []

    @property
    def log(self) -> List[Dict]:
        return st.session_state.audit_log

    @property
    def snapshots(self) -> List[pd.DataFrame]:
        return st.session_state.df_snapshots

    def save_snapshot(self, df: pd.DataFrame):
        """Save a DataFrame snapshot for undo support."""
        if len(self.snapshots) >= self.MAX_SNAPSHOTS:
            st.session_state.df_snapshots.pop(0)
        st.session_state.df_snapshots.append(df.copy())

    def record(
        self,
        operation: str,
        columns: Optional[List[str]] = None,
        rows_before: int = 0,
        rows_after: int = 0,
        cols_before: int = 0,
        cols_after: int = 0,
        details: str = "",
    ):
        """Record a cleaning operation in the audit log."""
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "operation": operation,
            "columns": columns or [],
            "rows_changed": abs(rows_before - rows_after),
            "rows_before": rows_before,
            "rows_after": rows_after,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "details": details,
        }
        st.session_state.audit_log.append(entry)

    def undo(self) -> Optional[pd.DataFrame]:
        """Revert to the last DataFrame snapshot. Returns the restored DataFrame or None."""
        if len(self.snapshots) > 0:
            restored = st.session_state.df_snapshots.pop()
            st.session_state.audit_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "operation": "↩️ Undo",
                "columns": [],
                "rows_changed": 0,
                "rows_before": 0,
                "rows_after": len(restored),
                "cols_before": 0,
                "cols_after": len(restored.columns),
                "details": "Reverted to previous snapshot",
            })
            return restored
        return None

    def can_undo(self) -> bool:
        return len(self.snapshots) > 0

    def clear(self):
        """Clear all audit history."""
        st.session_state.audit_log = []
        st.session_state.df_snapshots = []

    def export_log(self) -> pd.DataFrame:
        """Export audit log as a DataFrame."""
        if not self.log:
            return pd.DataFrame()
        return pd.DataFrame(self.log)

    def get_summary(self) -> Dict:
        """Get summary statistics of all cleaning operations."""
        if not self.log:
            return {"total_operations": 0}

        operations = [e["operation"] for e in self.log if e["operation"] != "↩️ Undo"]
        total_rows_changed = sum(e["rows_changed"] for e in self.log)

        return {
            "total_operations": len(operations),
            "total_rows_affected": total_rows_changed,
            "operations_list": operations,
            "first_operation": self.log[0]["timestamp"] if self.log else "",
            "last_operation": self.log[-1]["timestamp"] if self.log else "",
        }


def get_audit_trail() -> AuditTrail:
    """Get or create the singleton AuditTrail instance."""
    return AuditTrail()
