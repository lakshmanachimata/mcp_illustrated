"""
MCP server: leave records for three fixed employees, persisted as JSON next to this script.

Run: python leave_mcp_server.py
Configure MCP clients with command "python" and args ["/absolute/path/leave_mcp_server.py"].

MCP tools: apply/list leaves, **revoke_leave**, and **approve_leave** (restore revoked → approved).

Data file (default): ``leave-mcp-leaves.json`` in the same directory as ``leave_mcp_server.py``.
Override with env ``LEAVE_MCP_DATA_FILE=/path/to.json``. The store is reloaded from disk at the
start of each tool call so a new MCP subprocess still sees leaves written by an earlier one.

Logging: stderr and default ``leave-mcp.log`` (logger ``leave_mcp``; lines are flushed after
each tool call). Override path with ``LEAVE_MCP_LOG_FILE``, or set ``LEAVE_MCP_NO_FILE_LOG=1``
to disable the file log handler.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Employee Leave")

# Fixed name so the same logger is used when run as ``python leave_mcp_server.py`` (__main__)
# or imported elsewhere; avoids split configuration across two logger namespaces.
_LOG = logging.getLogger("leave_mcp")

_DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "leave-mcp.log"
_DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "leave-mcp-leaves.json"


def _ensure_logging() -> None:
    """Log to stderr and, unless disabled, to ``leave-mcp.log`` beside this script."""
    if _LOG.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [leave-mcp] %(message)s")

    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    _LOG.addHandler(err)

    no_file = os.environ.get("LEAVE_MCP_NO_FILE_LOG", "").strip().lower() in ("1", "true", "yes")
    path_str = os.environ.get("LEAVE_MCP_LOG_FILE", "").strip()
    log_path = Path(path_str) if path_str else _DEFAULT_LOG_PATH

    if not no_file:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            _LOG.addHandler(fh)
        except OSError as exc:
            sys.stderr.write(f"[leave-mcp] WARNING: could not open log file {log_path!s}: {exc}\n")

    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _flush_log_handlers() -> None:
    """Push file log lines to disk promptly (long-lived stdio MCP + external tail viewers)."""
    for h in _LOG.handlers:
        try:
            h.flush()
        except OSError:
            pass


def _log_tool(tool_name: str, **params: Any) -> None:
    _ensure_logging()
    _LOG.info("tool_call %s %s", tool_name, json.dumps(params, default=str, sort_keys=True))
    _flush_log_handlers()

EMPLOYEES: dict[str, dict[str, Any]] = {
    "EMP001": {"id": "EMP001", "name": "Alice Johnson", "department": "Engineering"},
    "EMP002": {"id": "EMP002", "name": "Bob Smith", "department": "Product"},
    "EMP003": {"id": "EMP003", "name": "Carol Williams", "department": "Operations"},
}

_NAME_TO_ID = {info["name"].lower(): eid for eid, info in EMPLOYEES.items()}
_ALIAS_TO_ID = {"alice": "EMP001", "bob": "EMP002", "carol": "EMP003"}

_leaves: list[dict[str, Any]] = []


def _data_path() -> Path:
    raw = os.environ.get("LEAVE_MCP_DATA_FILE", "").strip()
    return Path(raw).expanduser() if raw else _DEFAULT_DATA_PATH


def _load_from_disk() -> None:
    """Replace ``_leaves`` from the JSON store (empty list if missing or invalid)."""
    global _leaves
    path = _data_path()
    if not path.is_file():
        _leaves = []
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("root must be a JSON array")
        _leaves = data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _ensure_logging()
        _LOG.warning("leave store load failed (%s): %s", path, exc)
        _flush_log_handlers()
        _leaves = []


def _save_to_disk() -> None:
    path = _data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(_leaves, indent=2, ensure_ascii=False)
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_date(value: str) -> date:
    value = value.strip()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Date must be ISO YYYY-MM-DD") from exc


def resolve_employee_id(raw: str) -> str | None:
    key = raw.strip()
    if not key:
        return None
    upper = key.upper()
    if upper in EMPLOYEES:
        return upper
    lower = key.lower()
    m = re.fullmatch(r"emp0*([123])", lower)
    if m:
        return f"EMP00{int(m.group(1))}"
    if lower in _ALIAS_TO_ID:
        return _ALIAS_TO_ID[lower]
    if lower in _NAME_TO_ID:
        return _NAME_TO_ID[lower]
    for eid, info in EMPLOYEES.items():
        if info["name"].lower() == lower:
            return eid
    return None


def _employee_ids_from_list_input(employee_ids_csv: str) -> tuple[list[str], list[str]]:
    """
    Parse comma/semicolon-separated chunks; each chunk is resolved as a full id/name first.
    If a chunk does not resolve and contains spaces, fall back to splitting on whitespace
    (so \"EMP001 EMP002\" works in one chunk).
    """
    raw = employee_ids_csv.strip()
    if not raw:
        return [], []
    chunks = [c.strip() for c in re.split(r"[,;]+", raw) if c.strip()]
    resolved: list[str] = []
    unknown: list[str] = []

    def add_eid(eid: str) -> None:
        if eid not in resolved:
            resolved.append(eid)

    for chunk in chunks:
        eid = resolve_employee_id(chunk)
        if eid:
            add_eid(eid)
            continue
        if " " in chunk:
            for word in chunk.split():
                w = word.strip()
                if not w:
                    continue
                eid_w = resolve_employee_id(w)
                if eid_w:
                    add_eid(eid_w)
                else:
                    unknown.append(w)
        else:
            unknown.append(chunk)
    return resolved, unknown


def _leaves_for_employee(employee_id: str) -> list[dict[str, Any]]:
    return [lv for lv in _leaves if lv["employee_id"] == employee_id]


def _format_leave_line(lv: dict[str, Any]) -> str:
    extra = ""
    if lv.get("status") == "revoked" and lv.get("revoked_at"):
        extra = f" revoked_at={lv['revoked_at']}"
    return (
        f"- [{lv['id']}] {lv['start_date']} → {lv['end_date']} "
        f"type={lv['leave_type']} status={lv['status']}"
        + extra
        + (f" notes={lv['notes']}" if lv.get("notes") else "")
    )


def _format_employee_header(eid: str) -> str:
    info = EMPLOYEES[eid]
    return f"### {info['name']} ({eid}) — {info['department']}"


@mcp.tool()
def get_leaves_for_employee(employee_id: str) -> str:
    """Return leave records for one employee as JSON (machine-friendly)."""
    _log_tool("get_leaves_for_employee", employee_id=employee_id)
    _load_from_disk()
    eid = resolve_employee_id(employee_id)
    if not eid:
        return json.dumps(
            {"error": "unknown_employee", "employee_id": employee_id, "known": list(EMPLOYEES.keys())}
        )
    rows = _leaves_for_employee(eid)
    return json.dumps({"employee": EMPLOYEES[eid], "leaves": rows}, indent=2)


@mcp.tool()
def show_leaves_for_employee(employee_id: str) -> str:
    """Human-readable summary of one employee's leaves."""
    _log_tool("show_leaves_for_employee", employee_id=employee_id)
    _load_from_disk()
    eid = resolve_employee_id(employee_id)
    if not eid:
        return (
            f"Unknown employee `{employee_id}`. "
            f"Use ids {list(EMPLOYEES.keys())}, first names alice/bob/carol, or full names."
        )
    lines = [_format_employee_header(eid), ""]
    rows = _leaves_for_employee(eid)
    if not rows:
        lines.append("_No leave records._")
    else:
        lines.extend(_format_leave_line(lv) for lv in sorted(rows, key=lambda x: x["start_date"]))
    return "\n".join(lines)


@mcp.tool()
def apply_leave_for_employee(
    employee_id: str,
    start_date: str,
    end_date: str,
    leave_type: str = "general",
    notes: str = "",
) -> str:
    """Apply (create) a leave for an employee. Dates are inclusive, ISO YYYY-MM-DD."""
    _log_tool(
        "apply_leave_for_employee",
        employee_id=employee_id,
        start_date=start_date,
        end_date=end_date,
        leave_type=leave_type,
        notes=notes,
    )
    _load_from_disk()
    eid = resolve_employee_id(employee_id)
    if not eid:
        return json.dumps({"ok": False, "error": "unknown_employee", "employee_id": employee_id})
    sd = _parse_date(start_date)
    ed = _parse_date(end_date)
    if ed < sd:
        return json.dumps({"ok": False, "error": "end_before_start", "start_date": start_date, "end_date": end_date})
    leave = {
        "id": str(uuid.uuid4()),
        "employee_id": eid,
        "start_date": sd.isoformat(),
        "end_date": ed.isoformat(),
        "leave_type": leave_type.strip() or "general",
        "status": "approved",
        "notes": notes.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _leaves.append(leave)
    _save_to_disk()
    return json.dumps({"ok": True, "leave": leave, "employee": EMPLOYEES[eid]}, indent=2)


@mcp.tool()
def revoke_leave(leave_id: str, employee_id: str = "") -> str:
    """
    Revoke a leave by its ``id`` (from apply_leave / get_leaves). Sets ``status`` to ``revoked``.
    Optionally pass ``employee_id`` (emp1, alice, EMP001, …) to ensure the leave belongs to that person.
    """
    _log_tool("revoke_leave", leave_id=leave_id, employee_id=employee_id)
    _load_from_disk()
    lid = leave_id.strip()
    if not lid:
        return json.dumps({"ok": False, "error": "missing_leave_id"})
    expected_eid: str | None = None
    raw_emp = employee_id.strip()
    if raw_emp:
        expected_eid = resolve_employee_id(raw_emp)
        if not expected_eid:
            return json.dumps({"ok": False, "error": "unknown_employee", "employee_id": employee_id})
    for lv in _leaves:
        if lv.get("id") != lid:
            continue
        if expected_eid and lv.get("employee_id") != expected_eid:
            return json.dumps(
                {"ok": False, "error": "leave_employee_mismatch", "leave_id": lid, "expected_employee": expected_eid},
                indent=2,
            )
        if lv.get("status") == "revoked":
            return json.dumps({"ok": False, "error": "already_revoked", "leave": lv}, indent=2)
        lv["status"] = "revoked"
        lv["revoked_at"] = datetime.now().isoformat(timespec="seconds")
        _save_to_disk()
        eid = lv.get("employee_id", "")
        emp = EMPLOYEES.get(eid) if isinstance(eid, str) else None
        return json.dumps({"ok": True, "leave": lv, "employee": emp}, indent=2)
    return json.dumps({"ok": False, "error": "leave_not_found", "leave_id": lid})


@mcp.tool()
def approve_leave(leave_id: str, employee_id: str = "") -> str:
    """
    Set a leave back to ``approved`` (e.g. after revoke). Pass ``leave_id``; optional ``employee_id``
    (emp1, alice, EMP001, …) must match the leave's employee.
    """
    _log_tool("approve_leave", leave_id=leave_id, employee_id=employee_id)
    _load_from_disk()
    lid = leave_id.strip()
    if not lid:
        return json.dumps({"ok": False, "error": "missing_leave_id"})
    expected_eid: str | None = None
    raw_emp = employee_id.strip()
    if raw_emp:
        expected_eid = resolve_employee_id(raw_emp)
        if not expected_eid:
            return json.dumps({"ok": False, "error": "unknown_employee", "employee_id": employee_id})
    for lv in _leaves:
        if lv.get("id") != lid:
            continue
        if expected_eid and lv.get("employee_id") != expected_eid:
            return json.dumps(
                {"ok": False, "error": "leave_employee_mismatch", "leave_id": lid, "expected_employee": expected_eid},
                indent=2,
            )
        if lv.get("status") == "approved":
            return json.dumps({"ok": False, "error": "already_approved", "leave": lv}, indent=2)
        lv["status"] = "approved"
        lv.pop("revoked_at", None)
        _save_to_disk()
        eid = lv.get("employee_id", "")
        emp = EMPLOYEES.get(eid) if isinstance(eid, str) else None
        return json.dumps({"ok": True, "leave": lv, "employee": emp}, indent=2)
    return json.dumps({"ok": False, "error": "leave_not_found", "leave_id": lid})


@mcp.tool()
def show_leaves_all_employees() -> str:
    """Human-readable leaves for every employee (including those with none)."""
    _log_tool("show_leaves_all_employees")
    _load_from_disk()
    blocks: list[str] = ["## All employees — leaves", ""]
    for eid in sorted(EMPLOYEES.keys()):
        blocks.append(_format_employee_header(eid))
        blocks.append("")
        rows = _leaves_for_employee(eid)
        if not rows:
            blocks.append("_No leave records._")
        else:
            blocks.extend(_format_leave_line(lv) for lv in sorted(rows, key=lambda x: x["start_date"]))
        blocks.append("")
    return "\n".join(blocks).strip()


@mcp.tool()
def show_leaves_multiple_employees(employee_ids_csv: str) -> str:
    """
    Human-readable leaves for several employees.
    Pass comma- or space-separated ids, aliases, or names (e.g. EMP001, bob, Carol Williams).
    """
    _log_tool("show_leaves_multiple_employees", employee_ids_csv=employee_ids_csv)
    _load_from_disk()
    resolved, unknown = _employee_ids_from_list_input(employee_ids_csv)
    if not resolved and unknown:
        return f"Could not resolve any employees from `{employee_ids_csv}`. Unknown: {unknown}"
    blocks: list[str] = ["## Selected employees — leaves", ""]
    if unknown:
        blocks.append(f"_Skipped unknown tokens: {', '.join(unknown)}_")
        blocks.append("")
    for eid in resolved:
        blocks.append(_format_employee_header(eid))
        blocks.append("")
        rows = _leaves_for_employee(eid)
        if not rows:
            blocks.append("_No leave records._")
        else:
            blocks.extend(_format_leave_line(lv) for lv in sorted(rows, key=lambda x: x["start_date"]))
        blocks.append("")
    return "\n".join(blocks).strip()


def main() -> None:
    _ensure_logging()
    _load_from_disk()
    _LOG.info("leave_mcp_server starting (stdio MCP) leaves=%d data_file=%s", len(_leaves), _data_path())
    _flush_log_handlers()
    mcp.run()


if __name__ == "__main__":
    main()
