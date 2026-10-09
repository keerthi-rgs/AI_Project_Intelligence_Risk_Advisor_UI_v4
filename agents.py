"""Rule-based project intelligence agents used by app.py (local, no external API required).

Every extracted item carries `Source` and `Location` so dashboards, reports, and the
conversational assistant can cite the exact document passage it came from.
"""
import re
from datetime import date
from typing import Any, Dict, List, Optional

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
HEADING_RE = re.compile(r"^\d+\.\s+[A-Z][^.:]*$")
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
SEVERITY_WEIGHT = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
SEVERITY_BASE = {"Critical": (0.85, 4.6), "High": (0.7, 3.9), "Medium": (0.5, 3.0), "Low": (0.25, 1.8)}
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were", "our", "has", "have", "into",
    "during", "under", "over", "due", "remains", "status", "what", "which", "who", "whom", "how", "why",
    "when", "where", "is", "of", "to", "in", "on", "by", "be", "we", "us", "any", "all", "there", "their",
}
RACI_RE = re.compile(r"^([A-Z][\w /&-]{2,40}):\s*(Accountable|Responsible|Consulted|Informed)\b\s*(.*)$")


# ─────────────────────────────────────────────────────────────
#  Shared parsing helpers
# ─────────────────────────────────────────────────────────────
def _clean(line: str) -> str:
    line = re.sub(r"^[\s\-•*·]+", "", line).strip()
    return re.sub(r"\s+", " ", line)


def _is_noise(line: str) -> bool:
    return (not line or len(line) < 8 or line.endswith(":") or line.startswith("===")
            or bool(HEADING_RE.match(line)) or line.lower().startswith(("date:", "location:", "type:")))


def _csv_fields(text: str) -> Optional[Dict[str, str]]:
    fields = {}
    for part in (p.strip() for p in text.split(" | ")):
        if ": " not in part:
            return None
        k, v = part.split(": ", 1)
        fields[k.strip()] = v.strip()
    return fields if "Category" in fields and "Description" in fields else None


def _segments(records: List[Dict[str, Any]]):
    """Yield (line, source, location, csv_fields) for every meaningful text line."""
    for rec in records:
        text, src, loc = rec.get("text", ""), rec.get("source", ""), rec.get("location", "")
        fields = _csv_fields(text)
        if fields:
            yield fields["Description"], src, loc, fields
            continue
        for raw in re.split(r"\n|(?<=[.!?])\s+(?=[-•]\s)", text):
            line = _clean(raw)
            if not _is_noise(line):
                yield line, src, loc, None


def _field(line: str, name: str) -> str:
    m = re.search(rf"{name}\s*:\s*([^|]+?)(?:\.\s+[A-Z][\w ]*:|\s*\||\.?\s*$)", line, re.I)
    return m.group(1).strip().rstrip(".") if m else ""


def _head(line: str, prefix: str) -> str:
    body = re.sub(rf"^{prefix}\s*:\s*", "", line, flags=re.I)
    return re.split(r"\.\s+(?=[A-Z][\w ]*:)|\s+\|\s+", body)[0].strip().rstrip(".")


def _norm_status(status: str, default: str = "Open") -> str:
    s = status.lower()
    if any(k in s for k in ("complet", "done", "closed")) or s == "resolved":
        return "Completed"
    if "progress" in s:
        return "In Progress"
    if "plan" in s:
        return "Planned"
    return default if not s or any(k in s for k in ("unresolved", "open", "pending")) else status.title()


def tokens(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOPWORDS}


def _similar(a: str, b: str) -> bool:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / len(ta | tb) >= 0.5 or (min(len(ta), len(tb)) >= 4 and inter / min(len(ta), len(tb)) >= 0.75)


def _dedupe(items: List[Dict[str, Any]], key: str, rank_key: Optional[str] = None, fuzzy: bool = False) -> List[Dict[str, Any]]:
    """Merge duplicate items (same fact stated in several documents), keeping the most severe.

    fuzzy=True also merges paraphrases (token overlap); otherwise only normalized-identical text merges.
    """
    def norm(v: Any) -> str:
        return re.sub(r"\W+", " ", str(v).lower()).strip()

    out: List[Dict[str, Any]] = []
    for it in items:
        dup = next((o for o in out if norm(o.get(key, "")) == norm(it.get(key, ""))
                    or (fuzzy and _similar(str(o.get(key, "")), str(it.get(key, ""))))), None)
        if dup is None:
            out.append(dict(it))
            continue
        if rank_key and SEVERITY_ORDER.index(it.get(rank_key, "Low")) < SEVERITY_ORDER.index(dup.get(rank_key, "Low")):
            dup[rank_key] = it[rank_key]
        also = {s for s in str(dup.get("Also In", "")).split(", ") if s}
        if it.get("Source") and it.get("Source") != dup.get("Source"):
            also.add(it["Source"])
        if also:
            dup["Also In"] = ", ".join(sorted(also))
    return out


def _cite(item: Dict[str, Any]) -> str:
    return f"[{item.get('Source', '?')} · {item.get('Location', '?')}]"


def _days_until(due: str) -> Optional[int]:
    if not DATE_RE.fullmatch(due or ""):
        return None
    return (date.fromisoformat(due) - date.today()).days


# ─────────────────────────────────────────────────────────────
#  Scope & Deliverable Extraction Agent
# ─────────────────────────────────────────────────────────────
class ScopeDeliverableAgent:
    """Extracts goals, milestones, deliverables, and responsibilities from any project document.

    Works in three layers: explicit "Key: value" formats, section headings (Objectives, Timeline,
    Deliverables, Roles...), and common phrasing ("aims to", "is responsible for", "due by <date>").
    """
    SECTION_KEYS = [
        ("raci", ("raci", "responsibilit", "roles", "role ", "team structure", "stakeholder", "ownership", "team members")),
        ("deliverables", ("deliverable", "outputs", "work products", "artifacts", "artefacts")),
        ("milestones", ("milestone", "timeline", "schedule", "roadmap", "phases", "delivery plan", "work plan")),
        ("goals", ("goal", "objective", "aims", "purpose", "vision", "mission", "expected outcome", "success criteria")),
        ("overview", ("overview", "introduction", "background", "problem statement", "abstract", "summary", "scope")),
    ]
    SKIP_PREFIX = ("action item", "blocker", "issue", "pending decision", "decision", "risk", "date:", "location:",
                   "type:", "attendees", "note:")
    RISK_LABEL = re.compile(r"^[^:]{0,40}\b(risk|issue|blocker|blocked|dependency|bottleneck|complexity|concern|problem)\b[^:]{0,40}:")
    MILESTONE_WORDS = re.compile(r"\b(milestone|phase|sprint|release|launch|go-live|kick-?off|sign-?off|deadline|due|complete[sd]?|delivery)\b", re.I)
    GOAL_VERB = re.compile(r"\b(build|develop|create|deliver|enable|automate|provide|reduce|improve|increase|help|"
                           r"support|design|implement|detect|ensure|achieve|aims?|objective|purpose)\b", re.I)
    GOAL_SENTENCE = re.compile(r"\b((main |primary |overall )?(goal|objective|aim|purpose)( of [\w\s]+?)? (is|are) to|"
                               r"aims? to|intended to|seeks? to)\b", re.I)
    MILESTONE_HEAD = re.compile(r"^(milestone|phase|sprint|stage|release|iteration|increment|week|wk)\s*[-#]?\s*\d+[\w.]*", re.I)
    RESPONSIBLE = re.compile(r"^(?:[Tt]he\s+)?([A-Za-z][\w ]{1,30}?\s(?:lead|manager|team|officer|owner|engineer|analyst|architect)|[A-Z][\w.&/-]*(?:\s+[A-Z&][\w.&/-]*){0,4})"
                             r"\s+(?:is|are|will be|shall be)\s+(responsible|accountable)\s+for\s+(.+)$")
    OWNS = re.compile(r"^([A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3})\s+(?:owns|leads|will own|will lead|is in charge of)\s+(.+)$")
    OWNER_RE = re.compile(r"\(\s*(?:owner|lead|responsible|by|assigned to)\s*[:\-]?\s*([^)]+?)\s*\)|"
                          r"\b(?:owner|assigned to|responsible|lead|assignee)\s*[:\-]\s*([A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,3})", re.I)
    NAME_KEYS = ("description", "task", "name", "title", "item", "milestone", "deliverable", "goal", "objective", "phase", "activity")
    OWNER_KEYS = ("assignee", "owner", "responsible", "assigned to", "lead", "person")
    DATE_KEYS = ("due date", "deadline", "due", "target date", "end date", "finish", "date")
    CAPS = {"goals": 15, "milestones": 20, "deliverables": 30, "responsibilities": 20}

    @staticmethod
    def _goal_category(text: str) -> str:
        t = text.lower()
        if any(k in t for k in ("reduce", "effort", "cost", "%", "efficien")):
            return "Efficiency"
        if any(k in t for k in ("risk", "detect", "gap")):
            return "Risk Management"
        if any(k in t for k in ("automat", "extract")):
            return "Automation"
        if any(k in t for k in ("user", "customer", "team", "stakeholder")):
            return "Stakeholder Value"
        return "Strategic"

    # ── helpers ──
    @staticmethod
    def _date(text: str) -> str:
        return _find_date(text)

    @staticmethod
    def _status(text: str, default: str = "Planned") -> str:
        explicit = _field(text, "Status")
        if explicit:
            return _norm_status(explicit, default)
        t = text.lower()
        if any(k in t for k in ("completed", "complete", "done", "delivered", "finished")):
            return "Completed"
        if any(k in t for k in ("in progress", "ongoing", "underway", "started")):
            return "In Progress"
        return default

    def _owner(self, text: str) -> str:
        m = self.OWNER_RE.search(text)
        return (m.group(1) or m.group(2)).strip().rstrip(".") if m else ""

    @staticmethod
    def _name(text: str) -> str:
        text = re.sub(r"^(goal|objective(\s*\d+)?|deliverable|milestone)\s*[:\-]\s*", "", text, flags=re.I)
        text = re.split(r"\s+\|\s+|\s+\(\s*(?:owner|lead|due|by|target)|\s+[–—]\s+(?:due|by|owner)|,?\s+due\s+(?:on|by)?\s*|"
                        r"\.\s+(?:Owner|Status|Due|Target)", text, flags=re.I)[0]
        return text.strip().rstrip(".;,")

    def _section_of(self, heading: str) -> str:
        h = " " + heading.lower() + " "
        for name, keys in self.SECTION_KEYS:
            if any(k in h for k in keys):
                return name
        return "other"

    @staticmethod
    def _kv(text: str) -> Optional[Dict[str, str]]:
        parts = [p.strip() for p in text.split(" | ")]
        if len(parts) < 2 or not all(": " in p for p in parts):
            return None
        return {k.strip().lower(): v.strip() for k, v in (p.split(": ", 1) for p in parts)}

    @staticmethod
    def _pick(fields: Dict[str, str], keys) -> str:
        for k in keys:
            if fields.get(k):
                return fields[k]
        return ""

    @staticmethod
    def _lines(text: str):
        """Yield (line, is_list_item), merging PDF hard-wrapped continuation lines."""
        out: List[List[Any]] = []
        for raw in text.split("\n"):
            if not raw.strip():
                continue
            bullet = bool(re.match(r"^\s*([-•*·▪●◦]|\d+(\.\d+)*[.)]|[a-z][.)]|\(\w{1,3}\))\s+", raw))
            line = re.sub(r"^\s*([-•*·▪●◦]|\d+(\.\d+)*[.)]|[a-z][.)]|\(\w{1,3}\))\s+", "", raw).strip()
            line = re.sub(r"\s+", " ", line)
            if out and not bullet and line[:1].islower() and not out[-1][0].endswith((".", ":")):
                out[-1][0] += " " + line
                continue
            out.append([line, bullet])
        for line, bullet in out:
            yield line, bullet

    def _heading(self, line: str, bullet: bool) -> Optional[str]:
        """Return the section a heading line opens, or None if the line is content."""
        if len(line) > 90 or len(line.split()) > 10 or (line.endswith(".") and not line.endswith("etc.")):
            return None
        stripped = line.rstrip(":").strip()
        if self.MILESTONE_HEAD.match(stripped) and not bullet:
            return "milestone_item"
        section = self._section_of(stripped)
        if section != "other":
            return section if (not bullet or line.endswith(":") or len(stripped.split()) <= 6) else None
        if bullet:
            return None
        core = re.sub(r"\([^)]*\)", "", stripped).split(":")[0]
        words = [w for w in re.findall(r"[A-Za-z][\w'&-]*", core) if len(w) > 3]
        if line.endswith(":") or (words and sum(w[0].isupper() for w in words) / len(words) >= 0.6):
            return "other"
        return None

    # ── main ──
    def analyze(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        goals, milestones, deliverables, raci = [], [], [], []

        def add_goal(text, cite):
            text = self._name(text)
            if len(text) >= 12:
                goals.append({"Goal / Objective": text, "Category": self._goal_category(text), **cite})

        def add_milestone(text, cite, owner="", date_hint="", status=None, scope=""):
            name = re.split(r"\s+\|\s+(?![^()]*\))", text)[0].strip().rstrip(".:")
            name = re.sub(r"^milestone\s*:\s*", "", name, flags=re.I)
            name = re.sub(r"\s*(?:[:\-–]\s*|\s)(due|by|target)\b.*$", "", name, flags=re.I) or name
            milestones.append({"Milestone / Phase": name,
                               "Target Date": _field(text, "Target Date") or date_hint or self._date(text) or "TBD",
                               "Status": status or self._status(text), "Scope": scope or _field(text, "Scope"),
                               "Owner": owner or self._owner(text), **cite})

        def add_deliverable(text, cite, owner="", date_hint="", status=None, milestone=""):
            name = self._name(text)
            if len(name) < 6:
                return
            item = {"Deliverable": name, "Owner": owner or self._owner(text) or "Unassigned",
                    "Target Completion": _field(text, "Target Completion") or self._date(text) or date_hint or "TBD",
                    "Status": status or self._status(text), **cite}
            if milestone:
                item["Milestone"] = milestone
            deliverables.append(item)

        def add_raci(role, kind, resp, cite):
            role, resp = role.strip().strip("*:"), resp.strip().rstrip(".")
            if 2 <= len(role) <= 50 and resp:
                raci.append({"Role": role, "RACI": kind.title(), "Responsibility": resp, **cite})

        for rec in records:
            text, src, loc = rec.get("text", ""), rec.get("source", ""), rec.get("location", "")
            cite = {"Source": src, "Location": loc}

            fields = self._kv(text) if "\n" not in text.strip() else None
            if fields:
                self._from_row(fields, cite, add_goal, add_milestone, add_deliverable, add_raci)
                continue
            if str(loc).startswith("Table") and self._from_table(text, cite, add_milestone, add_deliverable, add_raci):
                continue

            section, parent = _SECTION_STATE.get(src, ("other", None))
            for line, bullet in self._lines(text):
                low = line.lower()
                head = self._heading(line, bullet)
                if head == "milestone_item":
                    add_milestone(line, cite)
                    section, parent = "milestone_tasks", milestones[-1]
                    continue
                if head:
                    section, parent = head, None
                    continue
                if len(line) < 8 or low.startswith(self.SKIP_PREFIX) or self.RISK_LABEL.match(low):
                    continue

                # 1. explicit formats
                if re.match(r"^(goal|objective(\s*\d+)?)\s*[:\-]", low):
                    add_goal(line, cite)
                elif low.startswith("deliverable:"):
                    add_deliverable(_head(line, "Deliverable"), cite, owner=_field(line, "Owner"),
                                    status=_norm_status(_field(line, "Status"), "Planned"))
                elif RACI_RE.match(line):
                    m = RACI_RE.match(line)
                    add_raci(m.group(1), m.group(2), m.group(3), cite)
                elif self.RESPONSIBLE.match(line):
                    m = self.RESPONSIBLE.match(line)
                    add_raci(m.group(1), m.group(2), m.group(3), cite)
                elif self.OWNS.match(line) and section in ("raci", "deliverables"):
                    m = self.OWNS.match(line)
                    add_raci(m.group(1), "Responsible", m.group(2), cite)
                elif self.MILESTONE_HEAD.match(line) and (self._date(line) or ":" in line or "|" in line):
                    add_milestone(line, cite)
                elif "milestone" in low and self._date(line) and not low.startswith("goal"):
                    add_milestone(line, cite)
                # 2. section context
                elif section == "goals":
                    add_goal(line, cite)
                elif section == "milestones":
                    if self._date(line) or self.MILESTONE_WORDS.search(line):
                        add_milestone(line, cite)
                elif section == "milestone_tasks" and parent is not None:
                    add_deliverable(line, cite, date_hint=parent["Target Date"] if parent["Target Date"] != "TBD" else "",
                                    status=None, milestone=parent["Milestone / Phase"])
                elif section == "deliverables":
                    add_deliverable(line, cite)
                elif section == "raci":
                    m = re.match(r"^([A-Z][\w .&/()-]{1,40}?)\s*(?::|[–—-]\s)\s*(.+)$", line)
                    if m:
                        role, resp = m.group(1), m.group(2)
                        kind = next((k for k in ("Accountable", "Responsible", "Consulted", "Informed") if k.lower() in resp.lower()[:20]), "Responsible")
                        add_raci(role, kind, re.sub(r"^(accountable|responsible|consulted|informed)\s*(for|on)?\s*", "", resp, flags=re.I) or resp, cite)
                # 3. phrasing anywhere
                elif self.GOAL_SENTENCE.search(line):
                    add_goal(line, cite)
                elif section == "overview" and self.GOAL_VERB.search(line) and len(line) < 220:
                    add_goal(line, cite)
                elif re.search(r"\bdeliverables?\s+(include|are|will be)\b", low):
                    body = re.split(r"\bdeliverables?\s+(?:include|are|will be)\s*:?\s*", line, flags=re.I)[-1]
                    for part in re.split(r",\s*(?:and\s+)?|;\s*|\s+and\s+", body):
                        add_deliverable(part, cite)
            _SECTION_STATE[src] = (section, parent)
        _SECTION_STATE.clear()

        if not raci:
            owners: Dict[str, List[Dict[str, Any]]] = {}
            for d in deliverables + milestones:
                o = d.get("Owner") or ""
                if o and o not in ("Unassigned", "TBD"):
                    owners.setdefault(o, []).append(d)
            for o, items in owners.items():
                names = [i.get("Deliverable") or i.get("Milestone / Phase") for i in items]
                raci.append({"Role": o, "RACI": "Responsible", "Responsibility": "; ".join(names[:4]),
                             "Source": items[0]["Source"], "Location": items[0]["Location"]})

        result = {
            "goals": _dedupe(goals, "Goal / Objective"),
            "milestones": _dedupe(milestones, "Milestone / Phase"),
            "deliverables": _dedupe(deliverables, "Deliverable"),
            "responsibilities": _dedupe(raci, "Role"),
        }
        return {k: v[: self.CAPS[k]] for k, v in result.items()}

    def _from_row(self, f, cite, add_goal, add_milestone, add_deliverable, add_raci):
        cat = self._pick(f, ("category", "type", "item type", "kind")).lower()
        name = self._pick(f, self.NAME_KEYS)
        owner, due = self._pick(f, self.OWNER_KEYS), self._pick(f, self.DATE_KEYS)
        status = _norm_status(f.get("status", ""), "Planned") if f.get("status") else None
        if "role" in f and any(k.startswith("responsib") for k in f):
            resp = next(v for k, v in f.items() if k.startswith("responsib"))
            role = f["role"] + (f" ({f['name']})" if f.get("name") else "")
            add_raci(role, f.get("raci", "Responsible") or "Responsible", resp, cite)
        elif "milestone" in cat or "phase" in cat or (not cat and f.get("milestone")):
            add_milestone(f.get("milestone") or name, cite, owner=owner, date_hint=due, status=status,
                          scope=f.get("notes", "") or f.get("scope", ""))
        elif "deliverable" in cat or (not cat and f.get("deliverable")):
            add_deliverable(f.get("deliverable") or name, cite, owner=owner, date_hint=due, status=status)
        elif "goal" in cat or "objective" in cat or (not cat and (f.get("goal") or f.get("objective"))):
            add_goal(f.get("goal") or f.get("objective") or name, cite)

    def _from_table(self, text, cite, add_milestone, add_deliverable, add_raci) -> bool:
        rows = [[c.strip() for c in r.split(" | ")] for r in text.split("\n") if r.strip()]
        if len(rows) < 2:
            return False
        hdr = [h.lower() for h in rows[0]]

        def col(*keys):
            return next((i for i, h in enumerate(hdr) if any(k in h for k in keys)), None)

        role_c, resp_c, name_c = col("role"), col("responsib", "duties", "tasks"), col("name", "member", "person")
        ms_c, del_c = col("milestone", "phase", "sprint", "stage"), col("deliverable", "output")
        owner_c, date_c, status_c = col(*self.OWNER_KEYS), col(*self.DATE_KEYS, "week"), col("status")
        get = lambda r, i: r[i] if i is not None and i < len(r) else ""
        if role_c is not None and (resp_c is not None or name_c is not None):
            for r in rows[1:]:
                role = get(r, role_c) + (f" ({get(r, name_c)})" if name_c is not None and get(r, name_c) else "")
                add_raci(role, "Responsible", get(r, resp_c) or get(r, role_c), cite)
            return True
        if ms_c is not None or del_c is not None:
            for r in rows[1:]:
                status = _norm_status(get(r, status_c), "Planned") if get(r, status_c) else None
                if del_c is not None and get(r, del_c):
                    add_deliverable(get(r, del_c), cite, owner=get(r, owner_c), date_hint=get(r, date_c), status=status,
                                    milestone=get(r, ms_c))
                elif get(r, ms_c):
                    add_milestone(get(r, ms_c), cite, owner=get(r, owner_c), date_hint=get(r, date_c), status=status)
            return True
        return False


_SECTION_STATE: Dict[str, Any] = {}
_BLK_STATE: Dict[str, str] = {}
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DATE_PATTERNS = [
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.](?:20)?\d{2}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+20\d{{2}}\b", re.I),
    re.compile(rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+20\d{{2}}\b", re.I),
    re.compile(r"\bweeks?\s*\d+(?:\s*[-–]\s*\d+)?\b", re.I),
    re.compile(r"\bQ[1-4]\s*20\d{2}\b"),
    re.compile(rf"\b{_MONTH}\s+20\d{{2}}\b", re.I),
]


def _find_date(text: str) -> str:
    for pat in _DATE_PATTERNS:
        m = pat.search(text or "")
        if m:
            return m.group(0)
    return ""


# ─────────────────────────────────────────────────────────────
#  Risk Detection & Delivery Forecasting Agent
# ─────────────────────────────────────────────────────────────
class RiskForecastingAgent:
    CATEGORIES = [
        ("Resource", ("understaffed", "shortage", "resource", "capacity", "bandwidth")),
        ("Budget", ("budget", "overrun", "cost")),
        ("Schedule", ("slippage", "delay", "behind schedule", "late ", "overdue", "deadline")),
        ("Security & Compliance", ("security", "compliance", "oauth", "token", "encryption")),
        ("Technical", ("latency", "fail", "bug", "error", "complexity", "migration", "crash", "spike")),
        ("Dependency", ("vendor", "third-party", "dependency", "waiting for", "blocked by", "prerequisite")),
    ]
    MITIGATION = {
        "Schedule": "Re-baseline the milestone plan, protect the critical path, and track slippage weekly.",
        "Resource": "Rebalance workload or request temporary staff/infrastructure for the affected sprint.",
        "Budget": "Escalate the overrun to the sponsor with a cost/benefit justification and fallback option.",
        "Security & Compliance": "Schedule a security review and add regression tests before release.",
        "Technical": "Assign an owner to root-cause the defect and add automated tests to prevent recurrence.",
        "Dependency": "Escalate to the external owner, agree a committed date, and prepare a workaround.",
    }
    DEFAULT_OWNER = {
        "Schedule": "Project Manager", "Resource": "Project Manager", "Budget": "Product Sponsor",
        "Security & Compliance": "Security Officer", "Technical": "Lead Architect", "Dependency": "Project Manager",
    }
    DEP_KEYS = ("dependency", "waiting for", "blocked by", "prerequisite", "depends on", "awaiting", "sign-off")
    EXCLUDE_PREFIX = ("action item", "goal", "objective", "deliverable", "milestone phase")

    def _category(self, text: str) -> Optional[str]:
        t = text.lower()
        if t.startswith(("dependency", "blocked by", "waiting for")):
            return "Dependency"
        for cat, keys in self.CATEGORIES:
            if any(k in t for k in keys):
                return cat
        return None

    @staticmethod
    def _severity(text: str, csv: Optional[Dict[str, str]]) -> str:
        explicit = _field(text, "Severity") or (csv or {}).get("Priority", "")
        for s in SEVERITY_ORDER:
            if s.lower() in explicit.lower():
                return s
        t = text.lower()
        if any(k in t for k in ("critical", "fails", "blocking", "shortage")):
            return "Critical"
        if any(k in t for k in ("delay", "slippage", "understaffed", "behind schedule", "overrun", "rate limit")):
            return "High"
        if any(k in t for k in ("latency", "waiting", "incomplete", "silently")):
            return "Medium"
        return "Low"

    def analyze(self, records: List[Dict[str, Any]], milestones: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        milestones = milestones or []
        risks, deps, all_text = [], [], []
        for line, src, loc, csv in _segments(records):
            all_text.append(line)
            low = line.lower()
            cite = {"Source": src, "Location": loc}
            csv_cat = (csv or {}).get("Category", "").lower()

            if any(k in low for k in self.DEP_KEYS) and not low.startswith(self.EXCLUDE_PREFIX + ("pending decision",)):
                deps.append({"Dependency / Prerequisite": re.sub(r"^(dependency|blocked by)\s*:\s*", "", line, flags=re.I).rstrip("."),
                             "Type": "External" if any(k in low for k in ("vendor", "third-party", "it security")) else "Internal",
                             "Impact": "Blocks downstream milestone work", **cite})

            if csv and csv_cat != "blocker":
                continue
            if low.startswith(self.EXCLUDE_PREFIX) or RACI_RE.match(line):
                continue
            is_decision = low.startswith("pending decision")
            if is_decision and not any(k in low for k in ("budget", "overrun", "cost")):
                continue
            is_blocker = csv_cat == "blocker" or low.startswith(("blocker:", "issue:"))
            cat = self._category(line)
            if not cat:
                continue
            if cat == "Technical" and not is_blocker and not any(k in low for k in ("risk", "fail", "latency", "spike")):
                continue
            desc = re.sub(r"^(blocker|issue|pending decision|dependency|blocked by|[\w ]{0,30}risk|resource bottleneck|technical complexity)\s*:\s*",
                          "", line, flags=re.I)
            desc = re.split(r"\.\s+(?=(Status|Severity|Decision Owner)\s*:)", desc)[0].rstrip(".")
            risks.append({"Severity": self._severity(line, csv), "Risk Category": cat, "Risk Description": desc,
                          "Signal": "Blocker" if is_blocker else ("Decision" if is_decision else "Risk"),
                          "Owner": (csv or {}).get("Assignee") or self.DEFAULT_OWNER[cat],
                          "Mitigation Strategy": self.MITIGATION[cat], **cite})

        risks = _dedupe(risks, "Risk Description", rank_key="Severity", fuzzy=True)
        risks.sort(key=lambda r: -SEVERITY_WEIGHT[r["Severity"]])
        seen_pos: Dict[tuple, int] = {}
        for i, r in enumerate(risks, start=1):
            p, imp = SEVERITY_BASE[r["Severity"]]
            n = seen_pos.get((p, imp), 0)
            seen_pos[(p, imp)] = n + 1
            step = (n + 1) // 2
            r["Risk ID"] = f"RSK-{i:02d}"
            r["Probability"] = round(min(0.98, max(0.05, p + ((-1) ** n) * 0.04 * step)), 2)
            r["Impact"] = round(min(5.0, max(1.0, imp + ((-1) ** (n + 1)) * 0.15 * step)), 2)
        cols = ("Risk ID", "Severity", "Risk Category", "Risk Description", "Probability", "Impact",
                "Mitigation Strategy", "Owner", "Signal", "Source", "Location", "Also In")
        risks = [{k: r.get(k, "") for k in cols} for r in risks]
        deps = _dedupe(deps, "Dependency / Prerequisite", fuzzy=True)
        return {"risks": risks, "dependencies": deps, "forecast": self._forecast(risks, deps, milestones, all_text)}

    def _forecast(self, risks, deps, milestones, all_text) -> Dict[str, Any]:
        crit = sum(r["Severity"] == "Critical" for r in risks)
        high = sum(r["Severity"] == "High" for r in risks)
        index = sum(SEVERITY_WEIGHT[r["Severity"]] for r in risks) + len(deps)
        status = "CRITICAL RISK" if crit >= 3 or index >= 30 else "AT RISK" if crit or high >= 2 or index >= 10 else "ON TRACK"
        weeks = [int(w) for w in re.findall(r"(\d+)[- ]weeks?", " ".join(all_text), re.I)]
        if weeks:
            slip = f"~{max(weeks)} week(s)"
        else:
            slip = "None expected" if status == "ON TRACK" else ("~1 week" if status == "AT RISK" else "~2+ weeks")
        forecasts = []
        for m in milestones:
            days = _days_until(m.get("Target Date", ""))
            if m.get("Status") == "Completed":
                fc = "Delivered"
            elif days is not None and days < 0:
                fc = "Overdue"
            elif status != "ON TRACK" and m.get("Status") == "In Progress":
                fc = "At Risk"
            else:
                fc = "On Track"
            forecasts.append({"Milestone": m.get("Milestone / Phase", ""), "Target Date": m.get("Target Date", "TBD"),
                              "Current Status": m.get("Status", ""), "Forecast": fc})
        dates = sorted(m["Target Date"] for m in milestones if DATE_RE.fullmatch(str(m.get("Target Date", ""))))
        recs = []
        if crit:
            top = next(r for r in risks if r["Severity"] == "Critical")
            recs.append(f"Resolve {crit} critical risk(s) first, starting with {top['Risk ID']}: {top['Risk Description']} {_cite(top)}.")
        if deps:
            recs.append(f"Escalate {len(deps)} dependency gap(s) to external owners and agree committed dates.")
        if any(r["Risk Category"] == "Resource" for r in risks):
            recs.append("Address resource shortages before the next sprint to protect milestone dates.")
        if any(r["Risk Category"] == "Budget" for r in risks):
            recs.append("Get a sponsor decision on the budget overrun so infrastructure work is not stalled.")
        at_risk = [f["Milestone"] for f in forecasts if f["Forecast"] in ("At Risk", "Overdue")]
        if at_risk:
            recs.append(f"Re-baseline at-risk milestones ({', '.join(at_risk)}) and communicate revised dates.")
        if not recs:
            recs.append("Delivery is on track — keep weekly risk reviews in place.")
        return {"health_status": status, "estimated_slippage": slip, "high_risk_count": crit + high,
                "risk_index": index, "milestone_forecasts": forecasts, "recommendations": recs,
                "target_date": dates[-1] if dates else "TBD"}


# ─────────────────────────────────────────────────────────────
#  Blocker & Action Item Agent
# ─────────────────────────────────────────────────────────────
class BlockerActionItemAgent:
    """Finds blockers, pending decisions, and action items in any wording.

    Layers: explicit "Blocker:/Action Item:/Pending Decision:" lines and CSV columns, section headings
    (Issues, Blockers, Decisions, Next Steps, Action Items...), and phrasing ("blocked by", "unable to",
    "need to decide", "<Name> will/to <task>").
    """
    BLOCK_RE = re.compile(r"\b(blocked|blocking|cannot (start|proceed|continue|access|deploy)|can't (start|proceed|access)|"
                          r"unable to|stuck|impediment|showstopper|on hold|outage|not working|is down|keeps (failing|crashing|timing out))\b", re.I)
    WAIT_RE = re.compile(r"\b(waiting (on|for)|awaiting|pending)\b", re.I)
    DECISION_RE = re.compile(r"\b(needs? to decide|decision (is )?(needed|required|pending)|to be decided|\btbd\b|"
                             r"awaiting (approval|sign-?off|confirmation)|pending (approval|sign-?off)|yet to (decide|approve|confirm)|"
                             r"whether (to|we)|open question)\b", re.I)
    ACTION_RE = re.compile(r"^(?:action|todo|to-do|ai|next step|follow[- ]up)\s*[:\-]\s*(.+)$", re.I)
    WILL_RE = re.compile(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?|@\w+)\s+(?:will|to|should|must|needs to|is to)\s+([a-z].+)$")
    PRONOUNS = {"we", "they", "it", "this", "that", "he", "she", "i", "you", "there", "these", "those", "the", "our", "all"}
    SECTIONS = [
        ("decisions", ("decision", "open question", "approval", "to be decided")),
        ("actions", ("action", "next step", "to do", "to-do", "todo", "follow-up", "follow up", "assigned", "tasks")),
        ("blockers", ("blocker", "impediment", "issue", "challenge", "problem", "obstacle", "roadblock", "concern")),
    ]
    EXPLICIT = ("action item:", "pending decision:", "blocker:", "issue:", "blocked by:")

    def _section(self, heading: str) -> str:
        h = heading.lower()
        for name, keys in self.SECTIONS:
            if any(k in h for k in keys):
                return name
        return "other"

    def _action(self, line: str, cite: Dict[str, str], in_section: bool) -> Optional[Dict[str, Any]]:
        text, owner = line, ""
        m = self.ACTION_RE.match(line)
        w = self.WILL_RE.match(line)
        if m:
            text = m.group(1)
        elif w and (w.group(1).lower() not in self.PRONOUNS or in_section):
            owner = "" if w.group(1).lower() in self.PRONOUNS else w.group(1).lstrip("@")
            text = w.group(2)
        elif not in_section:
            return None
        text = re.split(r"\.\s+(?:Assigned|Owner|Priority|Due|Status)\b", text)[0]
        owner = owner or _field(line, "Assigned to") or _field(line, "Owner") or ScopeDeliverableAgent()._owner(line) or "Unassigned"
        text = re.sub(r"\s+(?:by|before|on)\s+" + re.escape(_find_date(text)) + r".*$", "", text) if _find_date(text) else text
        text = text.strip().rstrip(".")
        if len(text) < 6:
            return None
        return {"Action Item": text[0].upper() + text[1:], "Assignee": owner,
                "Priority": (_field(line, "Priority") or ("High" if re.search(r"\b(urgent|asap|critical)\b", line, re.I) else "Medium")).title(),
                "Status": _norm_status(_field(line, "Status")), "Due Date": _find_date(line), **cite}

    def _free_text(self, records, blockers, decisions, actions):
        scope = ScopeDeliverableAgent()
        for rec in records:
            text, src, loc = rec.get("text", ""), rec.get("source", ""), rec.get("location", "")
            if _csv_fields(text) or ScopeDeliverableAgent._kv(text.strip()):
                continue
            cite = {"Source": src, "Location": loc}
            section = _BLK_STATE.get(src, "other")
            for line, bullet in scope._lines(text):
                low = line.lower()
                own_head = (not bullet and len(line.split()) <= 6 and not line.endswith(".")
                            and self._section(line.rstrip(":")) != "other")
                if own_head or (scope._heading(line, bullet) is not None and not line.endswith(".")):
                    section = self._section(line)
                    continue
                if len(line) < 10 or low.startswith(self.EXPLICIT) or scope.RISK_LABEL.match(low) and section == "other":
                    continue
                if section == "blockers" or self.BLOCK_RE.search(line) or (section != "decisions" and self.WAIT_RE.search(line) and bullet and section == "blockers"):
                    if section in ("actions", "decisions") and not self.BLOCK_RE.search(line):
                        pass
                    else:
                        blockers.append({"Blocker": re.sub(r"^(issue|problem|challenge)\s*[:\-]\s*", "", line, flags=re.I).rstrip("."),
                                         "Severity": "Critical" if re.search(r"\b(critical|outage|showstopper|is down)\b", low) else "High",
                                         "Owner": "", "Status": "Unresolved", "Notes": "", **cite})
                        continue
                if section == "decisions" or self.DECISION_RE.search(line):
                    if section == "decisions" or not self.WILL_RE.match(line):
                        decisions.append({"Decision": re.sub(r"^decision\s*[:\-]\s*", "", line, flags=re.I).rstrip(".?") ,
                                          "Decision Owner": _field(line, "Owner") or "TBD", "Due Date": _find_date(line), "Notes": "", **cite})
                        continue
                a = self._action(line, cite, section == "actions")
                if a:
                    actions.append(a)
            _BLK_STATE[src] = section
        _BLK_STATE.clear()

    def analyze(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        blockers, decisions, actions = [], [], []
        for line, src, loc, csv in _segments(records):
            cite = {"Source": src, "Location": loc}
            low = line.lower()
            if csv:
                cat = csv.get("Category", "").lower()
                if cat == "blocker":
                    blockers.append({"Blocker": line, "Severity": csv.get("Priority", "High"),
                                     "Owner": csv.get("Assignee", ""), "Status": csv.get("Status", "Unresolved"),
                                     "Notes": csv.get("Notes", ""), **cite})
                elif cat == "pending decision":
                    decisions.append({"Decision": line, "Decision Owner": csv.get("Assignee", ""),
                                      "Due Date": csv.get("Due Date", ""), "Notes": csv.get("Notes", ""), **cite})
                elif cat == "action item":
                    actions.append({"Action Item": line, "Assignee": csv.get("Assignee", "Unassigned"),
                                    "Priority": csv.get("Priority", "Medium"), "Status": _norm_status(csv.get("Status", "")),
                                    "Due Date": csv.get("Due Date", ""), **cite})
                continue
            row = ScopeDeliverableAgent._kv(line)
            if row:
                kind = (row.get("type") or row.get("category") or "").lower()
                name = ScopeDeliverableAgent._pick(row, ScopeDeliverableAgent.NAME_KEYS) or line
                owner = ScopeDeliverableAgent._pick(row, ScopeDeliverableAgent.OWNER_KEYS)
                due = ScopeDeliverableAgent._pick(row, ScopeDeliverableAgent.DATE_KEYS)
                status = row.get("status", "")
                if "block" in status.lower() or "blocker" in kind:
                    blockers.append({"Blocker": name, "Severity": (row.get("priority") or "High").title(), "Owner": owner,
                                     "Status": "Unresolved", "Notes": row.get("notes", ""), **cite})
                elif "decision" in kind:
                    decisions.append({"Decision": name, "Decision Owner": owner or "TBD", "Due Date": due,
                                      "Notes": row.get("notes", ""), **cite})
                elif any(k in kind for k in ("task", "action", "todo", "to-do")):
                    actions.append({"Action Item": name, "Assignee": owner or "Unassigned",
                                    "Priority": (row.get("priority") or "Medium").title(),
                                    "Status": _norm_status(status), "Due Date": due, **cite})
                continue
            if low.startswith("action item:"):
                actions.append({"Action Item": _head(line, "Action Item"),
                                "Assignee": _field(line, "Assigned to") or "Unassigned",
                                "Priority": _field(line, "Priority") or "Medium",
                                "Status": _norm_status(_field(line, "Status")),
                                "Due Date": _field(line, "Due Date"), **cite})
            elif low.startswith("pending decision:"):
                decisions.append({"Decision": _head(line, "Pending Decision"),
                                  "Decision Owner": _field(line, "Decision Owner") or "TBD",
                                  "Due Date": "", "Notes": "", **cite})
            elif low.startswith(("blocker:", "issue:", "blocked by:")):
                blockers.append({"Blocker": _head(line, r"(Blocker|Issue|Blocked by)"),
                                 "Severity": (_field(line, "Severity") or ("Critical" if "critical" in low else "High")).title(),
                                 "Owner": "", "Status": _field(line, "Status") or "Unresolved", "Notes": "", **cite})
        self._free_text(records, blockers, decisions, actions)
        if not actions:
            for d in ScopeDeliverableAgent().analyze(records)["deliverables"]:
                if d["Status"] != "Completed":
                    actions.append({"Action Item": d["Deliverable"], "Assignee": d["Owner"], "Priority": "Medium",
                                    "Status": "In Progress" if d["Status"] == "In Progress" else "Open",
                                    "Due Date": d["Target Completion"] if d["Target Completion"] != "TBD" else "",
                                    "Derived From": "Open deliverable", "Source": d["Source"], "Location": d["Location"]})
        blockers = [b for b in _dedupe(blockers, "Blocker", rank_key="Severity", fuzzy=True) if b["Status"].lower() != "resolved"]
        decisions = _dedupe(decisions, "Decision")
        actions = _dedupe(actions, "Action Item")
        for a in actions:
            days = _days_until(a["Due Date"])
            a["Overdue"] = "Yes" if days is not None and days < 0 and a["Status"] != "Completed" else "No"
        actions.sort(key=lambda a: (a["Status"] == "Completed", SEVERITY_ORDER.index(a["Priority"]) if a["Priority"] in SEVERITY_ORDER else 9))
        return {
            "blockers": blockers, "decisions": decisions, "action_items": actions,
            "summary": {"total_blockers": len(blockers), "total_decisions": len(decisions),
                        "total_action_items": len(actions),
                        "open_action_items": sum(a["Status"] != "Completed" for a in actions),
                        "overdue_action_items": sum(a["Overdue"] == "Yes" for a in actions)},
        }


# ─────────────────────────────────────────────────────────────
#  Documentation Generation Agent
# ─────────────────────────────────────────────────────────────
class DocumentationGenerationAgent:
    PERSONAS = [
        (("risk", "forecast"), "delivery manager", "I can act on schedule risks before milestones slip"),
        (("blocker", "action item", "decision"), "team lead", "nothing blocking the team goes unnoticed or unowned"),
        (("report", "dashboard", "executive", "demo"), "executive sponsor", "I get a trustworthy status view without manual reporting"),
        (("ingest", "rag", "index", "document", "extract"), "project manager", "project knowledge is searchable and up to date"),
    ]

    def _persona(self, text: str):
        t = text.lower()
        for keys, role, benefit in self.PERSONAS:
            if any(k in t for k in keys):
                return role, benefit
        return "project manager", "the project delivers on its committed scope"

    def analyze(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        scope = ScopeDeliverableAgent().analyze(records)
        risk = RiskForecastingAgent().analyze(records, scope["milestones"])
        blk = BlockerActionItemAgent().analyze(records)

        stories = []
        def add(feature: str, owner: str, priority: str, criteria: str, src: str):
            role, benefit = self._persona(feature)
            feature = feature.rstrip(".")
            stories.append({"Story ID": f"US-{len(stories) + 1:02d}",
                            "User Story": f"As a {role}, I want {feature[0].lower() + feature[1:]} so that {benefit}.",
                            "Priority": priority, "Owner": owner, "Acceptance Criteria": criteria, "Source": src})

        for d in scope["deliverables"]:
            add(f"a working {d['Deliverable']}", d["Owner"], "High",
                f"Given the uploaded documents, when the {d['Deliverable']} runs, then results are shown with source citations; "
                f"delivered by {d['Target Completion']}.", d["Source"])
        for g in scope["goals"]:
            add(f"to {g['Goal / Objective']}", "Project Manager", "Medium",
                f"Progress toward '{g['Goal / Objective'][:60]}' is measured and reported in each governance review.", g["Source"])
        for b in blk["blockers"]:
            add(f"the blocker '{b['Blocker']}' resolved", b["Owner"] or "Project Manager", b["Severity"],
                "Root cause identified, fix verified in staging, and blocker status set to Resolved.", b["Source"])

        register = [{"Risk ID": r["Risk ID"], "Description": r["Risk Description"], "Category": r["Risk Category"],
                     "Severity": r["Severity"], "Probability": r["Probability"], "Impact": r["Impact"],
                     "Score": round(r["Probability"] * r["Impact"], 2),
                     "Response": "Escalate" if r["Severity"] == "Critical" else "Mitigate" if r["Severity"] in ("High", "Medium") else "Monitor",
                     "Mitigation": r["Mitigation Strategy"], "Owner": r["Owner"], "Status": "Open", "Source": r["Source"]}
                    for r in risk["risks"]]
        actions = [{"#": i, **{k: a[k] for k in ("Action Item", "Assignee", "Priority", "Status", "Due Date", "Overdue", "Source")}}
                   for i, a in enumerate(blk["action_items"], start=1)]
        return {"user_stories": stories, "risk_register": register, "action_items": actions,
                "full_report": self._report(scope, risk, blk, stories, register, actions)}

    @staticmethod
    def _table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
        if not rows:
            return "_None identified._\n"
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |" for r in rows]
        return "\n".join(out) + "\n"

    def _report(self, scope, risk, blk, stories, register, actions) -> str:
        fc, s = risk["forecast"], blk["summary"]
        health = ProjectHealthScorer().calculate_health_score([], scope, risk, blk)
        parts = [
            "# Project Intelligence & Governance Report\n",
            f"_Generated {date.today().isoformat()} by the AI Project Intelligence & Risk Advisor._\n",
            "## Executive Summary\n",
            f"- Project health: **{health['overall_score']}/100 — {health['health_status']}**",
            f"- Delivery forecast: **{fc['health_status']}** (estimated slippage: {fc['estimated_slippage']}, final target {fc['target_date']})",
            f"- Risks detected: {len(register)} ({fc['high_risk_count']} high/critical)",
            f"- Unresolved blockers: {s['total_blockers']} · Pending decisions: {s['total_decisions']}",
            f"- Action items: {s['total_action_items']} ({s['open_action_items']} open, {s['overdue_action_items']} overdue)\n",
            "## Goals & Objectives\n", self._table(scope["goals"], ["Goal / Objective", "Category", "Source"]),
            "## Milestones & Forecast\n", self._table(fc["milestone_forecasts"], ["Milestone", "Target Date", "Current Status", "Forecast"]),
            "## Deliverables\n", self._table(scope["deliverables"], ["Deliverable", "Owner", "Target Completion"]),
            "## User Stories\n", self._table(stories, ["Story ID", "User Story", "Priority", "Owner"]),
            "## Risk Register\n", self._table(register, ["Risk ID", "Severity", "Category", "Description", "Score", "Response", "Owner", "Mitigation"]),
            "## Dependency Gaps\n", self._table(risk["dependencies"], ["Dependency / Prerequisite", "Type", "Source"]),
            "## Unresolved Blockers\n", self._table(blk["blockers"], ["Blocker", "Severity", "Status", "Source"]),
            "## Pending Decisions\n", self._table(blk["decisions"], ["Decision", "Decision Owner", "Source"]),
            "## Action Items\n", self._table(actions, ["#", "Action Item", "Assignee", "Priority", "Status", "Due Date", "Overdue"]),
            "## Recommendations\n", "\n".join(f"- {r}" for r in fc["recommendations"] + health["recommendations"]) + "\n",
        ]
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
#  Project Health Scorer
# ─────────────────────────────────────────────────────────────
class ProjectHealthScorer:
    """Overall = 30% scope clarity + 40% timeline + 30% blocker impact. Every dimension is 0–100, higher = healthier."""

    def calculate_health_score(self, records, scope_data, risk_data, blocker_data) -> Dict[str, Any]:
        pos = {"scope": [], "timeline": [], "blockers": []}
        pen = {"scope": [], "timeline": [], "blockers": []}

        g, m, d, r = (len(scope_data[k]) for k in ("goals", "milestones", "deliverables", "responsibilities"))
        scope = 20 + min(g, 4) * 6 + min(m, 4) * 6 + min(d, 4) * 5 + min(r, 4) * 3
        for n, label in ((g, "goals"), (m, "milestones"), (d, "deliverables"), (r, "RACI roles")):
            if n:
                pos["scope"].append(f"{n} {label} documented")
            else:
                pen["scope"].append(f"No {label} documented — scope is unclear")

        risks = risk_data["risks"]
        crit = sum(x["Severity"] == "Critical" for x in risks)
        high = sum(x["Severity"] == "High" for x in risks)
        med = sum(x["Severity"] == "Medium" for x in risks)
        mf = risk_data["forecast"]["milestone_forecasts"]
        overdue = sum(f["Forecast"] == "Overdue" for f in mf)
        n_deps = len(risk_data["dependencies"])
        timeline = 100 - crit * 8 - high * 5 - med * 2 - overdue * 6 - n_deps * 2
        if crit: pen["timeline"].append(f"{crit} critical risk(s) threaten delivery dates (−{crit * 8})")
        if high: pen["timeline"].append(f"{high} high-severity risk(s) detected (−{high * 5})")
        if med: pen["timeline"].append(f"{med} medium-severity risk(s) (−{med * 2})")
        if overdue: pen["timeline"].append(f"{overdue} milestone(s) past target date (−{overdue * 6})")
        if n_deps: pen["timeline"].append(f"{n_deps} unresolved dependency gap(s) (−{n_deps * 2})")
        done = sum(f["Forecast"] == "Delivered" for f in mf)
        if done: pos["timeline"].append(f"{done} milestone(s) already delivered")
        if not crit and not high: pos["timeline"].append("No high or critical risks detected")

        s = blocker_data["summary"]
        blockers = 100 - s["total_blockers"] * 10 - s["total_decisions"] * 5 - s["open_action_items"] * 2
        if s["total_blockers"]: pen["blockers"].append(f"{s['total_blockers']} unresolved blocker(s) (−{s['total_blockers'] * 10})")
        if s["total_decisions"]: pen["blockers"].append(f"{s['total_decisions']} pending decision(s) (−{s['total_decisions'] * 5})")
        if s["open_action_items"]: pen["blockers"].append(f"{s['open_action_items']} open action item(s) (−{s['open_action_items'] * 2})")
        closed = s["total_action_items"] - s["open_action_items"]
        if closed: pos["blockers"].append(f"{closed} action item(s) completed")
        if not s["total_blockers"]: pos["blockers"].append("No active blockers")

        dims = {"scope_clarity": int(max(5, min(100, scope))),
                "timeline_risk": int(max(5, min(100, timeline))),
                "blocker_impact": int(max(5, min(100, blockers)))}
        overall = int(round(0.3 * dims["scope_clarity"] + 0.4 * dims["timeline_risk"] + 0.3 * dims["blocker_impact"]))
        if overall >= 75:
            status, badge = "HEALTHY / ON TRACK", "badge-green"
        elif overall >= 50:
            status, badge = "MODERATE RISK / AT RISK", "badge-orange"
        else:
            status, badge = "CRITICAL RISK", "badge-red"

        recs = []
        if crit or high: recs.append("Run a focused risk burn-down session on the top critical/high risks this week.")
        if s["total_blockers"]: recs.append("Assign a single owner to every unresolved blocker with a daily check-in.")
        if s["total_decisions"]: recs.append("Bring pending decisions to the steering committee for a dated resolution.")
        if s.get("overdue_action_items"): recs.append(f"Close or re-plan {s['overdue_action_items']} overdue action item(s).")
        if dims["scope_clarity"] < 70: recs.append("Strengthen scope documentation: goals, milestones, deliverables, and RACI.")
        if not recs: recs.append("Maintain the current cadence — project indicators are healthy.")

        weakest = min(dims, key=dims.get)
        summary = (f"Overall health {overall}/100 ({status}) = 30% scope clarity + 40% timeline + 30% blocker impact. "
                   f"Weakest dimension: {weakest.replace('_', ' ')} ({dims[weakest]}/100). "
                   f"Based on {len(records)} content units.")
        return {"overall_score": overall, "health_status": status, "badge_class": badge, "dimension_scores": dims,
                "positive_drivers": pos, "penalties": pen, "recommendations": recs, "summary_description": summary}


# ─────────────────────────────────────────────────────────────
#  Conversational Project Intelligence Assistant (grounded RAG)
# ─────────────────────────────────────────────────────────────
class ConversationalProjectIntelligenceAssistant:
    INTENTS = [
        ("scope_raci", ("responsib", "raci", "accountable", "who owns", "owner of", "in charge of")),
        ("project_health", ("on track", "health", "status", "progress", "schedule", "deadline", "behind")),
        ("risk_analysis", ("risk", "threat", "concern")),
        ("blocker_tracking", ("blocker", "blocked", "blocking", "impediment")),
        ("action_tracking", ("action item", "task", "to do", "todo", "assigned", "overdue")),
        ("decision_tracking", ("decision",)),
        ("scope_raci", ("responsible", "raci", "owner", "who ", "deliverable", "accountable")),
        ("scope_overview", ("goal", "objective", "milestone", "scope")),
    ]
    GENERIC = tokens("are we on track health status progress schedule what our biggest top main risks risk "
                     "active open blockers blocker action items item assigned team who responsible deliverables "
                     "deliverable owner raci pending decisions decision goals milestones scope project tell me show list "
                     "current overall there any the most")

    def _intent(self, q: str) -> str:
        q = q.lower() + " "
        for name, keys in self.INTENTS:
            if any(k in q for k in keys):
                return name
        return "document_retrieval"

    @staticmethod
    def _retrieve(query, chunks, vectorizer, matrix, top_k):
        if not chunks:
            return []
        if vectorizer is not None and matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
        else:
            q = tokens(query)
            scores = [len(q & tokens(c.text)) / (len(q) or 1) for c in chunks]
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:top_k]
        return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def _focus(self, items: List[Dict[str, Any]], fields: List[str], query: str) -> List[Dict[str, Any]]:
        """Narrow to items matching the specific (non-generic) terms of the question, if any."""
        specific = tokens(query) - self.GENERIC
        if not specific:
            return items
        hits = [it for it in items if specific & tokens(" ".join(str(it.get(f, "")) for f in fields))]
        return hits or items

    @staticmethod
    def _best_sentences(query: str, retrieved, limit: int = 3):
        q = tokens(query)
        cands = []
        for ch, score in retrieved:
            for sent in re.split(r"\n|(?<=[.!?])\s+", ch.text):
                sent = _clean(sent)
                if _is_noise(sent):
                    continue
                overlap = len(q & tokens(sent))
                if overlap:
                    cands.append((overlap + score, sent, ch))
        cands.sort(key=lambda c: -c[0])
        out, seen = [], set()
        for _, sent, ch in cands:
            if sent not in seen:
                seen.add(sent)
                out.append((sent, ch))
            if len(out) == limit:
                break
        return out

    @staticmethod
    def _src(item: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {"Source": item.get("Source", ""), "Location": item.get("Location", ""), "Reason": reason}

    def answer_query(self, query: str, records, chunks, vectorizer, matrix, top_k: int = 4,
                     scope_data=None, risk_data=None, blocker_data=None, health_data=None) -> Dict[str, Any]:
        intent = self._intent(query)
        retrieved = self._retrieve(query, chunks, vectorizer, matrix, top_k)
        lines, sources = [], []

        if intent == "project_health" and health_data and risk_data:
            fc = risk_data["forecast"]
            verdict = "No" if fc["health_status"] != "ON TRACK" else "Yes"
            lines.append(f"**{verdict} — the project is {fc['health_status']}.** Health score is "
                         f"**{health_data['overall_score']}/100 ({health_data['health_status']})** and estimated slippage is "
                         f"{fc['estimated_slippage']}.")
            if fc["milestone_forecasts"]:
                lines.append("\n**Milestone forecast:**")
                lines += [f"- {f['Milestone']} (target {f['Target Date']}): **{f['Forecast']}**" for f in fc["milestone_forecasts"][:5]]
            top = [r for r in risk_data["risks"] if r["Severity"] in ("Critical", "High")][:3]
            if top:
                lines.append("\n**What is driving this:**")
                for r in top:
                    lines.append(f"- {r['Severity']}: {r['Risk Description']} {_cite(r)}")
                    sources.append(self._src(r, f"{r['Severity']} risk {r['Risk ID']}"))
            lines.append(f"\n**Next step:** {fc['recommendations'][0]}")
        elif intent == "risk_analysis" and risk_data:
            focus = self._focus(risk_data["risks"], ["Risk Description", "Risk Category"], query)
            lines.append(f"**{len(risk_data['risks'])} risks detected** "
                         f"({risk_data['forecast']['high_risk_count']} high/critical). Top risks by severity:")
            for r in focus[:5]:
                lines.append(f"- **{r['Risk ID']} · {r['Severity']} · {r['Risk Category']}** — {r['Risk Description']} {_cite(r)}\n"
                             f"  - Mitigation: {r['Mitigation Strategy']} (owner: {r['Owner']})")
                sources.append(self._src(r, f"{r['Risk ID']} {r['Severity']} risk"))
        elif intent == "blocker_tracking" and blocker_data:
            focus = self._focus(blocker_data["blockers"], ["Blocker", "Notes"], query)
            lines.append(f"**{len(blocker_data['blockers'])} unresolved blocker(s):**")
            for b in focus:
                owner = f", owner: {b['Owner']}" if b["Owner"] else ""
                lines.append(f"- **[{b['Severity']}]** {b['Blocker']} — {b['Status']}{owner} {_cite(b)}")
                sources.append(self._src(b, "Unresolved blocker"))
        elif intent == "action_tracking" and blocker_data:
            acts = self._focus(blocker_data["action_items"], ["Action Item", "Assignee"], query)
            if "overdue" in query.lower():
                acts = [a for a in acts if a.get("Overdue") == "Yes"] or acts
            s = blocker_data["summary"]
            lines.append(f"**{s['total_action_items']} action item(s)** — {s['open_action_items']} not completed, "
                         f"{s.get('overdue_action_items', 0)} overdue:")
            for a in acts:
                due = f" · due {a['Due Date']}" if a["Due Date"] else ""
                flag = " · **OVERDUE**" if a.get("Overdue") == "Yes" else ""
                lines.append(f"- {a['Action Item']} — **{a['Assignee']}** · {a['Priority']} · {a['Status']}{due}{flag} {_cite(a)}")
                sources.append(self._src(a, f"Action item for {a['Assignee']}"))
        elif intent == "decision_tracking" and blocker_data:
            lines.append(f"**{len(blocker_data['decisions'])} pending decision(s):**")
            for d in self._focus(blocker_data["decisions"], ["Decision", "Notes"], query):
                lines.append(f"- {d['Decision']} — owner: {d['Decision Owner']} {_cite(d)}")
                sources.append(self._src(d, "Pending decision"))
        elif intent == "scope_raci" and scope_data:
            dels = self._focus(scope_data["deliverables"], ["Deliverable", "Owner"], query)
            lines.append("**Deliverable owners:**")
            for d in dels:
                lines.append(f"- {d['Deliverable']} — **{d['Owner']}** (target {d['Target Completion']}) {_cite(d)}")
                sources.append(self._src(d, f"Deliverable owned by {d['Owner']}"))
            if scope_data["responsibilities"]:
                lines.append("\n**RACI matrix:**")
                for r in scope_data["responsibilities"]:
                    lines.append(f"- {r['Role']} — *{r['RACI']}*: {r['Responsibility']} {_cite(r)}")
                    sources.append(self._src(r, f"RACI: {r['Role']}"))
        elif intent == "scope_overview" and scope_data:
            lines.append("**Goals:**")
            lines += [f"- {g['Goal / Objective']} {_cite(g)}" for g in scope_data["goals"]]
            lines.append("\n**Milestones:**")
            lines += [f"- {m['Milestone / Phase']} — {m['Target Date']} ({m['Status']}) {_cite(m)}" for m in scope_data["milestones"]]
            sources += [self._src(x, "Scope item") for x in scope_data["goals"] + scope_data["milestones"]]

        best = self._best_sentences(query, retrieved, limit=2 if lines else 3)
        if best:
            if not lines:
                lines.append("**From your project documents:**")
                lines += [f"- {sent} *[{ch.source} · {ch.location}]*" for sent, ch in best]
            else:
                lines.append("\n**Supporting evidence:**")
                lines += [f"> {sent} — *{ch.source}, {ch.location}*" for sent, ch in best]
            score_of = {id(ch): sc for ch, sc in retrieved}
            for sent, ch in best:
                sources.append({"Source": ch.source, "Location": ch.location,
                                "Reason": f"Retrieved chunk {ch.chunk_id}: “{sent[:70]}…”",
                                "Score": round(score_of.get(id(ch), 0.0), 3)})

        uniq, seen = [], set()
        for s in sources:
            k = (s["Source"], s["Location"], s["Reason"])
            if k not in seen:
                seen.add(k)
                uniq.append(s)

        grounded = bool(uniq)
        answer = "\n".join(lines) if grounded else (
            "I couldn't find information about that in the ingested project documents, so I won't guess. "
            "Try rephrasing, or upload documents that cover this topic.")
        top_score = max((s for _, s in retrieved), default=0.0)
        if not grounded:
            confidence = "None"
        elif intent != "document_retrieval":
            confidence = "High"
        else:
            confidence = "Medium" if top_score > 0.15 else "Low"
        return {"answer": answer, "grounded": grounded, "sources": uniq, "intent": intent, "confidence": confidence}
