from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
from rapidfuzz import fuzz


@dataclass
class PromptQualityReport:
    has_role: bool
    has_format: bool
    has_constraints: bool
    clarity_score: int  # 0-100 (heuristic)
    notes: List[str]


ROLE_PATTERNS = [
    r"\byou are\b",
    r"\bact as\b",
    r"\brole:\b",
]

FORMAT_PATTERNS = [
    r"\boutput\b",
    r"\bformat\b",
    r"\bin (markdown|json|table|bullets)\b",
    r"\bsections?:\b",
]

CONSTRAINT_PATTERNS = [
    r"\bconstraints?\b",
    r"\brequirements?\b",
    r"\bdo not\b",
    r"\bmust\b",
    r"\bavoid\b",
]


def analyze_prompt(prompt: str) -> PromptQualityReport:
    p = (prompt or "").strip().lower()
    has_role = any(re.search(rx, p) for rx in ROLE_PATTERNS)
    has_format = any(re.search(rx, p) for rx in FORMAT_PATTERNS)
    has_constraints = any(re.search(rx, p) for rx in CONSTRAINT_PATTERNS)

    notes = []
    base = 50
    if has_role: base += 15
    else: notes.append("Missing explicit role (e.g., 'You are a ...').")
    if has_format: base += 15
    else: notes.append("Missing explicit output format (e.g., sections/table/JSON).")
    if has_constraints: base += 10
    else: notes.append("Missing constraints (scope, assumptions, safety, length).")

    # encourage specificity
    length = len(p)
    if length < 60:
        base -= 10
        notes.append("Prompt is very short; likely underspecified.")
    elif length > 800:
        base -= 5
        notes.append("Prompt is very long; consider making it tighter.")

    clarity_score = max(0, min(100, base))
    return PromptQualityReport(has_role, has_format, has_constraints, clarity_score, notes)


def compare_outputs(reference: str, candidate: str) -> int:
    """
    Rough similarity score (0-100). This is NOT correctness.
    Useful for showing "how close" you are to dataset response.
    """
    ref = (reference or "").strip()
    cand = (candidate or "").strip()
    if not ref or not cand:
        return 0
    return int(fuzz.token_set_ratio(ref, cand))
