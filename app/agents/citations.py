"""Citation validation + markdown rendering.

A report is only returned when every citation marker in the body resolves to a
real finding collected during the run. Validation failures produce structured
feedback for one writer retry; a second failure ends the run FAILED.
"""

from __future__ import annotations

import re

from ..state.schemas import Finding, ResearchReport

_MARKER = re.compile(r"\[(\d+)\]")


def validate_citations(report: ResearchReport,
                       findings: list[Finding]) -> list[str]:
    """Return a list of problems; empty means the report is valid."""
    problems: list[str] = []
    finding_ids = {f.finding_id for f in findings}
    finding_urls = {f.url for f in findings}

    bodies = " ".join(s.body_markdown for s in report.sections)
    markers = [int(m) for m in _MARKER.findall(bodies)]
    if not markers:
        problems.append("report body contains no citation markers")
    for n in markers:
        if n < 1 or n > len(report.citations):
            problems.append(f"marker [{n}] has no matching citation entry")
    for i, c in enumerate(report.citations, start=1):
        if c.marker != f"[{i}]":
            problems.append(f"citation {i} marker is {c.marker!r}, expected '[{i}]'")
        if c.finding_id not in finding_ids:
            problems.append(f"citation [{i}] references unknown finding {c.finding_id!r}")
        if c.url not in finding_urls:
            problems.append(f"citation [{i}] url {c.url!r} was not fetched this run")
    cited = {int(m) for m in _MARKER.findall(bodies) if 1 <= int(m) <= len(report.citations)}
    for i in range(1, len(report.citations) + 1):
        if i not in cited:
            problems.append(f"citation [{i}] is never referenced in the body")
    return problems


def render_markdown(report: ResearchReport) -> str:
    lines = [f"# {report.title}", "", report.summary, ""]
    for s in report.sections:
        lines += [f"## {s.heading}", "", s.body_markdown, ""]
    lines += ["## Sources", ""]
    for c in report.citations:
        lines.append(f"{c.marker} {c.title} — {c.url}")
    lines.append("")
    return "\n".join(lines)
